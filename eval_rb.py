#eval_rb.py
import dill
import copy
import time
from syntax_robustfill import SyntaxCheckingRobustFill

import torch
import argparse
import os
import time

from model import MiniscanRBBaseline, WordToNumber
from util import get_episode_generator, timeSince, get_supervised_batchsize, GenData, cuda_a_dict
#from agent import
from train import gen_samples, train_batched_step, eval_ll, batchtime
from generate_episode import exact_perm_doubled_rules
from interpret_grammar import Grammar

from test import Example, State, ParseError, UnfinishedError, REPLError

parser = argparse.ArgumentParser()
parser.add_argument('--num_pretrain_episodes', type=int, default=100000, help='number of episodes for training')
parser.add_argument('--lr', type=float, default=0.001, help='ADAM learning rate', dest='adam_learning_rate')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers in the LSTM')
parser.add_argument('--max_length_eval', type=int, default=50, help='maximum generated sequence length when evaluating the network')
parser.add_argument('--emb_size', type=int, default=200, help='size of sequence embedding (also, nhidden for encoder and decoder LSTMs)')
parser.add_argument('--dropout_p', type=float, default=0.1, help=' dropout applied to embeddings and LSTMs')
parser.add_argument('--fn_out_model', type=str, default='', help='filename for saving the model')
parser.add_argument('--dir_model', type=str, default='out_models', help='directory for saving model files')
parser.add_argument('--episode_type', type=str, default='scan_simple_original', help='what type of episodes do we want')
parser.add_argument('--batchsize', type=int, default=16 )
parser.add_argument('--type', type=str, default="miniscanRBbase")
parser.add_argument('--use_saved_val', action='store_true', help='use saved validation problems')
parser.add_argument('--save_path', type=str, default='robustfill_baseline.p')
parser.add_argument('--load_data', type=str, default='data/horules4.p')
parser.add_argument('--resultsfile', type=str, default='results/robustfill_test.p')
parser.add_argument('--parallel', type=int, default=None)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=50)
parser.add_argument('--save_old_freq', type=int, default=1000)
parser.add_argument('--positional', action='store_true')
parser.add_argument('--timeout', type=int, default=30)
parser.add_argument('--max_n_test', type=int, default=20)
parser.add_argument('--new_test_ep', type=str, default='')
args = parser.parse_args()
args.use_cuda = False #torch.cuda.is_available()

def tokenize_target_rule(rule): #ONLY FOR MINISCAN
    tokenized_rules = []
    rl = len(rule)
    for i, r in enumerate(rule):
        tokenized_rules.extend(r)
        if i+1 != rl: tokenized_rules.append('\n')
    return tokenized_rules

def g_to_target(full_g):
    full_rule_list = [str(r).split(' ') for r in full_g.rules]
    return tokenize_target_rule(full_rule_list)

if __name__ == '__main__':
    batchsize = args.batchsize
    #args stuff
    
    model = MiniscanRBBaseline.new(args)

    def check_candidate_support(sample, candidate):
        examples = {Example(cur, tgt) for cur, tgt in zip(sample['xs'], sample['ys']) }
        rules = model.detokenize_action(candidate)
        test_state = State(examples, rules)
        try:
            testout=model.REPL(test_state, None)
        except (ParseError, UnfinishedError, REPLError):
            #print("YOU ERRORED ON BEST GUESS")
            return 0.0

        return (len(test_state.examples) - len(testout.examples) )/len(test_state.examples)

    def check_candidate_query(sample, candidate):
        if candidate is None: return 0.0
        query_examples = {Example(cur, tgt) for cur, tgt in zip(sample['xq'], sample['yq']) if cur not in sample['xs'] }
        rules = model.detokenize_action(candidate)
        test_state = State(query_examples, rules)
        try:
            testout=model.REPL(test_state, None)
        except (ParseError, UnfinishedError, REPLError):
            print("YOU ERRORED ON BEST GUESS")
            return 0.0

        return (len(test_state.examples) - len(testout.examples) )/len(test_state.examples)



    path = os.path.join(args.dir_model, args.fn_out_model)

    generate_episode_train, generate_episode_test, input_lang, output_lang, prog_lang = get_episode_generator(args.episode_type)

    
    m = torch.load(args.save_path)
    m.cuda()
    m.max_length = 50

    with open(args.load_data, 'rb') as h:
        test_samples = dill.load(h)

    # if args.new_test_ep:
    #     print("generating new test examples")
    #     generate_episode_train, generate_episode_test, input_lang, output_lang, prog_lang = get_episode_generator(
    #                args.new_test_ep, model_in_lang=model.input_lang,
    #                                 model_out_lang=model.output_lang,
    #                                 model_prog_lang=model.prog_lang)
    #     #model.tabu_episodes = set([])
    #     test_samples = []
    #     for i in range(N_TEST_NEW):
    #         sample = generate_episode_test({})
    #         #model.samples_val.append(sample)
    #         #if not args.duplicate_test: model.tabu_episodes = tabu_update(model.tabu_episodes, sample['identifier'])

    #     model.input_lang = input_lang
    #     model.output_lang = output_lang
    #     if not args.val_ll_only:
    #         model.prog_lang = prog_lang

    # if args.load_data:
    #     if os.path.isfile(args.load_data):
    #         print('loading test data ... ')
    #         with open(args.load_data, 'rb') as h:
    #             test_samples = dill.load(h)
    #         #model.samples_val = test_samples
    #     else:
    #         print("no test data found, so saving current test data as new")
    #         with open(args.load_data, 'wb') as h:
    #             dill.dump(model.samples_val, h) 
    #         test_samples = model.samples_val

    def get_inputs_tgts(ep):
        #ep = generate_episode_test({})
        inputs = list(zip(ep['xs'], ep['ys'])) 
        tgt = g_to_target(ep['grammar'])
        return inputs, tgt

    def makeBatch(batchsize):
        #including padding
        inps, tgts = [], []
        for _ in range(batchsize):
            inp, tgt = get_inputs_tgts()
            inps.append(inp)
            tgts.append(tgt)  

        max_len = max(len(i) for i in inps)
        print(max_len)

        padded_inps = []
        for inp in inps:
            padded_inp = copy.deepcopy(inp)
            diff = max_len - len(inp)
            if diff > 0:
                for _ in range(diff):
                    print(padded_inp[-1])
                    padded_inp.append(padded_inp[-1])
            padded_inps.append(padded_inp)
        return padded_inps, tgts

    scores = []
    progs = []

    for i, ep in enumerate(test_samples):
        if i >= args.max_n_test: break
        print("testing on")
        print(ep['grammar'])
        best_support_score = 0
        best_prog = None
        start = time.time()
        hit_sup = False
        while time.time() - start < args.timeout and not hit_sup:
            inputs, tgt = get_inputs_tgts(ep)
            candidates = m.sample([inputs]*batchsize)
            for candidate in candidates:
                #print(candidate)
                sup_score = check_candidate_support(ep, candidate)
                #print(sup_score)
                if sup_score > best_support_score:
                    best_support_score = sup_score
                    best_prog = candidate
                if sup_score == 1.0: 
                    hit_sup = True
                    print("HIT A SUPPORT SET")
                    break

        query_score = check_candidate_query(ep, best_prog)
        #print('q score', query_score)
        scores.append(query_score)
        progs.append(best_prog)
        print("Score on this grammar:", query_score)


    avg = sum(scores)/len(scores)
    print(f"AVERAGE {avg*100} % examples")
    from scipy import stats
    print(f"STANDARD ERROR", stats.sem(scores)*100)

    results = (ep, scores, progs)
    with open(args.resultsfile, 'wb') as h:
        dill.dump(results, h)
    print('results saved at', args.resultsfile)