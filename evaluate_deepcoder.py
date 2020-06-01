#evaluate.py

import torch
import argparse
import os
import time

import math

import dill

from model import MiniscanRBBaseline, WordToNumber 
from util import get_episode_generator, timeSince, tabu_update, cuda_a_dict, get_supervised_batchsize
from train import gen_samples, train_batched_step, eval_ll
from test import batched_test_with_sampling
from generate_episode import exact_perm_doubled_rules
from interpret_grammar import Grammar

from collections import namedtuple

from scanPrimitives import buildBaseGrammar
from test_deepcoder import test_deepcoder


SearchResult = namedtuple("SearchResult", "hit solution stats")


def compute_val_ll(model, samples_val=None):
    if samples_val==None: 
        samples_val = model.samples_val[:N_TEST_NEW]
    model.val_states = []
    for s in model.samples_val[:N_TEST_NEW]:
        states, rules = model.sample_to_statelist(s)
        for i in range(len(rules)):
            model.val_states.append( model.state_rule_to_sample(states[i], rules[i]) )

    val_loss = eval_ll(model.val_states, model)
    return val_loss

if __name__=='__main__':

    #args
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length_eval', type=int, default=15, 
        help='maximum generated sequence length when evaluating the network')
    parser.add_argument('--max_num_rules', type=int, default=12, help='maximum generated num rules')
    parser.add_argument('--fn_out_model', type=str, default='', help='filename for saving the model')
    parser.add_argument('--dir_model', type=str, default='out_models', help='directory for saving model files')
    parser.add_argument('--gpu', type=int, default=0, help='set which GPU we want to use')
    parser.add_argument('--batchsize', type=int, default=64 )
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--mode', type=str, default='sample', choices=['smc', 'sample']) #beam, 
    parser.add_argument('--type', type=str, default="miniscanRBbase")
    parser.add_argument('--savefile', type=str, default="results/dc.p")
    parser.add_argument('--use_large_support', action='store_true')
    parser.add_argument('--use_rules_hard', action='store_true')
    parser.add_argument('--use_scan_large_s', action='store_true')
    parser.add_argument('--new_test_ep', type=str, default='')
    parser.add_argument('--load_data', type=str, default='')
    parser.add_argument('--val_ll', action='store_true')
    parser.add_argument('--val_ll_only', action='store_true')
    parser.add_argument('--n_test', type=int, default=50)
    parser.add_argument('--duplicate_test', action='store_true')
    parser.add_argument('--positional', action='store_true', default=True) # positional rule encodings
    parser.add_argument('--hack_gt_g', action='store_true')
    parser.add_argument('--nosearch', action='store_true')
    parser.add_argument('--partial_credit', action='store_true', default=True)
    parser.add_argument('--seperate_query', action='store_true')
    parser.add_argument('--human_miniscan', action='store_true')
    parser.add_argument('--enum_only', action='store_true')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--emb_size', type=int, default=200, help='size of sequence embedding (also, nhidden for encoder and decoder LSTMs)')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers in the LSTM')
    parser.add_argument('--dropout_p', type=float, default=0.1, help=' dropout applied to embeddings and LSTMs')
    parser.add_argument('--lr', type=float, default=0.001, help='ADAM learning rate', dest='adam_learning_rate')
    parser.add_argument('--use_saved_val', action='store_true', help='use saved validation problems')
    parser.add_argument('--num_pretrain_episodes', type=int, default=100000, help='number of episodes for training')
    parser.add_argument('--cegis', type=int, default=False, help='reduce number of examples to check')
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    args.episode_type = args.new_test_ep

    if args.val_ll_only: args.val_ll = True
    path = os.path.join(args.dir_model, args.fn_out_model)
    filename = args.savefile 

    N_TEST_NEW = args.n_test
    # model = MiniscanModel.load('out_models/REPLMiniscan0')
    # samples_val = model.samples_val

    #load model
    # if args.type == 'miniscanRBbase':
    #     model = MiniscanRBBaseline.load(path)
    # elif args.type == 'WordToNumber':
    #     model = WordToNumber.load(path)
    # else:
    #     assert False, "not implemented yet"
    # #

    print("new model for random stuff...")
    if args.type == 'miniscanRBbase':
        model = MiniscanRBBaseline.new(args)
    elif args.type == 'WordToNumber':
        model = WordToNumber.new(args)
    else:
        assert False, "not implemented yet"


    g = buildBaseGrammar(model.input_lang, model.output_lang, model.prog_lang)
    path = os.path.join(args.dir_model, args.fn_out_model)
    if os.path.isfile(path):
        dc_model = torch.load(path)
        dc_model.grammar = g
    else: assert False

    if args.new_test_ep:
        print("generating new test examples")
        generate_episode_train, generate_episode_test, input_lang, output_lang, prog_lang = get_episode_generator(
                   args.new_test_ep, model_in_lang=model.input_lang,
                                    model_out_lang=model.output_lang,
                                    model_prog_lang=model.prog_lang)
        #model.tabu_episodes = set([])
        dc_model.samples_val = []
        for i in range(N_TEST_NEW):
            sample = generate_episode_test(model.tabu_episodes)
            if args.hack_gt_g: sample['grammar'] = Grammar( exact_perm_doubled_rules() , model.input_lang.symbols)
            dc_model.samples_val.append(sample)
            if not args.duplicate_test: model.tabu_episodes = tabu_update(model.tabu_episodes, sample['identifier'])

        model.input_lang = input_lang
        model.output_lang = output_lang
        if not args.val_ll_only:
            model.prog_lang = prog_lang

        dc_model.input_lang = input_lang
        dc_model.output_lang = output_lang
        if not args.val_ll_only:
            dc_model.prog_lang = prog_lang


    if args.load_data:
        if os.path.isfile(args.load_data):
            print('loading test data ... ')
            with open(args.load_data, 'rb') as h:
                test_samples = dill.load(h)
            dc_model.samples_val = test_samples
        else:
            print("no test data found, so saving current test data as new")
            with open(args.load_data, 'wb') as h:
                dill.dump(dc_model.samples_val, h) 
    
    if args.val_ll:
        val_ll = compute_val_ll(model)
        print("val ll:", val_ll)

    if args.val_ll_only: assert False

    #print("batchsize:", args.batchsize)
    count = 0
    results = []
    frac_exs_hits = []
    for j, sample in enumerate(dc_model.samples_val):
        if j == args.n_test: break

        print()
        print(f"Task {j+1} out of {len(dc_model.samples_val)}")
        print("ground truth grammar:")
        print(sample['identifier'])


        if args.human_miniscan:
            from miniscan_state import examples_train, examples_test
            examples = examples_train
            query_examples = examples_test
        else:
            examples, query_examples = None, None

        #val_states = []

        hit, solution, stats = test_deepcoder(sample, dc_model, model,
                                    examples=examples,
                                    query_examples=query_examples,
                                    symbols=model.input_lang.symbols,
                                    enum_only=args.enum_only,
                                    timeout=args.timeout,
                                    mdlIncrement=50,
                                    cegis=args.cegis)

        print(hit)
        print(solution)
        print("nodes:", stats['nodes_expanded'])
        print("frac hit", stats['fraction_query_hit'])

        # batched_test_with_sampling(sample, model, max_len=1 if 'RB' in args.type or 'Word' in args.type else 15, 
        #                             examples=examples,
        #                             query_examples=query_examples,
        #                             timeout=args.timeout,
        #                             verbose=True, 
        #                             min_len=0, 
        #                             batch_size=args.batchsize,
        #                             nosearch=args.nosearch,
        #                             partial_credit=args.partial_credit,
        #                             seperate_query=args.seperate_query,
        #                             max_rule_size= 100 if 'RB' in args.type or 'Word' in args.type else 15)


        #should be in stats
        frac_exs_hits.append(stats['fraction_query_hit'])

        if solution:
            if hit: print("SUCCESS!!!!!!!")
            print("found grammar:", flush=True)
            rules = solution.rules
            for r in rules: print(r)
            if hit: count +=1

        results.append( (sample , SearchResult(hit, solution, stats)) )

        with open(filename, 'wb') as savefile:
            dill.dump(results, savefile)
        print("prelim results file saved at", filename)


    print(f"HIT {count} out of {len(model.samples_val)}")
    avg = sum(frac_exs_hits)/len(frac_exs_hits)
    print(f"AVERAGE {avg*100} % examples")

    print(f"average nodes expanded: {sum(result.stats['nodes_expanded'] for samp, result in results )/len(results)}")
    variance = sum([(f - avg)**2 for f in frac_exs_hits ])/len(frac_exs_hits)
    print(f"standard error: {math.sqrt(variance)/math.sqrt(len(frac_exs_hits))*100}")
    #save results
