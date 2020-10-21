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
parser.add_argument('--batchsize', type=int, default=None )
parser.add_argument('--type', type=str, default="miniscanRBbase")
parser.add_argument('--use_saved_val', action='store_true', help='use saved validation problems')
parser.add_argument('--save_path', type=str, default='robustfill_baseline0.p')
parser.add_argument('--parallel', type=int, default=None)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=50)
parser.add_argument('--save_old_freq', type=int, default=1000)
parser.add_argument('--positional', action='store_true')
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
    #args stuff
    
    model = MiniscanRBBaseline.new(args)
    path = os.path.join(args.dir_model, args.fn_out_model)

    generate_episode_train, generate_episode_test, input_lang, output_lang, prog_lang = get_episode_generator(args.episode_type)

    try:
        m = torch.load(args.save_path)
        print('loaded model')
    except:
        print('didnt load model .. new one')
        m = SyntaxCheckingRobustFill((input_lang.symbols, output_lang.symbols), prog_lang.symbols, hidden_size=512, embedding_size=128, max_length=50)
        m.iter = 0
        m.cuda()

    #print(inputs)
    
    #m.sample([inputs])

    g = Grammar( exact_perm_doubled_rules() , input_lang.symbols)


    tgt = str(g).split()

    d = {
    'RTURN': 'I_TURN_RIGHT',
    'LTURN': 'I_TURN_LEFT',
    'RUN': 'I_RUN',
    'JUMP': 'I_JUMP',
    'LOOK': 'I_LOOK',
    'WALK': 'I_WALK',
    }

    tgt = [d.get(x, x) for x in tgt]
    print(tgt)

    batchsize = 16

    def get_inputs_tgts():
        ep = generate_episode_test({})
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
        #print(max_len)

        padded_inps = []
        for inp in inps:
            padded_inp = copy.deepcopy(inp)
            diff = max_len - len(inp)
            if diff > 0:
                for _ in range(diff):
                    #print(padded_inp[-1])
                    padded_inp.append(padded_inp[-1])
            padded_inps.append(padded_inp)
        return padded_inps, tgts



    t = time.time()
    for i in range(1, 1000000):
        #inps_tgts = [get_inputs_tgts() for _ in range(batchsize)]
        
        # inps, tgts = [], []
        # inp, tgt = get_inputs_tgts()
        # print(len(inp))
        # for _ in range(batchsize):   
        #     inps.append(inp)
        #     tgts.append(tgt)

        #m.optimiser_step([inp], [tgt] )
        
        inps, tgts = makeBatch(batchsize)
        score, syntax_score = m.optimiser_step(inps,tgts)
        m.iter += 1
        
        print(f"total time: {time.time() - t}, total num ex processed: {(i+1)*batchsize}, avg time per ex: {(time.time() - t)/((i+1)*batchsize)}, score: {score}")

        if i%args.save_freq==0:
            torch.save(m, args.save_path)
            print('saved model')
        if i%args.save_old_freq==0:
            torch.save(m, args.save_path+str(m.iter))




