#pretrain.py
import torch
import argparse
import os
import time

from model import MiniscanRBBaseline, WordToNumber
from util import get_episode_generator, timeSince, get_supervised_batchsize, GenData, cuda_a_dict
#from agent import
from train import gen_samples, train_batched_step, eval_ll, batchtime


from deepcoderModel import DeepcoderRecognitionModel, deepcoder_training_step, dc_eval_ll
from scanPrimitives import buildBaseGrammar

parser = argparse.ArgumentParser()
parser.add_argument('--num_pretrain_episodes', type=int, default=100000, help='number of episodes for training')
parser.add_argument('--lr', type=float, default=0.001, help='ADAM learning rate', dest='adam_learning_rate')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers in the LSTM')
parser.add_argument('--max_length_eval', type=int, default=50, help='maximum generated sequence length when evaluating the network')
parser.add_argument('--emb_size', type=int, default=200, help='size of sequence embedding (also, nhidden for encoder and decoder LSTMs)')
parser.add_argument('--dropout_p', type=float, default=0.1, help=' dropout applied to embeddings and LSTMs')
parser.add_argument('--fn_out_model', type=str, default='', help='filename for saving the model')
parser.add_argument('--dir_model', type=str, default='out_models', help='directory for saving model files')
parser.add_argument('--episode_type', type=str, default='auto', help='what type of episodes do we want')
parser.add_argument('--batchsize', type=int, default=None )
parser.add_argument('--type', type=str, default="miniscanRBbase")
parser.add_argument('--use_saved_val', action='store_true', help='use saved validation problems')
parser.add_argument('--saved_val_path', type=str, default='miniscan_hard_saved_val.p')
parser.add_argument('--parallel', type=int, default=None)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=500)
parser.add_argument('--positional', action='store_true')
args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()
if __name__ == '__main__':
    #args stuff

    
    print("new model for random stuff...")
    if args.type == 'miniscanRBbase':
        model = MiniscanRBBaseline.new(args)
    elif args.type == 'WordToNumber':
        model = WordToNumber.new(args)
    else:
        assert False, "not implemented yet"


    path = os.path.join(args.dir_model, args.fn_out_model)
    if os.path.isfile(path):
        dc_model = torch.load(path)
    else:
        g = buildBaseGrammar(model.input_lang, model.output_lang) #TODO
        dc_model = DeepcoderRecognitionModel(model.encoder, g, inputDimensionality=args.emb_size)   
        dc_model.num_pretrain_episodes = args.num_pretrain_episodes
    dc_model.train()

    if args.num_pretrain_episodes > dc_model.num_pretrain_episodes:
        dc_model.num_pretrain_episodes = args.num_pretrain_episodes
    generate_episode_train, _, _, _, _ = get_episode_generator(model.episode_type)

    if hasattr(dc_model, 'samples_val'):
        samples_val = model.samples_val = dc_model.samples_val
    else:
        dc_model.samples_val = samples_val = model.samples_val

    val_states = []
    for s in samples_val:
        states, rules = model.sample_to_statelist(s)
        #for state, rule in zip(states, rules):
        for i in range(len(rules)):
            val_states.append( model.state_rule_to_sample(states[i], rules[i] ) )

    if args.parallel:
        dataqueue = GenData(lambda: gen_samples(
                                generate_episode_train, model),
                                        batchsize=args.batchsize, n_processes=args.parallel)

    avg_train_loss = 0.
    counter = 0 # used to count updates since the loss was last reported
    start = time.time()
    for episode, samples in enumerate(
                            dataqueue.batchIterator() if args.parallel else
                            get_supervised_batchsize(
                            lambda: gen_samples(
                                generate_episode_train, model),
                                        batchsize=args.batchsize), model.pretrain_episode + 1):

        dc_model.pretrain_episode = episode
        if episode > dc_model.num_pretrain_episodes: break
        # Generate a random episode
        
        train_loss = deepcoder_training_step(samples, dc_model)
        
        avg_train_loss += train_loss
        counter += 1
         
        if episode == 1 or episode % args.print_freq == 0 or episode == dc_model.num_pretrain_episodes:
            val_loss = dc_eval_ll(val_states, dc_model) #TODO
            print('{:s} ({:d} {:.0f}% finished) TrainLoss: {:.4f}, ValLoss: {:.4f}'.format(timeSince(start, float(episode) / float(dc_model.num_pretrain_episodes)),
                                 episode, float(episode) / float(dc_model.num_pretrain_episodes) * 100., avg_train_loss/counter, val_loss), flush=True)
            avg_train_loss = 0.
            counter = 0

            print('gen sample stats', batchtime.items())
            batchtime['max']=0
            batchtime['mean']=0
            batchtime['count']=0
            if episode % args.save_freq == 0 or episode == dc_model.num_pretrain_episodes:
                torch.save(dc_model, path)
            if episode % 10000 == 0 or episode == dc_model.num_pretrain_episodes:
                torch.save(dc_model, path+'_'+str(dc_model.pretrain_episode))

