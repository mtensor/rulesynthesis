from __future__ import print_function
import argparse
import random
import os
from copy import deepcopy, copy
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from metanet_attn import MetaNetRNN, AttnDecoderRNN, DecoderRNN, describe_model, WrapperEncoderRNN
from masked_cross_entropy import *
from interpret_grammar import Grammar, Rule
import dill
import util
from collections import defaultdict

USE_CUDA = torch.cuda.is_available()

# Special tokens
SOS_token = "SOS"
EOS_token = "EOS"
PAD_token = SOS_token
class Lang:

    def __init__(self, symbols):
        n = len(symbols)
        self.symbols = symbols
        self.index2symbol = {n: SOS_token, n+1: EOS_token}
        self.symbol2index = {SOS_token : n, EOS_token : n+1}
        for idx,s in enumerate(symbols):
            self.index2symbol[idx] = s
            self.symbol2index[s] = idx
        self.n_symbols = len(self.index2symbol)

    def variableFromSymbols(self, mylist, add_eos=True):
        # convert a list of symbols to variable of indices (adding a EOS token)
        mylist = copy(mylist)
        if add_eos:
            mylist.append(EOS_token)
        indices = [self.symbol2index[s] for s in mylist]
        output = torch.LongTensor(indices)
        if USE_CUDA:
            output = output.cuda()
        return output

    def symbolsFromVector(self, v):
        # convert indices to symbols, breaking where we get a EOS token
        mylist = []
        for x in v:
            s = self.index2symbol[x]
            if s == EOS_token:
                break
            mylist.append(s)
        return mylist

# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def make_hashable(G):
    # Transform grammar or list into a hashable string
    G_str = str(G).split('\n')
    G_str.sort()
    out = '\n'.join(G_str)
    return out.strip()

def tabu_update(tabu_list,identifier):
    # Add all elements of "identifier" to the 'tabu_list', and return updated list
    if isinstance(identifier,(list,set)):
        tabu_list = tabu_list.union(identifier)
    elif isinstance(identifier,str):
        tabu_list.add(identifier)
    else:
        assert False
    return tabu_list

def get_unique_words(sentences):
    # Get a list of all the unique words in a sentence
    # Input
    #  sentences: list of strings
    # Output
    #   words : list of all unique words in sentences
    words = []
    for s in sentences:
        for w in s.split(' '):
            if w not in words:
                words.append(w)
    return words

# pad sequence with the PAD_token symbol
def pad_seq(seq, max_length):
    # seq : list of symbols
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def build_padded_var(list_seq, lang):
    # Transform python list into a padded torch tensor
    # 
    # Input
    #  list_seq : python list of n sequences (each is a python list of symbols)
    #  lang : language object for translation
    #
    # Output
    #  z_padded : LongTensor (n x max_length)
    #  z_lenghts : python list of sequence lengths (list of scalars)
    n = len(list_seq)
    if n==0: return [],[]
    z_eos = [z+[EOS_token] for z in list_seq]
    z_lengths = [len(z) for z in z_eos]
    max_len = max(z_lengths)
    z_padded = [pad_seq(z, max_len) for z in z_eos]
    z_padded = [lang.variableFromSymbols(z, add_eos=False).unsqueeze(0) for z in z_padded]
    z_padded = torch.cat(z_padded,dim=0)
    if USE_CUDA:
        z_padded = z_padded.cuda()
    return z_padded,z_lengths

def build_sample(x_support,y_support,x_query,y_query,input_lang,output_lang,myhash,grammar=''):
    # convert lists to episode format
    sample = {}
    sample['identifier'] = myhash # unique identifier for this episode (order invariant)
    sample['xs'] = x_support # support 
    sample['ys'] = y_support
    sample['xq'] = x_query # query
    sample['yq'] = y_query
    sample['grammar'] = grammar
    
    # # create lists of tensors
    sample['xs_padded'],sample['xs_lengths'] = build_padded_var(x_support,input_lang) # (ns x max_length)
    sample['ys_padded'],sample['ys_lengths'] = build_padded_var(y_support,output_lang) # (ns x max_length)
    sample['xq_padded'],sample['xq_lengths'] = build_padded_var(x_query,input_lang) # (nq x max_length)
    sample['yq_padded'],sample['yq_lengths'] = build_padded_var(y_query,output_lang) # (nq x max_length)
    return sample

def extract(include,arr):
    # Create a new list only using the included elements of arr 
    #
    # Input
    #  include : [n len] boolean array
    #  arr [ n length array]
    assert len(include)==len(arr)
    return [a for idx,a in enumerate(arr) if include[idx]]

def evaluation_battery(sample_eval_list, encoder, decoder, input_lang, output_lang, max_length, verbose=False, strict_eval=False, length_generalization=False):
    # Evaluate a set of episodes
    #
    # Input 
    #   sample_eval_list : list of evaluation sets to iterate through
    #   max_length : maximum length of a generated sequence
    # 
    # Output
    #   (acc_novel, acc_autoencoder) average accuracy for novel items in query set, and support items in query set


    if length_generalization:
        accuracy = {}
        counts = defaultdict(int)
    else:
        accuracy = None
        counts = None

    list_acc_val_novel = []
    list_acc_val_autoencoder = []
    for idx,sample in enumerate(sample_eval_list):
        acc_val_novel, acc_val_autoencoder, yq_predict, in_support, all_attention_by_query, memory_attn_steps, accuracy, counts = evaluate(sample, encoder, decoder, input_lang, output_lang, max_length, accuracy, counts)
        if strict_eval: 
            if acc_val_novel != 100.:
                 acc_val_novel = 0.0
        list_acc_val_novel.append(acc_val_novel)
        list_acc_val_autoencoder.append(acc_val_autoencoder)

        mean_length_ys = np.mean( [len(y) for y in sample['ys']] )
        max_length_ys = max( [len(y) for y in sample['ys']] )
        # max_length_y = max( [len(y) for y in sample['ys']] )

        if verbose:
            print('')
            print('Evaluation episode ' + str(idx))
            if sample['grammar']:
                print("")
                print(sample['grammar'])
            print('  support items: ')
            display_input_output(sample['xs'],sample['ys'],sample['ys'])
            print('  retrieval items; ' + str(round(acc_val_autoencoder,3)) + '% correct')
            display_input_output(extract(in_support,sample['xq']),extract(in_support,yq_predict),extract(in_support,sample['yq']))
            print('  generalization items; ' + str(round(acc_val_novel,3)) + '% correct')
            display_input_output(extract(np.logical_not(in_support),sample['xq']),extract(np.logical_not(in_support),yq_predict),extract(np.logical_not(in_support),sample['yq']))
    
    from scipy import stats
    if type(accuracy)==dict:
        return np.mean(list_acc_val_novel), np.mean(list_acc_val_autoencoder), accuracy, max_length_ys, mean_length_ys, stats.sem(list_acc_val_novel), stats.sem(list_acc_val_autoencoder)
    else:
        return np.mean(list_acc_val_novel), np.mean(list_acc_val_autoencoder), stats.sem(list_acc_val_novel), stats.sem(list_acc_val_autoencoder)

def evaluate(sample, encoder, decoder, input_lang, output_lang, max_length, accuracy=None, counts=None):
    # Evaluate an episode
    # 
    # Input
    #   sample : generated validation episode
    #
    # Output
    #   acc_novel : accuracy on novel items in query set
    #   acc_autoencoder : accuracy of support items in query set
    #   yq_preidct : list of predicted output sequences for all items
    #   is_support : [n x 1 bool] indicates for each query item whether it is in the support set
    #   all_attn_by_time : list (over time step) of batch_size x max_input_length tensors
    #   memory_attn_steps : attention over support items at every step for each query [max_xq_length x nq x ns]
    encoder.eval()
    decoder.eval()

    # Run words through encoder
    encoder_embedding, dict_encoder = encoder(sample)
    encoder_embedding_steps = dict_encoder['embed_by_step']
    memory_attn_steps = dict_encoder['attn_by_step']
    
    # Prepare input and output variables
    nq = len(sample['yq'])
    decoder_input = torch.tensor([output_lang.symbol2index[SOS_token]]*nq) # nq length tensor
    decoder_hidden = decoder.initHidden(encoder_embedding)

    # Store output words and attention states
    decoded_words = []
    
    # Run through decoder
    all_decoder_outputs = np.zeros((nq, max_length), dtype=int)
    all_attn_by_time = [] # list (over time step) of batch_size x max_input_length tensors
    if USE_CUDA:
        decoder_input = decoder_input.cuda()    
    for t in range(max_length):
        if type(decoder) is AttnDecoderRNN:
            decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_embedding_steps)
            all_attn_by_time.append(attn_weights)
        elif type(decoder) is DecoderRNN:        
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        else:
            assert False
        
        # Choose top symbol from output
        topv, topi = decoder_output.cpu().data.topk(1)
        decoder_input = topi.view(-1)
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
        all_decoder_outputs[:,t] = topi.numpy().flatten()

    # get predictions
    in_support = np.array([x in sample['xs'] for x in sample['xq']])
    yq_predict = []
    for q in range(nq):
        myseq = output_lang.symbolsFromVector(all_decoder_outputs[q,:])
        if args.episode_type == 'wordToNumber':
            myseq = [int(''.join(myseq)) if myseq else float('-inf')] #hopefully this works
        yq_predict.append(myseq)
    
    # compute accuracy
    v_acc = np.zeros(nq)

    for q in range(nq):
        
        v_acc[q] = yq_predict[q] == sample['yq'][q]

        if accuracy is not None and not in_support[q]:
            length = len(sample['yq'][q])
            accuracy[length] = (accuracy.get(length, 0) * counts[length] + v_acc[q] )/ (counts[length]+1)
            counts[length] += 1

    acc_autoencoder = np.mean(v_acc[in_support])*100.
    acc_novel = np.mean(v_acc[np.logical_not(in_support)])*100.


    
    return acc_novel, acc_autoencoder, yq_predict, in_support, all_attn_by_time, memory_attn_steps, accuracy, counts

def train(sample, encoder, decoder, encoder_optimizer, decoder_optimizer, input_lang, output_lang):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder.train()
    decoder.train()

    # Run words through encoder
    encoder_embedding, dict_encoder = encoder(sample)
    encoder_embedding_steps = dict_encoder['embed_by_step']
    
    # Prepare input and output variables
    nq = len(sample['yq'])
    decoder_input = torch.tensor([output_lang.symbol2index[SOS_token]]*nq) # nq length tensor
    decoder_hidden = decoder.initHidden(encoder_embedding)
    target_batches = torch.transpose(sample['yq_padded'], 0, 1) # (max_length x nq tensor) ... batch targets with padding    
    target_lengths = sample['yq_lengths']
    max_target_length = max(target_lengths)
    all_decoder_outputs = torch.zeros(max_target_length, nq, decoder.output_size)
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        target_batches = target_batches.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()
    
    # Run through decoder one time step at a time
    for t in range(max_target_length):
        if type(decoder) is AttnDecoderRNN:
            decoder_output, decoder_hidden, attn_by_query = decoder(decoder_input, decoder_hidden, encoder_embedding_steps)
        elif type(decoder) is DecoderRNN:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        else:
            assert False
        all_decoder_outputs[t] = decoder_output # max_len x nq x output_size
        decoder_input = target_batches[t]

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        torch.transpose(all_decoder_outputs, 0, 1).contiguous(), # -> nq x max_length
        torch.transpose(target_batches, 0, 1).contiguous(), # nq x max_length
        target_lengths
    )

    # gradient update
    loss.backward()
    encoder_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    decoder_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    if encoder_norm > clip or decoder_norm > clip:
        print("Gradient clipped:")
        print("  Encoder norm: " + str(encoder_norm))
        print("  Decoder norm: " + str(decoder_norm))
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.cpu().item()

def display_input_output(input_patterns,output_patterns,target_patterns):
    # Verbose analysis of performance on query items
    # 
    # Input
    #   input_patterns : list of input sequences (each in list form)
    #   output_patterns : list of output sequences, which are actual outputs (each in list form)
    #   target_patterns : list of targets
    nq = len(input_patterns)
    if nq == 0:
        print('     no patterns')
        return
    for q in range(nq):
        assert isinstance(input_patterns[q],list)
        assert isinstance(output_patterns[q],list)
        is_correct = output_patterns[q] == target_patterns[q]        
        print('     ',end='')
        print(' '.join(input_patterns[q]),end='')
        print(' -> ',end='')
        print(' '.join(output_patterns[q]),end='')
        if not is_correct:
            print(' (** target: ',end='')
            print(' '.join(target_patterns[q]),end='')
            print(')',end='')
        print('')

def get_episode_generator(episode_type):
    #  Returns function that generates episodes, 
    #   and language class for the input and output language
    #
    # Input
    #  episode_type :
    #
    # Output
    #  generate_episode: function handle for generating episodes
    #  input_lang: Language object for input sequence
    #  output_lang: Language object for output sequence
    
    if episode_type in ['scan_random',
            'scan_length_original', 'scan_simple_original', 'scan_around_right_original', 'scan_jump_original',
            'wordToNumber', 'rules_gen'] or 'lang_' in episode_type:

        #todo: check that it uses 
        generate_episode_train, generate_episode_test, input_lang, output_lang, _ = util.get_episode_generator(episode_type)

    else:
        raise Exception("episode_type is not valid" )
    return generate_episode_train, generate_episode_test, input_lang, output_lang
        
if __name__ == "__main__":

    # Training parameters
    num_episodes_val = 5 # number of episodes to use as validation throughout learning
    clip = 50.0 # clip gradients with larger magnitude than this
    max_try_novel = 100 # number of attempts to find a novel episode (not in tabu list) before throwing an error
    
    # Adjustable parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=1000000, help='number of episodes for training')
    parser.add_argument('--lr', type=float, default=0.001, help='ADAM learning rate')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers in the LSTM')
    parser.add_argument('--max_length_eval', type=int, default=50, help='maximum generated sequence length when evaluating the network')
    parser.add_argument('--emb_size', type=int, default=200, help='size of sequence embedding (also, nhidden for encoder and decoder LSTMs)')
    parser.add_argument('--dropout', type=float, default=0.5, help=' dropout applied to embeddings and LSTMs')
    parser.add_argument('--fn_out_model', type=str, default='', help='filename for saving the model')
    parser.add_argument('--dir_model', type=str, default='out_models', help='directory for saving model files')
    parser.add_argument('--episode_type', type=str, default='auto', help='what type of episodes do we want')
    parser.add_argument('--disable_memory', action='store_true', help='Disable external memory, ignore support set, and use simple RNN encoder')
    parser.add_argument('--disable_attention', action='store_true', help='Disable the decoder attention')
    parser.add_argument('--disable_recon_loss', action='store_true', help='Disable reconstruction loss, where support items are included also as query items')
    parser.add_argument('--gpu', type=int, default=0, help='set which GPU we want to use')
    parser.add_argument('--new_test_ep', type=str, default='')
    parser.add_argument('--load_data', type=str, default='')
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--duplicate_test', action='store_true')
    parser.add_argument('--human_miniscan', action='store_true')
    args = parser.parse_args()
    fn_out_model = args.fn_out_model
    episode_type = args.episode_type
    dir_model = args.dir_model
    gpu_num = args.gpu
    emb_size = args.emb_size
    nlayers = args.nlayers
    num_episodes = args.num_episodes
    dropout_p = args.dropout
    adam_learning_rate = args.lr
    max_length_eval = args.max_length_eval
    disable_memory = args.disable_memory
    disable_recon_loss = args.disable_recon_loss
    use_resconstruct_loss = not disable_recon_loss
    use_attention = not args.disable_attention

    if fn_out_model=='':
        fn_out_model = 'net_'  + episode_type + '.tar'
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)
    fn_out_model = os.path.join(dir_model, fn_out_model)

    if not os.path.isfile(fn_out_model) or args.cont:
        if not args.cont:
            print("Training a new network...")
            print("  Episode type is " + episode_type)        
            generate_episode_train, generate_episode_test, input_lang,output_lang = get_episode_generator(episode_type)
            if USE_CUDA:
                torch.cuda.set_device(gpu_num)
                print('  Training on GPU ' + str(torch.cuda.current_device()), end='')
            else:
                print('  Training on CPU', end='')
            print(' for ' + str(num_episodes) + ' episodes')
            input_size = input_lang.n_symbols
            output_size = output_lang.n_symbols
            
            if disable_memory:
                encoder = WrapperEncoderRNN(emb_size, input_size, output_size, nlayers, dropout_p)
            else:
                encoder = MetaNetRNN(emb_size, input_size, output_size, nlayers, dropout_p)
            if use_attention:
                decoder = AttnDecoderRNN(emb_size, output_size, nlayers, dropout_p)
            else:
                decoder = DecoderRNN(emb_size, output_size, nlayers, dropout_p)
            if USE_CUDA:
                encoder = encoder.cuda()
                decoder = decoder.cuda()
            print('  Set learning rate to ' + str(adam_learning_rate))
            encoder_optimizer = optim.Adam(encoder.parameters(),lr=adam_learning_rate)
            decoder_optimizer = optim.Adam(decoder.parameters(),lr=adam_learning_rate)
            print("")
            print("Architecture options...")
            print(" Decoder attention is USED" ) if use_attention else print(" Decoder attention is NOT used")
            print(" External memory is USED" ) if not disable_memory else print(" External memory is NOT used")
            print(" Reconstruction loss is USED" ) if not disable_recon_loss else print(" Reconstruction loss is NOT used")
            print("")
            describe_model(encoder)
            describe_model(decoder)

            # create validation episodes
            tabu_episodes = set([])
            samples_val = []
            for i in range(num_episodes_val):
                sample = generate_episode_test(tabu_episodes)
                samples_val.append(sample)
                tabu_episodes = tabu_update(tabu_episodes,sample['identifier'])

            start_episode = 1
        else:
            generate_episode_train, generate_episode_test, input_lang,output_lang = get_episode_generator(episode_type)
            print('Loading model: ' + fn_out_model)        
            checkpoint = torch.load(fn_out_model, map_location='cpu') # evaluate model on CPU
            if 'episode' in checkpoint: print(' Loading epoch ' + str(checkpoint['episode']) + ' of ' + str(checkpoint['num_episodes']))
            input_lang = checkpoint['input_lang']
            output_lang = checkpoint['output_lang']
            emb_size = checkpoint['emb_size']
            nlayers = checkpoint['nlayers']
            dropout_p = checkpoint['dropout']
            input_size = input_lang.n_symbols
            output_size = output_lang.n_symbols
            samples_val = checkpoint['episodes_validation']
            tabu_episodes = set([])
            for sample in samples_val:
                tabu_episodes = tabu_update(tabu_episodes,sample['identifier'])
            disable_memory = checkpoint['disable_memory']
            max_length_eval = checkpoint['max_length_eval']
            if disable_memory:
                encoder = WrapperEncoderRNN(emb_size, input_size, output_size, nlayers, dropout_p)
            else:
                encoder = MetaNetRNN(emb_size, input_size, output_size, nlayers, dropout_p)        
            if use_attention:
                decoder = AttnDecoderRNN(emb_size, output_size, nlayers, dropout_p)
            else:
                decoder = DecoderRNN(emb_size, output_size, nlayers, dropout_p)
            if USE_CUDA:
                encoder = encoder.cuda()
                decoder = decoder.cuda()
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            print('  Set learning rate to ' + str(adam_learning_rate))
            encoder_optimizer = optim.Adam(encoder.parameters(),lr=adam_learning_rate)
            decoder_optimizer = optim.Adam(decoder.parameters(),lr=adam_learning_rate)
            describe_model(encoder)
            describe_model(decoder)

            start_episode = checkpoint['episode']
        # assert False
        samples_val = [ util.cuda_a_dict(s) for s in samples_val ]
        # train over a set of random episodes
        avg_train_loss = 0.
        counter = 0 # used to count updates since the loss was last reported
        start = time.time()
        for episode in range(start_episode, num_episodes+1):

            # Generate a random episode
            start_gen_epi = time.time()
            sample = generate_episode_train(tabu_episodes)
            # print("generate episode time: " + str(time.time()-start_gen_epi))
            # assert(sample['identifier']!=samples_val[0]['identifier'])
            
            # Batch updates (where batch includes the entire support set)
            start_update = time.time()
            sample = util.cuda_a_dict(sample)
            train_loss = train(sample, encoder, decoder, encoder_optimizer, decoder_optimizer, input_lang, output_lang)
            # print("network update time: " + str(time.time()-start_update))
            avg_train_loss += train_loss
            counter += 1

            if episode == 1 or episode % 1000 == 0 or episode == num_episodes:
                acc_val_gen, acc_val_retrieval, _, _ = evaluation_battery(samples_val, encoder, decoder, input_lang, output_lang, max_length_eval)
                print('{:s} ({:d} {:.0f}% finished) TrainLoss: {:.4f}, ValAccRetrieval: {:.1f}, ValAccGeneralize: {:.1f}'.format(timeSince(start, float(episode) / float(num_episodes)),
                                         episode, float(episode) / float(num_episodes) * 100., avg_train_loss/counter, acc_val_retrieval, acc_val_gen), flush=True)
                avg_train_loss = 0.
                counter = 0
                if episode % 1000 == 0 or episode == num_episodes:
                    state = {'encoder_state_dict': encoder.state_dict(),
                                'decoder_state_dict': decoder.state_dict(),
                                'input_lang': input_lang,
                                'output_lang': output_lang,
                                'episodes_validation': samples_val,
                                'episode_type': episode_type,
                                'emb_size':emb_size,
                                'dropout':dropout_p,
                                'nlayers':nlayers,
                                'episode':episode,
                                'disable_memory':disable_memory,
                                'disable_recon_loss':disable_recon_loss,
                                'use_attention':use_attention,
                                'max_length_eval':max_length_eval,
                                'num_episodes':num_episodes,
                                'args':args}
                    print('Saving model as: ' + fn_out_model)
                    
                    if episode % 50000 == 0:
                        torch.save(state, fn_out_model+'_'+str(episode))
                    torch.save(state, fn_out_model)


        print('Training complete')
        acc_val_gen, acc_val_retrieval, _, _ = evaluation_battery(samples_val, encoder, decoder, input_lang, output_lang, max_length_eval, verbose=False)
        print('Acc Retrieval (val): ' + str(round(acc_val_retrieval,1)))
        print('Acc Generalize (val): ' + str(round(acc_val_gen,1)))
    else: # evaluate model
        #USE_CUDA = False
        print('Results file already exists. Loading file and evaluating...')
        print('Loading model: ' + fn_out_model)        
        checkpoint = torch.load(fn_out_model, map_location='cpu') # evaluate model on CPU
        if 'episode' in checkpoint: print(' Loading epoch ' + str(checkpoint['episode']) + ' of ' + str(checkpoint['num_episodes']))
        input_lang = checkpoint['input_lang']
        output_lang = checkpoint['output_lang']
        emb_size = checkpoint['emb_size']
        nlayers = checkpoint['nlayers']
        dropout_p = checkpoint['dropout']
        input_size = input_lang.n_symbols
        output_size = output_lang.n_symbols
        samples_val = checkpoint['episodes_validation']
        disable_memory = checkpoint['disable_memory']
        max_length_eval = checkpoint['max_length_eval']
        if disable_memory:
            encoder = WrapperEncoderRNN(emb_size, input_size, output_size, nlayers, dropout_p)
        else:
            encoder = MetaNetRNN(emb_size, input_size, output_size, nlayers, dropout_p)        
        if use_attention:
            decoder = AttnDecoderRNN(emb_size, output_size, nlayers, dropout_p)
        else:
            decoder = DecoderRNN(emb_size, output_size, nlayers, dropout_p)
        if USE_CUDA:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        describe_model(encoder)
        describe_model(decoder)

        if args.new_test_ep:
            N_TEST_NEW = 20
            print("generating new test examples")
            generate_episode_train, generate_episode_test, input_lang, output_lang = get_episode_generator(
                       args.new_test_ep)

            tabu_episodes = set([])
            test_samples = []
            for i in range(N_TEST_NEW):
                sample = generate_episode_test(tabu_episodes)
                test_samples.append(sample)
                if not args.duplicate_test: 
                    tabu_episodes = tabu_update(tabu_episodes, sample['identifier'])

        if args.load_data:
            if os.path.isfile(args.load_data):
                print('loading test data ... ')
                with open(args.load_data, 'rb') as h:
                    test_samples = dill.load(h)
            else: assert False

        else:
            test_samples = samples_val

        if args.human_miniscan:
            from miniscan_state import examples_train, examples_test

            x_support = [list(ex.current) for ex in examples_train ]
            y_support = [list(ex.target) for ex in examples_train ]
            x_query = [list(ex.current) for ex in examples_test ]
            y_query = [list(ex.target) for ex in examples_test ]

            test_samples = [ build_sample(x_support,y_support,x_query,y_query,
                input_lang, output_lang, 'myhash', ) for _ in range(20)]


        test_samples = [ util.cuda_a_dict(s) for s in test_samples ]
        acc_test_gen, acc_test_retrieval, accuracy, max_length_ys, mean_length_ys, error_gen, error_retrieval = evaluation_battery(test_samples, encoder, decoder, input_lang, output_lang, max_length_eval, verbose=False, length_generalization=True)#, strict_eval=True)
        print('Acc Retrieval (test): ' + str(round(acc_test_retrieval,4)))
        print()
        print('Acc Generalize (test): ' + str(round(acc_test_gen,4)))
        print('std error test:', str(round(error_gen,4)))
        acc_test_gen_strict, _,  error_gen, error_retrieval  = evaluation_battery(test_samples, encoder, decoder, input_lang, output_lang, max_length_eval, verbose=False, strict_eval=True)
        print('Acc Generalize Strict (test): ' + str(round(acc_test_gen_strict,4)))
        # generate_episode_train, generate_episode_test, input_lang,output_lang = get_episode_generator(episode_type)
        # acc_train_gen, acc_train_retrieval = evaluation_battery([generate_episode_train([]) for _ in range(100)], encoder, decoder, input_lang, output_lang, max_length_eval, verbose=False)#, strict_eval=True)
        # print('Acc Retrieval (train): ' + str(round(acc_train_retrieval,1)))
        # print('Acc Generalize (train): ' + str(round(acc_train_gen,1)))

        # acc_val_gen, acc_val_retrieval = evaluation_battery(samples_val, encoder, decoder, input_lang, output_lang, max_length_eval, verbose=False)#, strict_eval=True)
        # print('Acc Retrieval (val): ' + str(round(acc_val_retrieval,1)))
        # print('Acc Generalize (val): ' + str(round(acc_val_gen,1)))
