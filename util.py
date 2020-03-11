#util.py
import time
import math
from copy import deepcopy, copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import generate_episode as ge
from generate_episode import DataSamplingError
from interpret_grammar import Grammar, Rule

from torch.multiprocessing import Queue, Process

class UnfinishedError(Exception):
    pass
class REPLError(Exception):
    pass

class TOKENError(Exception):
    pass

cuda_a_dict = lambda d: {key: val.cuda() if type(val) is torch.Tensor else val for key, val in d.items()}


SOS_token = "SOS"
EOS_token = "EOS"
PAD_token = SOS_token

USE_CUDA = False #torch.cuda.is_available()

    # Training parameters
num_episodes_val = 20 # number of episodes to use as validation throughout learning
clip = 400.0 # clip gradients with larger magnitude than this
max_try_novel = 100 # number of attempts to find a novel episode (not in tabu list) before throwing an error

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

def pad_seq(seq, max_length):
    # seq : list of symbols
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def digitize(number):
    lst = []
    tgt_str = str(number)
    for c in tgt_str:
        lst.append(c)
    return lst

def build_padded_var(list_seq, lang, max_length=None, add_eos=True, add_sos=False, integerIO=False):
    # Transform python list into a padded torch tensor
    # 
    # Input
    #  list_seq : python list of n sequences (each is a python list of symbols)
    #  lang : language object for translation
    #  add_eos : add end of sentence token
    #  add_sos : add start of sentence token
    #
    # Output
    #  z_padded : LongTensor (n x max_length)
    #  z_lenghts : python list of sequence lengths (list of scalars)
    if max_length:
        assert not integerIO

    if integerIO:
        # we have to convert integers into lists of strings
        new_list_seq = []
        for lst in list_seq:
            new_lst = []
            for token in lst:
                if type(token) == int:
                    new_lst.extend(digitize(token))
                else:
                    new_lst.append(token)
            new_list_seq.append(new_lst)

        list_seq = new_list_seq

    n = len(list_seq)
    if n==0: return [],[]
    z_eos = list_seq
    if add_sos: 
        z_eos = [[SOS_token]+z for z in z_eos]
    if add_eos:
        z_eos = [z+[EOS_token] for z in z_eos]    
    z_lengths = [len(z) for z in z_eos]

    if max_length:
        max_len = max_length
    else: 
        max_len = max(z_lengths)

    z_padded = [pad_seq(z, max_len) for z in z_eos]
    z_padded = [lang.variableFromSymbols(z, add_eos=False).unsqueeze(0) for z in z_padded]
    z_padded = torch.cat(z_padded,dim=0)
    if USE_CUDA:
        z_padded = z_padded.cuda()
    return z_padded,z_lengths


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
        try:
            indices = [self.symbol2index[s] for s in mylist]
        except KeyError:
            #raise TOKENError
            import pdb; pdb.set_trace()
            #assert 0, 'need trace'
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


def tokenize_grammar(g):
    rules = g.split("\n")
    tokens = []

    for r in rules:
        x = r.split(' ')
        tokens.extend(x)
        tokens.append('\n')
    return tokens

def build_sample(x_support,y_support,x_query,y_query,
                input_lang,output_lang, prog_lang, myhash,
                grammar='', seenRules='', unseenRules='', integerIO=False):
    # convert lists to episode format
    sample = {}
    sample['identifier'] = myhash # unique identifier for this episode (order invariant)
    sample['xs'] = x_support # support 
    sample['ys'] = y_support
    sample['xq'] = x_query # query
    sample['yq'] = y_query
    sample['grammar'] = grammar
    # # create lists of tensors
    sample['xs_padded'],sample['xs_lengths'] = build_padded_var(x_support,input_lang, integerIO=integerIO) # (ns x max_length)
    sample['ys_padded'],sample['ys_lengths'] = build_padded_var(y_support,output_lang, integerIO=integerIO) # (ns x max_length)
    sample['xq_padded'],sample['xq_lengths'] = build_padded_var(x_query,input_lang, integerIO=integerIO) # (nq x max_length)
    sample['yq_padded'],sample['yq_lengths'] = build_padded_var(y_query,output_lang,integerIO=integerIO) # (nq x max_length)

    if unseenRules:
        sample['unseenRules'] = unseenRules
    if seenRules:
        sample['seenRules'] = seenRules
    #tokenized_g = tokenize_grammar(myhash)
    #sample['g_padded'],sample['g_length'] = build_padded_var([tokenized_g], prog_lang)
    return sample

def make_hashable(G, sort=False):
    # Transform grammar or list into a hashable string
    G_str = str(G).split('\n')
    if sort:
        G_str.sort()
    #if EASY_G:
    #hack for ME
    #G_str.sort()
        
    out = '\n'.join(G_str)
    return out.strip()

def tabu_update(tabu_list,identifier):
    # Add all elements of "identifier" to the 'tabu_list', and return updated list
    if isinstance(identifier, (list,set, tuple) ):
        tabu_list = tabu_list.union(identifier)
    elif isinstance(identifier , str):
        tabu_list.add(identifier)
    else:
        assert False
    return tabu_list

def generate_rules_episode(nsupport,
                            nquery,
                            nprims,
                            nrules,
                            input_lang,
                            output_lang,
                            prog_lang,
                            maxlen=6,
                            tabu_list=[],
                            model_in_lang=None,
                            model_out_lang=None,
                            model_prog_lang=None):
    # Generate episode based on a sampled set of rules
    #   ... randomly split data into train and test set
    # 
    # Input
    #  nsupport : number of support items
    #  nquery : number of query items
    #  nprims : number of unique primitives in each episode
    #  nrules : number of rules

    #the model langs allow you to sample from one lang but map to the one your model is using

    ntotal = nsupport+nquery
    count = 0
    input_symbols = input_lang.symbols
    output_symbols = output_lang.symbols

    #import pdb; pdb.set_trace()

    while True:
        G = ge.generate_random_rules(nprims,nrules,input_symbols,output_symbols)
        myhash = make_hashable(G)
        try:
            D = ge.sample_data(ntotal,G,maxlen_input=maxlen,maxlen_output=maxlen)
        except DataSamplingError:
            #print("hit a bad datum!")
            continue
        np.random.shuffle(D)
        x_total = [d[0].split(' ') for d in D]
        y_total = [d[1].split(' ') for d in D]
        x_support = x_total[:nsupport]
        y_support = y_total[:nsupport]
        x_query = x_total
        y_query = y_total
        if myhash not in tabu_list:
            break
        count += 1
        if count > max_try_novel:
            raise Exception('We were unable to generate an episode that is not on the tabu list')
    #import pdb; pdb.set_trace()

    return build_sample(x_support,y_support,x_query,y_query,input_lang,output_lang, prog_lang, myhash,grammar=G)



def generate_scan_episode(nsupport,
                            nquery,
                            nprims,
                            nurules,
                            nxrules,
                            input_lang,
                            output_lang,
                            prog_lang,
                            maxlen=30,
                            tabu_list=[],
                            u_type='exact',
                            model_in_lang=None,
                            model_out_lang=None,
                            model_prog_lang=None):
    # Generate episode based on a sampled set of rules
    #   ... randomly split data into train and test set
    # 
    # Input
    #  nsupport : number of support items
    #  nquery : number of query items
    #  nprims : number of unique primitives in each episode
    #  nrules : number of rules
    ntotal = nsupport+nquery
    count = 0
    input_symbols = input_lang.symbols
    output_symbols = output_lang.symbols

    while True:
        G = ge.generate_scan_rules(nprims, nurules, nxrules, input_symbols, output_symbols, u_type=u_type)
        myhash = make_hashable(G)
        try:
            D = ge.sample_data(ntotal,G,maxlen_input=maxlen, maxlen_output=maxlen, out_lang=output_lang, input_lang=input_lang)
        except DataSamplingError:
            #print("hit a bad datum!")
            continue
        np.random.shuffle(D)
        x_total = [d[0].split() for d in D]
        y_total = [d[1].split() for d in D]
        # if any(word not in output_symbols + ['SOS', 'EOS']  for y in y_total for word in y ):
        #     count += 1
        #     print("hit the thing", count)
        #     continue
        if any(('[' in word) or (']' in word) for y in y_total for word in y):
            #import pdb; pdb.set_trace()
            count += 1
            print(y_total)
            assert False, "this should have been covered"
            continue
        x_support = x_total[:nsupport]
        y_support = y_total[:nsupport]
        x_query = x_total
        y_query = y_total
        if myhash not in tabu_list:
            break
        count += 1
        if count > max_try_novel:
            import pdb; pdb.set_trace()
            raise Exception('We were unable to generate an episode that is not on the tabu list')
    #import pdb; pdb.set_trace()

    if model_in_lang and model_out_lang and model_prog_lang:
        return build_sample(x_support,y_support,x_query, y_query, model_in_lang, model_out_lang, model_prog_lang, myhash,grammar=G)
    else:
        return build_sample(x_support,y_support,x_query,y_query,input_lang,output_lang, prog_lang, myhash,grammar=G)


def get_episode_generator(episode_type, model_in_lang=None, model_out_lang=None, model_prog_lang=None):
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

    input_symbols_list_default = ['dax', 'lug', 'fep', 'blicket', 'kiki', 'tufa','gazzer', 'zup', 'wif'] #changed order for sorting
    output_symbols_list_default = ['RED', 'YELLOW', 'GREEN', 'BLUE', 'PURPLE', 'PINK', 'BLACK', 'WHITE']
    input_lang = Lang(input_symbols_list_default)
    output_lang = Lang(output_symbols_list_default)
    prog_symbols_list = input_symbols_list_default + output_symbols_list_default[:6] + ['->', '\n', 'x1', 'u1', '[x1]', '[u1]', 'x2', '[x2]', 'u2', '[u2]'] #TODO
    prog_lang = Lang(prog_symbols_list)

    if episode_type == 'rules_gen':
        
        input_lang = Lang(input_symbols_list_default + ['mup', 'dox', 'kleek'] ) #default has 9 symbols
        #output_lang defaults to 8 symbols, that works

        prog_lang = Lang (input_lang.symbols + output_lang.symbols + ['->', '\n', 'x1', 'u1', '[x1]', '[u1]', 'x2', '[x2]', 'u2', '[u2]']) 

        #what does it do to have unused query items?
        def generate_episode_train(tabu_episodes):
            nprims = random.choice((3,4))
            nsupp = random.choice(range(10,21))
            nrules = random.choice((2,3,4))
            return generate_rules_episode(nsupport=nsupp,nquery=10,nprims=nprims,nrules=nrules,input_lang=input_lang,output_lang=output_lang, prog_lang=prog_lang, tabu_list=tabu_episodes)      

        generate_episode_test = generate_episode_train

    elif 'rules_sup_' in episode_type:
        nSupp = int(episode_type.split('_')[-1])
        input_lang = Lang(input_symbols_list_default + ['mup', 'dox', 'kleek'] ) #default has 9 symbols
        #output_lang defaults to 8 symbols, that works
        prog_lang = Lang (input_lang.symbols + output_lang.symbols + ['->', '\n', 'x1', 'u1', '[x1]', '[u1]', 'x2', '[x2]', 'u2', '[u2]']) 
        #what does it do to have unused query items?
        def generate_episode_train(tabu_episodes):
            nprims = random.choice((3,4))
            nsupp = nSupp
            nrules = random.choice((2,3,4))
            return generate_rules_episode(nsupport=nsupp,nquery=10,nprims=nprims,nrules=nrules,input_lang=input_lang,output_lang=output_lang, prog_lang=prog_lang, tabu_list=tabu_episodes)      
        generate_episode_test = generate_episode_train

    elif 'rules_horules_' in episode_type:
        nHO = int(episode_type.split('_')[-1])
        input_lang = Lang(input_symbols_list_default + ['mup', 'dox', 'kleek'] ) #default has 9 symbols
        #output_lang defaults to 8 symbols, that works
        prog_lang = Lang (input_lang.symbols + output_lang.symbols + ['->', '\n', 'x1', 'u1', '[x1]', '[u1]', 'x2', '[x2]', 'u2', '[u2]']) 
        #what does it do to have unused query items?
        def generate_episode_train(tabu_episodes):
            nprims = random.choice((3,4))
            nsupp = 30 #random.choice(range(10,21))
            nrules = nHO
            return generate_rules_episode(nsupport=nsupp,nquery=10,nprims=nprims,nrules=nrules,input_lang=input_lang,output_lang=output_lang, prog_lang=prog_lang, tabu_list=tabu_episodes)      
        generate_episode_test = generate_episode_train

    elif 'rules_prims_' in episode_type:
        nPrims = int(episode_type.split('_')[-1])
        input_lang = Lang(input_symbols_list_default + ['mup', 'dox', 'kleek'] ) #default has 9 symbols
        #output_lang defaults to 8 symbols, that works
        prog_lang = Lang (input_lang.symbols + output_lang.symbols + ['->', '\n', 'x1', 'u1', '[x1]', '[u1]', 'x2', '[x2]', 'u2', '[u2]']) 
        #what does it do to have unused query items?
        def generate_episode_train(tabu_episodes):
            nprims = nPrims
            nsupp = 30 #random.choice(range(10,21))
            nrules = random.choice((2,3,4))
            return generate_rules_episode(nsupport=nsupp,nquery=10,nprims=nprims,nrules=nrules,input_lang=input_lang,output_lang=output_lang, prog_lang=prog_lang, tabu_list=tabu_episodes)      
        generate_episode_test = generate_episode_train

    elif 'lang_' in episode_type:
        lang = episode_type.split('_')[-1]
        from number_generate_model import generate_lang_test_episode
        from number_word_interpret_grammar import RHS_DICT
        tokens = ['token'+format(i, '02d') for i in range(1, 52)]
        input_lang = Lang(tokens)
        output_lang = Lang([str(i) for i in range(10)] )
        prog_symbols = ['1000000*','10000*', '1000*', '100*', '10*', '[x1]*10', '[x1]*100', '[x1]*1000',  '[x1]*10000',  '[x1]*1000000', '[x1]', '[u1]','[y1]', 'x1', 'u1', 'y1', '->', '\n'] + [str(i) for i in range(10)]
        prog_lang = Lang(prog_symbols+input_lang.symbols)
        nsupp = 25
        nquery = 100
        def generate_episode_train(tabu_examples):
            return generate_lang_test_episode(nsupp,
                                nquery,
                                input_lang,
                                output_lang,
                                prog_lang, 
                                tabu_examples,
                                lang=lang)
        generate_episode_test = generate_episode_train

    elif episode_type == 'wordToNumber':
        from number_generate_model import generate_wordToNumber_episode
        from number_word_interpret_grammar import RHS_DICT
        tokens = ['token'+format(i, '02d') for i in range(1, 52)]
        input_lang = Lang(tokens)
        output_lang = Lang([str(i) for i in range(10)] )
        prog_symbols = ['1000000*','10000*', '1000*', '100*', '10*', '[x1]*10', '[x1]*100', '[x1]*1000',  '[x1]*10000',  '[x1]*1000000', '[x1]', '[u1]','[y1]', 'x1', 'u1', 'y1', '->', '\n'] + [str(i) for i in range(10)]
        prog_lang = Lang(prog_symbols+input_lang.symbols)
        
        def generate_episode_train(tabu_examples):
            nsupp = random.choice(range(60,101)) #should vary this ...
            nquery = 10
            return generate_wordToNumber_episode(nsupp,
                                                nquery,
                                                input_lang,
                                                output_lang,
                                                prog_lang, 
                                                tabu_examples)

        generate_episode_test = generate_episode_train

    elif episode_type == 'scan_random':
        words = ['walk','look','run','jump','turn','left','right','opposite','around','twice','thrice','and','after'] + ['dax', 'blicket', 'lug', 'kiki']
        cmds = ['WALK','LOOK','RUN','JUMP','LTURN','RTURN'] + ['RED', 'BLUE', 'GREEN']
        input_lang = Lang( words )
        output_lang = Lang( cmds )
        prog_lang = Lang( words+cmds+ ['->', '\n', 'x1', 'u1', '[x1]', '[u1]', 'x2', '[x2]', 'u2', '[u2]', ""]) #, '[', ']'] ) #""

        tp = 'random'

        def generate_episode_train(tabu_episodes):
            nprims = random.choice(range(4,9))
            nsupp = random.choice(range(30,51)) 

            nurules = 0
            nxrules = random.choice((3,4,5,6,7))

            return generate_scan_episode(nsupport=nsupp,
                                        nquery=10,
                                        nprims=nprims,
                                        nurules=nurules,
                                        nxrules=nxrules,
                                        input_lang=input_lang,
                                        output_lang=output_lang, 
                                        prog_lang=prog_lang, 
                                        tabu_list=tabu_episodes, u_type=tp)      
        generate_episode_test = generate_episode_train
    
    elif episode_type in ['scan_simple_original', 'scan_jump_original', 'scan_around_right_original', 'scan_length_original']:

        dic = {'scan_simple_original':'simple',
                'scan_jump_original': 'addprim_jump',
                'scan_around_right_original':'template_around_right',
                'scan_length_original': 'length'  }

        scan_train = ge.load_scan_file( dic[episode_type],'train')
        scan_test = ge.load_scan_file( dic[episode_type],'test')
        #assert 0, "deal with langs"
        # input_symbols_scan = get_unique_words([c[0] for c in scan_train+scan_test])
        # output_symbols_scan = get_unique_words([c[1] for c in scan_train+scan_test])
        # input_lang = Lang(input_symbols_scan)
        # output_lang = Lang(output_symbols_scan)
        words = ['walk','look','run','jump','turn','left','right','opposite','around','twice','thrice','and','after'] + ['dax', 'blicket', 'lug', 'kiki']
        cmds = ['I_WALK','I_LOOK','I_RUN','I_JUMP','I_TURN_LEFT','I_TURN_RIGHT'] + ['RED', 'BLUE', 'GREEN']
        #assert set(words) == set(get_unique_words([c[0] for c in scan_train+scan_test]))
        #assert set(cmds) == set(get_unique_words([c[1] for c in scan_train+scan_test]))
        print("WARNING: vocab includes extra words, so beware")
        input_lang = Lang( words)
        output_lang = Lang( cmds )
        prog_lang = Lang( words+cmds+ ['->', '\n', 'x1', 'u1', '[x1]', '[u1]', 'x2', '[x2]', 'u2', '[u2]', ""])#, '[', ']'] ) #""


        generate_episode_train = lambda tabu_episodes : generate_traditional_synth_scan_episode(
                                                                nsupport=100, 
                                                                nquery=500, 
                                                                input_lang=input_lang, 
                                                                output_lang=output_lang, 
                                                                train_tuples=scan_train, 
                                                                test_tuples=scan_test, 
                                                                tabu_list=tabu_episodes)
        generate_episode_test = lambda tabu_episodes :  generate_traditional_synth_scan_episode(
                                                                nsupport=100, 
                                                                nquery=500, 
                                                                input_lang=input_lang, 
                                                                output_lang=output_lang, 
                                                                train_tuples=scan_train, 
                                                                test_tuples=scan_test, 
                                                                tabu_list=tabu_episodes)

    else:
        raise Exception("episode_type is not valid" )

    return generate_episode_train, generate_episode_test, input_lang, output_lang, prog_lang

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

def generate_traditional_synth_scan_episode(nsupport,nquery,input_lang,output_lang,train_tuples, test_tuples,tabu_list=[]):
    # Generate a batch of train SCAN commands in the support set, and test SCAN commands in the query set
    #  The tabu_list is not used in this case


    D_support, vv  = ge.sample_traditional_scan(nsupport,0,train_tuples) # no support set
    assert not vv
    vv, D_query = ge.sample_traditional_scan(0,nquery,test_tuples) # no support set
    assert not vv
    x_support = [d[0].split(' ') for d in D_support]
    y_support = [d[1].split(' ') for d in D_support]
    x_query = [d[0].split(' ') for d in D_query]
    y_query = [d[1].split(' ') for d in D_query]
    return build_sample(x_support,y_support,x_query,y_query,input_lang,output_lang, prog_lang=None, myhash='',grammar='')

def get_supervised_batchsize(fn, batchsize=200):
    #takes a generation function and outputs lists of optimal size
    remainder = []
    while True:
        preS = remainder
        if len(preS) > batchsize:
            yield preS[:batchsize]
            remainder = preS[batchsize:]
            continue

        S = fn()
        S = preS+S
        ln = len(S)

        if ln > batchsize:
            yield S[:batchsize]
            remainder = S[batchsize:]
            continue
        elif ln < batchsize:
            remainder = S
            continue
        elif ln == batchsize:
            yield S
            remainder = []
            continue
        else: assert 0, "uh oh, not a good place"

class GenData:
    def __init__(self, fn, n_processes=4, max_size=200, batchsize=200):
        ##what needs to happen:
        def consumer(Q):
            iterator = get_supervised_batchsize(fn, batchsize=batchsize) #todo 
            while True:
                try:
                    # get a new message
                    size = Q.qsize()
                    #print(size)
                    if size < max_size:
                        # process the data
                        ret = next(iterator)
                        Q.put( ret )
                    else:
                        time.sleep(2) 
                except ValueError as e:
                    print("I think you closed the thing while it was running, but that's okay")
                    break
                except Exception as e:
                    print("error!", e)
                    break

        self.Q = Queue()
        print("started queue ...")

        # instantiate workers
        self.workers = [Process(target=consumer, args=(self.Q,))
               for i in range(n_processes)]

        for w in self.workers:
            w.start()
        print("started parallel workers, ready to work!")

    def batchIterator(self):
        while True:
            yield self.Q.get()
        #yield from get_supervised_batchsize(self.Q.get, batchsize=batchsize) #is this a slow way of doing this??
        
    def kill(self):
        #KILL stuff
        # tell all workers, no more data (one msg for each)
        # join on the workers
        for w in self.workers:
            try:
                w.close() #this will cause a valueError apparently??
            except ValueError:
                print("killed a worker")
                continue

if __name__=='__main__':
    pass
