#model


import torch
import torch.nn as nn
import torch.optim as optim
from util import tabu_update, get_episode_generator, num_episodes_val, UnfinishedError, REPLError, build_padded_var

from batched_synth_net import BatchedRuleSynthEncoderRNN, BatchedDoubleAttnDecoderRNN
from metanet_attn import describe_model
from agent import Example, State, parse_rules, ParseError
import time
import copy

from number_word_interpret_grammar import IncompleteError
"""
tabu_update
get_episode_generator

"""

class Model:
    def cudaize(use_cuda):
        pass

    def load_val_episodes(self, path):
        val = torch.load(path)

        self.samples_val = val
        self.tabu_episodes = set([])
        for sample in self.samples_val:
            self.tabu_episodes = tabu_update(self.tabu_episodes,sample['identifier'])


    def generate_val_episodes(self):
        generate_episode_train, generate_episode_test, _, _, _ = get_episode_generator(self.episode_type)
        self.tabu_episodes = set([])
        self.samples_val = []
        for i in range(num_episodes_val):
            sample = generate_episode_test(self.tabu_episodes)
            self.samples_val.append(sample)
            self.tabu_episodes = tabu_update(self.tabu_episodes, sample['identifier'])

    @classmethod
    def new(cls, args):

        model = cls(args.use_cuda, args.episode_type, args.emb_size, args.nlayers, args.dropout_p, args.adam_learning_rate, args.positional)

        #deal with these
        
        if args.use_saved_val:
            model.load_val_episodes(args.saved_val_path)
        else:
            model.generate_val_episodes()

        model.num_pretrain_episodes = args.num_pretrain_episodes
        model.num_rl_episodes = args.num_pretrain_episodes
        model.max_length_eval = args.max_length_eval
        return model

    def __init__(self, use_cuda,
                         episode_type,
                         emb_size,
                         nlayers,
                         dropout_p,
                         adam_learning_rate,
                         positional,
                         use_prog_lang_for_input=False):
            
        self.USE_CUDA = use_cuda
        self.episode_type = episode_type
        self.emb_size = emb_size
        self.nlayers = nlayers
        self.dropout_p = dropout_p
        self.adam_learning_rate = adam_learning_rate
        self.positional = positional

        generate_episode_train, generate_episode_test, self.input_lang, self.output_lang, self.prog_lang = get_episode_generator(episode_type)

        if use_prog_lang_for_input:
            self.input_size = self.prog_lang.n_symbols
        else:
            self.input_size = self.input_lang.n_symbols
        self.output_size = self.output_lang.n_symbols
        self.prog_size = self.prog_lang.n_symbols

        self.encoder = BatchedRuleSynthEncoderRNN(emb_size, 
                self.input_size, 
                self.output_size, 
                self.prog_size, 
                nlayers, 
                dropout_p,
                tie_encoders=False,
                rule_positions=positional) 
        self.decoder = BatchedDoubleAttnDecoderRNN(emb_size, 
                self.prog_size, nlayers, dropout_p, 
                fancy_attn=False)

        if self.USE_CUDA:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()


        print('  Set learning rate to ' + str(adam_learning_rate))
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(),lr=adam_learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(),lr=adam_learning_rate)
        print("")
        print("Architecture options...")
        print(" Using Synthesis network")
        print("")
        describe_model(self.encoder)
        describe_model(self.decoder)



        self.pretrain_episode = 0
        self.rl_episode = 0

    def _load_data_from_checkpoint(self, checkpoint):

        #if 'episode' in checkpoint: print(' Loading epoch ' + str(checkpoint['episode']) + ' of ' + str(checkpoint['num_episodes']))
        self.samples_val = checkpoint['episodes_validation']

        self.tabu_episodes = set([])
        for sample in self.samples_val:
            self.tabu_episodes = tabu_update(self.tabu_episodes,sample['identifier'])

        #self.disable_memory = checkpoint['disable_memory']
        self.max_length_eval = checkpoint['max_length_eval'] #do something about this
        self.pretrain_episode = checkpoint['pretrain_episode']
        self.rl_episode = checkpoint['rl_episode']

        self.num_pretrain_episodes = checkpoint['num_pretrain_episodes']
        self.num_rl_episodes = checkpoint['num_rl_episodes']

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        #refresh optimizers
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(),lr=self.adam_learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(),lr=self.adam_learning_rate)

    @classmethod
    def load(cls, path, use_cuda=True):

        print('Loading model: ' + path)
        checkpoint = torch.load(path)

        episode_type = checkpoint['episode_type']
        emb_size = checkpoint['emb_size']
        nlayers = checkpoint['nlayers']
        dropout_p = checkpoint['dropout']
        adam_learning_rate = checkpoint['adam_learning_rate'] 
        if 'positional' in checkpoint.keys():
            positional = checkpoint['positional']
        else: positional = False

        #if i want to overwrite, here's my chance
        model = cls(use_cuda, episode_type, emb_size, nlayers, dropout_p, adam_learning_rate, positional)

        model._load_data_from_checkpoint(checkpoint)

        return model

    def save(self, path):        
        state = {'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'input_lang': self.input_lang,
            'output_lang': self.output_lang,
            'prog_lang': self.prog_lang,
            'episodes_validation': self.samples_val,
            'episode_type': self.episode_type,
            'emb_size':self.emb_size,
            'dropout':self.dropout_p,
            'nlayers':self.nlayers,
            'pretrain_episode': self.pretrain_episode,
            'rl_episode':self.rl_episode,
            'adam_learning_rate': self.adam_learning_rate,
            #'disable_memory':self.disable_memory,
            #'disable_recon_loss':self.disable_recon_loss,
            #'use_attention':self.use_attention,
            'max_length_eval':self.max_length_eval,
            'num_pretrain_episodes':self.num_pretrain_episodes,
            'num_rl_episodes': self.num_rl_episodes,
            #'args': self.args,
            'positional': self.encoder.rule_positions
            }

        print('Saving model as: ' + path)
        torch.save(state, path)

    def eval_mode(self):
        self.encoder.eval()
        self.decoder.eval()

    def train_mode(self):
        self.encoder.train()
        self.decoder.train()

    def re_pad_batch(self, samples, eval_mode=False):
        

        x_lens = [ max(sample['xs_lengths']) for sample in samples if sample['xs_lengths']]
        if x_lens: max_x_len = max( x_lens )

        y_lens = [max(sample['ys_lengths']) for sample in samples if sample['ys_lengths'] ]
        if y_lens: max_y_len = max(y_lens )
        r_lens = [max(sample['rs_lengths']) for sample in samples if sample['rs'] ]
        if r_lens: max_r_len = max( r_lens )
        if not eval_mode:
            #don't compute these if you are evaluating, because you don't have the info to do it and will get error
            max_g_len = max( max(sample['g_length']) for sample in samples )
            #max_g_sos_len = max( max(sample['g_sos_length']) for sample in samples )

        new_samples = []
        for sample in samples:

            if x_lens: sample['xs_padded'],_ = build_padded_var(sample['xs'], self.input_lang, max_length=max_x_len)
            if y_lens: sample['ys_padded'],_ = build_padded_var(sample['ys'], self.output_lang, max_length=max_y_len)
            if r_lens: sample['rs_padded'],_ = build_padded_var(sample['rs'], self.prog_lang, max_length=max_r_len)
            if not eval_mode:
                sample['g_padded'],_ = build_padded_var([sample['grammar']], self.prog_lang, max_length=max_g_len)
                #sample['g_sos_padded'], _ = build_padded_var([sample['grammar']], prog_lang, add_eos=False,add_sos=True, max_length=max_g_sos_len)

            new_samples.append(sample)
            #do i need to do gs?
        return new_samples

    def sample_to_statelist(self):
        raise NotImplementedError
    def state_rule_to_sample(self):
        raise NotImplementedError
    def tokenize_target_rule(self):
        raise NotImplementedError
    def detokenize_action(self):
        raise NotImplementedError
    def REPL(self):
        raise NotImplementedError
    def GroundTruthModel(self):
        raise NotImplementedError


class MiniscanRBBaseline(Model):
    #robustfill baseline

    def sample_to_statelist(self, sample):
        #assume current sample API for now, can modify if needed

        full_g = sample['grammar']
        full_rule_list = [str(r).split(' ') for r in full_g.rules]

        examples = {Example(cur, tgt) for cur, tgt in zip(sample['xs'], sample['ys']) }
        initial_state = State.new(examples)

        states = [initial_state]
        executed_actions = [full_rule_list]
        return states, executed_actions

    def REPL(self, state, action):
        if action is None:
            rules = state.rules
        else:
            rules = state.rules + action
        #print(rules)
        #try:
        g = parse_rules(rules, input_symbols=self.input_lang.symbols)

        new_examples = []
        for ex in state.examples:
            #new_ex = Example(g.apply(' '.join(ex.current)).split(' '), ex.target )
            #print('cur', tuple(g.apply(' '.join(ex.current)).split(' ')))
            #print('tgt', ex.target)
            if tuple(g.apply(' '.join(ex.current), max_recursion_count=300).split()) != ex.target:
                new_examples.append(ex) #i think this is right
            #else: print('something got hit!')

        new_state = State(new_examples, rules)
        return new_state

    def state_rule_to_sample(self, state, rule):
        #rule is the target rule, past_rules are the support rules
        sample = {}
        #print("PAST RULES INSIDE", past_rules)
        #sample['grammar'] = #todo

        tokenized_rules = self.tokenize_target_rule(rule)
        #sample['grammar'] = sample['identifier'] = tokenized_rules
        sample['grammar'] = tokenized_rules
        sample['identifier'] = "N/A"
        sample['g_padded'], sample['g_length'] = build_padded_var([tokenized_rules], self.prog_lang)

        sample['g_sos_padded'],sample['g_sos_length'] = build_padded_var(
                                                            [tokenized_rules], 
                                                            self.prog_lang, 
                                                            add_eos=False,
                                                            add_sos=True) # (nq x max_length)

        r_support = [ self.tokenize_target_rule([past_r]) for past_r in state.rules] #past_rules ]
        sample['rs'] = r_support
        if r_support:
            #this is the line:
            #print(r_support)
            sample['rs_padded'],sample['rs_lengths'] = build_padded_var(r_support, self.prog_lang)
        else:
            sample['rs_padded'], sample['rs_lengths'] = [],[]

        x_support = []
        y_support = []
        for ex in state.examples:
            x_support.append(list(ex.current))
            y_support.append(list(ex.target))

        sample['xs'] = x_support # support 
        sample['ys'] = y_support
        sample['xs_padded'],sample['xs_lengths'] = build_padded_var(x_support,self.input_lang) # (ns x max_length)
        sample['ys_padded'],sample['ys_lengths'] = build_padded_var(y_support,self.output_lang) # (ns x max_length)

        return sample

    def tokenize_target_rule(self, rule): #ONLY FOR MINISCAN
        tokenized_rules = []
        rl = len(rule)
        for i, r in enumerate(rule):
            tokenized_rules.extend(r)
            if i+1 != rl: tokenized_rules.append('\n')
        return tokenized_rules

    def detokenize_action(self, action):
        rules = []
        rule = []
        for token in action:
            if token == '\n':
                rules.append(rule)
                rule = []
                continue
            else:
                rule.append(token)
        if rule != []:
            rules.append(rule)
        return rules

    def GroundTruthModel(self, state, action):
        if action is None:
            rules = state.rules
        else:
            rules = state.rules + action

        g = parse_rules(rules, self.input_lang.symbols)

        new_examples = []
        for ex in state.examples:
            #try:
            new_ex = Example(g.apply(' '.join(ex.current)).split(), ex.target )
            # except:
            #     new_ex = '' #or somehting
            new_examples.append(new_ex) #i think this is right

        new_state = State(new_examples, rules)
        return new_state



class WordToNumber(MiniscanRBBaseline):
    #use seenRules and unseenRules

    def sample_to_statelist(self, sample):
        #assume current sample API for now, can modify if needed
        #full_g = sample['grammar']
        
        if 'seenRules' in sample:
            full_rule_list = [str(r).split(' ') for r in sample['seenRules']]

            full_rule_list = [[token for token in rule if not token == '(invalid)'] for rule in full_rule_list]
        else: full_rule_list = []

        examples = {Example(cur, tgt) for cur, tgt in zip(sample['xs'], sample['ys']) }
        initial_state = State(examples, [str(r).split(' ') for r in sample['unseenRules']] )

        states = [initial_state]
        executed_actions = [full_rule_list]
        return states, executed_actions

    def state_rule_to_sample(self, state, rule):
        #rule is the target rule, past_rules are the support rules
        sample = {}
        #print("PAST RULES INSIDE", past_rules)
        #sample['grammar'] = #todo

        tokenized_rules = self.tokenize_target_rule(rule)
        #sample['grammar'] = sample['identifier'] = tokenized_rules
        sample['grammar'] = tokenized_rules
        sample['identifier'] = "N/A"
        sample['g_padded'], sample['g_length'] = build_padded_var([tokenized_rules], self.prog_lang)

        sample['g_sos_padded'],sample['g_sos_length'] = build_padded_var(
                                                            [tokenized_rules], 
                                                            self.prog_lang, 
                                                            add_eos=False,
                                                            add_sos=True) # (nq x max_length)

        #r_support = [ self.tokenize_target_rule([past_r]) for past_r in state.rules] #past_rules ]
        r_support = None
        #if r_support: assert False #Need to think about htis

        sample['rs'] = r_support
        if r_support:
            #this is the line:
            #print(r_support)
            sample['rs_padded'],sample['rs_lengths'] = build_padded_var(r_support, self.prog_lang)
        else:
            sample['rs_padded'], sample['rs_lengths'] = [],[]

        x_support = []
        y_support = []
        for ex in state.examples:
            x_support.append(list(ex.current))

            y_support.append(self._digitize(ex.target)) #TODO

        sample['xs'] = x_support # support 
        sample['ys'] = y_support
        sample['xs_padded'],sample['xs_lengths'] = build_padded_var(x_support,self.input_lang) # (ns x max_length)
        sample['ys_padded'],sample['ys_lengths'] = build_padded_var(y_support,self.output_lang) # (ns x max_length)

        return sample

    def _digitize(self, target):
        assert len(target) == 1
        lst = []
        tgt_str = str(target[0])
        for c in tgt_str:
            lst.append(c)
        return lst

    def _parse_rules(self, rules, input_symbols=None):
        from number_generate_model import NumberGrammar
        from number_generate_model import Rule as NumberRule

        assert input_symbols
        Rules = []
        for rule in rules:
            #split into two on arrow
            if '->' in rule:
                idx = rule.index('->')
            else:
                raise ParseError
            lhs = rule[:idx]
            rhs = rule[idx+1:]

            lhs = ' '.join(lhs)
            rhs = ' '.join(rhs)
            try:
                Rules.append(NumberRule(lhs,rhs))
            except IncompleteError:
                raise ParseError

        return NumberGrammar(Rules, input_symbols)

    def REPL(self, state, action):
        if action is None:
            rules = state.rules
        else:
            rules = state.rules + action
        #print(rules)
        #try:
        g = self._parse_rules(rules, input_symbols=self.input_lang.symbols)
        #import pdb; pdb.set_trace()

        new_examples = []
        try:
            for ex in state.examples:
                #new_ex = Example(g.apply(' '.join(ex.current)).split(' '), ex.target )
                #print('cur', tuple(g.apply(' '.join(ex.current)).split(' ')))
                #print('tgt', ex.target)
                if g.apply(' '.join(ex.current)) != ex.target[0]:
                    new_examples.append(ex) #i think this is right
                #else: print('something got hit!')
        except IncompleteError:
            raise REPLError
        new_state = State(new_examples, rules)
        return new_state

    def tokenize_target_rule(self, rule): #ONLY FOR MINISCAN
        tokenized_rules = []
        rl = len(rule)
        for i, r in enumerate(rule):
            #TODO digitize tokens in r
            tokenized_rules.extend(r)
            if i+1 != rl: tokenized_rules.append('\n')
        return tokenized_rules

    def GroundTruthModel(self, state, action):
        if action is None:
            rules = state.rules
        else:
            rules = state.rules + action

        g = self._parse_rules(rules, self.input_lang.symbols)

        new_examples = []
        for ex in state.examples:
            try:
                new_ex = Example([g.apply(' '.join(ex.current))], ex.target ) #this is nasty 
            except IncompleteError:
                raise REPLError
            # except:
            #     new_ex = '' #or somehting
            new_examples.append(new_ex) #i think this is right

        new_state = State(new_examples, rules)
        return new_state

        #should be the same:
        #def detokenize_action(self, action):
        rules = []
        rule = []
        for token in action:
            if token == '\n':
                rules.append(rule)
                rule = []
                continue
            else:
                rule.append(token)
        if rule != []:
            rules.append(rule)
        return rules

