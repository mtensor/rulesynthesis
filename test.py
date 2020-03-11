#test.py
from batched_synth_net import BatchedDoubleAttnDecoderRNN, BatchedRuleSynthEncoderRNN
import torch
import numpy as np
from agent import Example, State
from agent import ParseError
import time
import copy
from collections import namedtuple
import math


from util import SOS_token, clip, UnfinishedError, REPLError, cuda_a_dict

"""
from train import state_rule_to_sample, re_pad_batch
"""

def sample_rules_batched(states, model, max_length=20, nosearch=False):
    
    #states: a list of the current states
    #past_rules_list: a list of lists of past rules, each corresponding to the states
    batch_size = len(states)
    #print(batch_size)

    samples = [model.state_rule_to_sample(state, []) for state in states]
    samples = model.re_pad_batch(samples, eval_mode=True) #TODO, i think this will crash bc no grammar
    samples = [cuda_a_dict(sample) for sample in samples]

    model.encoder.eval()
    model.decoder.eval()

    assert type(model.encoder) is BatchedRuleSynthEncoderRNN

    encoder_embedding, dict_encoder, rules_dict_encoder = model.encoder(samples)
    
    rules_encoder_embedding_steps = rules_dict_encoder['embed_by_step']
    rule_encoder_lengths = rules_dict_encoder['pad']

    encoder_embedding_steps = dict_encoder['embed_by_step']
    encoder_lengths = dict_encoder['pad']
    
     # Prepare input and output variables
    decoder_input = torch.tensor([model.prog_lang.symbol2index[SOS_token]] * batch_size) # nq length tensor
    decoder_hidden = model.decoder.initHidden(encoder_embedding)

    # Store output words and attention states
    decoded_words = []
    
    # Run through decoder
    #all_decoder_outputs = np.zeros((nq, max_length), dtype=int)
    all_decoder_outputs = np.zeros((batch_size, max_length), dtype=int)

    if model.USE_CUDA:
        decoder_input = decoder_input.cuda()    
    for t in range(max_length):
        assert type(model.decoder) is BatchedDoubleAttnDecoderRNN
        decoder_output, decoder_hidden, attn_by_query, rules_attn_by_query = model.decoder.forward_seq(
            decoder_input, decoder_hidden, encoder_embedding_steps,
            encoder_lengths, rules_encoder_embedding_steps, rule_encoder_lengths)
        
        if nosearch:
            # Choose top symbol from output
            _, topi = decoder_output.topk(1)
            topi = topi.squeeze(1)
        else:
            dist = torch.distributions.categorical.Categorical(logits=decoder_output) #should be batch_size x n_outputs
            topi = dist.sample() #should be batch_size

        #print("top i shape", topi.shape)
        #import pdb; pdb.set_trace()

        decoder_input = topi #TODO
        all_decoder_outputs[:,t] = topi.cpu().numpy()

    rules = []
    for i in range(batch_size):
        myseq = model.prog_lang.symbolsFromVector(all_decoder_outputs[i,:])
        rules.append(myseq)
    # for rule in rules: print(rule)
    # print()
    return rules

def batched_test_with_sampling(sample, model, examples=None, query_examples=None, max_len=10, timeout=10, verbose=False, 
                                        min_len=0, batch_size=64, max_rule_size=40,
                                        use_query_for_construction=True, nosearch=False, partial_credit=False, seperate_query=False):
    
    if nosearch:
        max_nodes_expanded = 1
    else: max_nodes_expanded = None

    model.encoder.eval()
    model.decoder.eval()

    start_time = time.time()

    stats = {
        'nodes_expanded': 0,
        'policy_runs': 0,
        'policy_gpu_runs': 0,
        'value_runs': 0,
        'value_gpu_runs': 0,
        'start_time': start_time
            }
    if examples and query_examples:
        initial_state = State.new(examples)

    else:
        if seperate_query:
            #print('hit sep query')
            query_examples = {Example(cur, tgt) for cur, tgt in zip(sample['xq'], sample['yq']) if cur not in sample['xs'] }
            #import pdb; pdb.set_trace()
        else:
            query_examples = {Example(cur, tgt) for cur, tgt in zip(sample['xq'], sample['yq'])}
        examples = {Example(cur, tgt) for cur, tgt in zip(sample['xs'], sample['ys']) }

        states, _ = model.sample_to_statelist(sample)
        initial_state = states[0]

    initial_states = [copy.deepcopy(initial_state) for _ in range(batch_size)]
    #initial_past_rules_list = [ [] for _ in range(batch_size)]

    best_state_n_ex = float('inf')
    num_samples = 0
    while time.time() - start_time < timeout:
        num_samples += 1
        states = initial_states
        #past_rules_list = initial_past_rules_list

        for i in range(max_len):
            #assert past_rules_list == [state.rules for state in states] #this assertion fails

            actions = sample_rules_batched(states, model, max_length=max_rule_size, nosearch=nosearch)
            #for action in actions: print(action)
            actions = [model.detokenize_action(action) for action in actions]
            # for action in actions: 
            #     for a in action:
            #         print(a)
            #     print()
            #     print()
            # # #print([len(s.examples) for s in states])
            # import pdb; pdb.set_trace()

            stats['policy_runs'] += len(states)
            stats['policy_gpu_runs'] += 1

            #if verbose: print("\t action:", action)
            new_states = []
            #new_past_rules_list = []
            for state, action in zip(states, actions):
                #for r in action: print(r)
                #import pdb; pdb.set_trace()
                try:
                    new_state = model.REPL(state, action)
                    #todo: if the state is too long then kill it maybe?? - oy vey
                    stats['nodes_expanded'] += 1
                except (ParseError, UnfinishedError, REPLError):
                    #if verbose: print("parse or unfinished error")
                    stats['nodes_expanded'] += 1
                    if max_nodes_expanded and stats['nodes_expanded'] >= max_nodes_expanded: break
                    continue
                # print(" new_state examples: ")
                # print(len(new_state.examples))
                #for ex in new_state.examples: print(ex.current, ex.target)
                if not new_state.examples and i+1 >= min_len:
                    #try on new query:
                    test_state = State(query_examples, new_state.rules)
                    try:
                        testout=model.REPL(test_state, None)
                    except (ParseError, UnfinishedError, REPLError):
                        if use_query_for_construction:
                            try:
                                testout2 = model.GroundTruthModel(test_state, None)
                            except REPLError:
                                continue
                            #print( [c in model.output_lang.symbols for ex in testout2.examples for c in ex.current]  )
                            if not all(c in model.output_lang.symbols for ex in testout2.examples for c in ex.current ):
                                if verbose: print("skipping this one bc didn't parse") #is this okay?????
                                continue
                            else:
                                r = 0
                                for ex in testout2.examples:
                                    if ex.current == ex.target: r +=1

                        if verbose: print("error on held out query")
                        hit = False
                        solution = new_state
                        stats['end_time'] = time.time()
                        #assert False, "you shouldnt have gotten an error"
                        stats['fraction_query_hit'] = r/len(test_state.examples)
                        return hit, solution, stats
                        #return test_state.rules, False
                    if not testout.examples:
                        if verbose: print("Hit task")
                        hit = True
                        solution = new_state
                        stats['end_time'] = time.time()
                        stats['fraction_query_hit'] = 1.0
                        return hit, solution, stats
                    else:
                        if verbose: print("solved given examples but failed query")
                        hit = False
                        solution = new_state
                        stats['end_time'] = time.time()
                        stats['fraction_query_hit'] = (len(test_state.examples) - len(testout.examples) )/len(test_state.examples)
                        return hit, solution, stats
                        #return new_state.rules, False
                else:
                    if partial_credit:
                        num_support_left = len(new_state.examples)
                        if num_support_left <= best_state_n_ex:
                            best_state = state
                            best_new_state = new_state
                            best_state_n_ex = num_support_left

                new_states.append(new_state)
                if max_nodes_expanded and stats['nodes_expanded'] >= max_nodes_expanded: break

            states = new_states

            #start over if all fail
            if not states: break
        if max_nodes_expanded and stats['nodes_expanded'] >= max_nodes_expanded: break
     



    if partial_credit and best_state_n_ex < float('inf'):
        new_state = best_new_state
        test_state = State(query_examples, new_state.rules)
        try:
            testout=model.REPL(test_state, None)
            solution = new_state
            stats['fraction_query_hit'] = (len(test_state.examples) - len(testout.examples) )/len(test_state.examples)
        except (ParseError, UnfinishedError, REPLError):
            print("YOU ERRORED ON BEST GUESS")
            stats['fraction_query_hit'] = 0.0
            solution = None
    else:
        solution = None
        stats['fraction_query_hit'] = 0.0

    if verbose: print("timed out on task")
    stats['end_time'] = time.time()
    return False, solution, stats
 

if __name__ == '__main__':
    
    #arguments
    results = test_agent(*args)

    #save results 
    with open(filename, 'wb') as savefile:
        dill.dump(results, savefile)
        print("results file saved at", filename)
