#training loop
import time
import torch
import math

from util import clip, SOS_token, cuda_a_dict
from batched_synth_net import BatchedRuleSynthEncoderRNN, BatchedDoubleAttnDecoderRNN
from masked_cross_entropy import *


global batchtime
batchtime = {'max':0, 'mean':0, 'count':0}


def convert_sample(sample, model):
    states, rules = model.sample_to_statelist(sample)
    samples = []
    #for state, rule in zip(states, rules):
    for i in range(len(rules)):
        samples.append( model.state_rule_to_sample(states[i], rules[i]) )
    return samples


def gen_samples(generate_episode_train, model):
    t = time.time()
    sample = generate_episode_train(model.tabu_episodes)
    #tabu_episodes = tabu_update(tabu_episodes, sample['identifier'])
    #if len(tabu_episodes)%500==0:
    #    print("len tabu:", len(tabu_episodes))
    curtime = time.time()-t
    
    if curtime > batchtime['max']: batchtime['max'] = curtime

    batchtime['mean'] = (batchtime['mean']*batchtime['count'] + curtime)/(batchtime['count'] + 1)
    batchtime['count'] += 1

    return convert_sample(sample, model)

def get_policy_loss(samples, model, eval_only=False):
    samples = model.re_pad_batch(samples)
    samples = [cuda_a_dict(sample) for sample in samples]

    batch_size = len(samples)
    assert batch_size
    #print('multiple_sample_MODE')
    # Zero gradients of both optimizers
    if model.encoder_optimizer:
        model.encoder_optimizer.zero_grad()
    if model.decoder_optimizer:
        model.decoder_optimizer.zero_grad()

    if eval_only:
        model.encoder.eval()
        model.decoder.eval()
    else:
        model.encoder.train()
        model.decoder.train()

    encoder_embedding, dict_encoder, rules_dict_encoder = model.encoder(samples)
    rules_encoder_embedding_steps = rules_dict_encoder['embed_by_step']
    rule_encoder_lengths = rules_dict_encoder['pad']

    encoder_embedding_steps = dict_encoder['embed_by_step']
    encoder_lengths = dict_encoder['pad']
    # Prepare input and output variables

    decoder_input = torch.tensor([model.prog_lang.symbol2index[SOS_token]] * batch_size) # nq length tensor
    decoder_hidden = model.decoder.initHidden(encoder_embedding)

    target_batches = torch.transpose(torch.cat([sample['g_padded'] for sample in samples], 0), 0,1) #let's see if this works ...

    target_lengths = [l for sample in samples for l in sample['g_length'] ]
    #print([sample['g_length'] for sample in samples]) 

    max_target_length = max(target_lengths)
    
    #print("max tgt len",max_target_length )
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, model.decoder.output_size)
    if model.USE_CUDA:
        decoder_input = decoder_input.cuda()
        target_batches = target_batches.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()
    
    # Run through decoder one time step at a time
    #import pdb; pdb.set_trace()
    for t in range(max_target_length):
        if type(model.decoder) is BatchedDoubleAttnDecoderRNN:
            decoder_output, decoder_hidden, attn_by_query, rules_attn_by_query = model.decoder.forward_seq(decoder_input, 
                                                                                            decoder_hidden, 
                                                                                            encoder_embedding_steps, 
                                                                                            encoder_lengths, 
                                                                                            rules_encoder_embedding_steps, 
                                                                                            rule_encoder_lengths)
        else:
            assert False
        all_decoder_outputs[t] = decoder_output # max_len x bs x output_size
        decoder_input = target_batches[t]

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        torch.transpose(all_decoder_outputs, 0, 1).contiguous(), # -> batch_size x max_length
        torch.transpose(target_batches, 0, 1).contiguous(), # batch_size x max_length
        target_lengths
    )
    return loss


def train_batched_step(samples, model, eval_only=False):

    loss = get_policy_loss(samples, model, eval_only=eval_only)

    # gradient update
    if not eval_only:
        loss.backward()
        encoder_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), clip)
        decoder_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), clip)
        if encoder_norm > clip or decoder_norm > clip:
            print("Gradient clipped:")
            print("  Encoder norm: " + str(encoder_norm))
            print("  Decoder norm: " + str(decoder_norm))
        model.encoder_optimizer.step()
        model.decoder_optimizer.step()
    return loss.cpu().item()

def eval_ll(samples, model):
    #avg_loss = 0
    #losses = []
    #tot_len = sum([l for sample in samples for l in sample['g_length'] ])
    #target_lengths = []
    loss = train_batched_step(samples, model, eval_only=True)
    #losses.append(loss)
    #target_lengths.append(sum([l for sample in batch for l in sample['g_length'] ]))
    #weighted_loss = sum(loss*float(tgt_len)/tot_len for loss, tgt_len in zip(losses, target_lengths))
    #print("unweighted loss:", sum(losses)/len(losses))
    return loss


if __name__ == '__main__':
    pass


