#batched synth net

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from metanet_attn import *

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 20):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = np.zeros((max_seq_len, d_model))
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        self.pe = torch.tensor(pe).float().cuda() #.unsqueeze(0).cuda()
        self.pe /= math.sqrt(self.d_model)
        #1 x max_seq_len x d_model

    def forward(self, x, lengths):
        #import pdb; pdb.set_trace()
        """x: BxLxE"""
        # b x max_n_rules x emb_dim 

        # make embeddings relatively larger
        #x = x * math.sqrt(self.d_model)
        #add constant to embedding
        #loop 
        #print("hit PositionalEncodercode")
        for i, L in enumerate(lengths):
            if L != 0:
                x[i,:L,:] += self.pe[:L,:]
        return x

        # TODO optimize

class BatchedRuleSynthEncoderRNN(nn.Module):
    # embeds Rules and support items

    # Produces context-sensitive embedding of the query items. 
    # The embeddings are sensitive to the support items, which are stored in an external memory.
    #
    #  Architecture
    #   1) RNN encoder for input symbols in query and support items (either shared or separate)
    #   2) RNN encoder for output symbols in the support items only
    #   3) Key-value memory for embedding query items with context
    #   3) MLP to reduce the dimensionality of the context-sensitive embedding
    def __init__(self, embedding_dim, input_size, output_size, prog_size, nlayers, dropout_p=0.1, bidirectional=True, tie_encoders=False, use_query=False, rule_positions=False):
        # Input
        #  embedding_dim : number of hidden units in RNN encoder, and size of all embeddings
        #  input_size : number of input symbols
        #  output_size : number of output symbols
        #  nlayers : number of hidden layers in RNN encoder
        #  dropout : dropout applied to symbol embeddings and RNNs
        #  tie_encoders : use the same encoder for the support and query items? (default=True)
        super(BatchedRuleSynthEncoderRNN, self).__init__()
        self.nlayers = nlayers
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout_p
        self.bi_encoder = bidirectional
        self.rule_positions = rule_positions
        #self.attn = FancyAttn(embedding_dim, embedding_dim)
        #self.attn2 = FancyAttn(embedding_dim, embedding_dim)
        self.suppport_embedding = EncoderRNN(input_size, embedding_dim, nlayers, dropout_p, bidirectional)
        if use_query:
            if tie_encoders:
                self.query_embedding = self.suppport_embedding
            else:    
                self.query_embedding = EncoderRNN(input_size, embedding_dim, nlayers, dropout_p, bidirectional)

        self.rule_embedding = EncoderRNN(prog_size, embedding_dim, nlayers, dropout_p, bidirectional)
        self.use_query = use_query  
        self.output_embedding = EncoderRNN(output_size, embedding_dim, nlayers, dropout_p, bidirectional)
        self.hidden = nn.Linear(embedding_dim*2,embedding_dim)
        self.tanh = nn.Tanh()
        self.first_rule_param = torch.nn.parameter.Parameter(torch.rand(1, 1, embedding_dim))
        self.no_example_param = torch.nn.parameter.Parameter(torch.rand(1, embedding_dim))
        self.concat_init_layer = nn.Linear(embedding_dim * 2, embedding_dim)
        if self.rule_positions:
            self.rulePositionalEncoding = PositionalEncoder(embedding_dim) #, max_seq_len=20)


    def forward(self, samples):
        #   z_padded : LongTensor (n x max_length); list of n padded input sequences
        #   z_lengths : Python list (length n) for length of each padded input sequence 


        #for now, assume they all have the correct padding and lengths
        batch_size = len(samples)

        support_exs = []
        rule_counts = []

        xs_lengths = []
        ys_lengths = []

        xs_padded = []
        ys_padded = []

        rs_padded = []
        rs_lengths = []

        for sample in samples:
            #print(sample)
            support_exs.append( len(sample['xs_lengths']) )
            rule_counts.append( len(sample['rs_lengths']) )

            xs_lengths.extend( sample['xs_lengths'] )
            ys_lengths.extend( sample['ys_lengths'] )

            if sample['xs_lengths']: 
                xs_padded.append(sample['xs_padded'] )
                ys_padded.append(sample['ys_padded'] )

            if sample['rs']:
                rs_lengths.extend(sample['rs_lengths'])
                rs_padded.append( sample['rs_padded'])

            #import pdb; pdb.set_trace()

        #print(f"stuff:\n support_exs: {support_exs}\nrule_counts {rule_counts}\n")

        max_n_examples = max(support_exs)
        max_n_rules = max(rule_counts)

        no_examples = max_n_examples==0
        if not no_examples:
            xs_padded = torch.cat( xs_padded, 0)
            ys_padded = torch.cat( ys_padded, 0)

            #print("xs_padded shape", xs_padded.shape)
            #print("xs_lengths", len(xs_lengths))

            embed_xs,_ = self.suppport_embedding(xs_padded, xs_lengths) # ns x embedding_dim
            embed_ys,_ = self.output_embedding(ys_padded, ys_lengths) # ns x embedding_dim
            
            flat_context = torch.cat((embed_xs, embed_ys), 1)
            flat_context = self.tanh(self.hidden(flat_context))
            #flat_context = flat_context.unsqueeze(1) #ns x 1 x embedding_dim

        if no_examples: 
            context = torch.zeros(batch_size, 1, self.embedding_dim).cuda()
        else:
            context = torch.zeros(batch_size, max_n_examples, self.embedding_dim).cuda()

        #TODO: redo this line with pytorch
        current = 0
        for i, ns in enumerate(support_exs):
            if ns>0:
                context[i, :ns, :] = flat_context[current:current+ns, :]
            else: 
                support_exs[i] = 1
                context[i, 0, :] = self.no_example_param
            current += ns



        #context = context.unsqueeze(1) #b x 1 x max_n_examples x embedding_dim

        no_rules = max_n_rules==0
        if not no_rules:
            rs_padded = torch.cat( rs_padded, 0)

            #print("rs_padded shape", rs_padded.shape)
            #print("rs_lengths", len(rs_lengths))

            flat_embed_rs,_ = self.rule_embedding(rs_padded, rs_lengths)
            #print(flat_embed_rs.shape)

        if no_rules:
            embed_rs = torch.zeros(batch_size, 1, self.embedding_dim).cuda()
        else:
            embed_rs = torch.zeros(batch_size, max_n_rules, self.embedding_dim).cuda()

        #TODO: redo this line with pytorch
        current = 0
        for i, nr in enumerate(rule_counts):
            if nr>0:
                embed_rs[i, :nr, :] = flat_embed_rs[current:current+nr, :]
            else:
                embed_rs[i, 0, :] = self.first_rule_param
                rule_counts[i] = 1
            current += nr
            #import pdb; pdb.set_trace()

        #   embed_rs: b x max_n_rules x emb_dim 

        if self.rule_positions:
            embed_rs = self.rulePositionalEncoding(embed_rs, rule_counts)

        #embed_rs = embed_rs.unsqueeze(1) #b x 1 x max_n_rules x embedding_dim
        
        #print(f"Post processing stuff:\n support_exs: {support_exs}\nrule_counts {rule_counts}\n")
        #print("embr", embed_rs.shape)
        #print("context", context.shape)


        x = torch.cat((context.sum(1), embed_rs.sum(1)), 1) #is this line wrong?????? #this is the offending line, i think...
        context_init = self.concat_init_layer(x) #b x hidden size 

        #print("context", context.shape)

        #print("transposed shape", context.transpose(0,1).shape)

        return context_init, {'embed_by_step' : context.transpose(0, 1), 'pad' : support_exs}, {'embed_by_step' : embed_rs.transpose(0,1), 'pad' : rule_counts}
        #context_init: b x embedding_dim
        #context: max_n_examples x b x embedding_dim
        #embed_rs: max_n_rules x b x embedding_dim

#TODO
class BatchedDoubleAttnDecoderRNN(nn.Module):

    # One-step batch decoder with Luong et al. attention
    def __init__(self, hidden_size, output_size, nlayers, dropout_p=0.1, fancy_attn=False):
        # Input        
        #  hidden_size : number of hidden units in RNN, and embedding size for output symbols
        #  output_size : number of output symbols
        #  nlayers : number of hidden layers
        #  dropout_p : dropout applied to symbol embeddings and RNNs
        super(BatchedDoubleAttnDecoderRNN, self).__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=nlayers, dropout=dropout_p)
        if fancy_attn:
            self.attn = FancyAttn(hidden_size, hidden_size)
            self.rule_attn = FancyAttn(hidden_size, hidden_size)
        else: 
            self.attn = Attn()
            self.rule_attn = Attn()

        self.attn_concat = nn.Linear(hidden_size * 2, hidden_size)
        self.concat = nn.Linear(hidden_size * 3, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, z_padded, z_lengths, init_hidden, encoder_outputs, encoder_lengths, encoder_rule_outputs, encoder_rule_lengths):
        # Run batch attn decoder forward for a series of steps
        #  Each decoder step considers all of the encoder_outputs through attention.
        #  Attention retrieval is based on decoder hidden state (not cell state)
        #
        # Input
        #   z_padded : LongTensor (batch_size x max_length);  padded target sequences
        #   z_lengths : Python list (length batch_size) for length of each padded target sequence        
        #   init_hidden: pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        #   encoder_outputs: all encoder outputs, max_input_length x batch_size x embedding_size
        #   encoder_lengths : [batch_size list] real length of each encoder sequence
        #
        # Output
        #   output : unnormalized log-score, max_length x batch_size x output_size
        #   hidden : current decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)




        z_embed = self.embedding(z_padded) # batch_size x max_length x embedding_size
        z_embed = self.dropout(z_embed) # batch_size x max_length x embedding_size

        # Sort the sequences by length in descending order
        batch_size = len(z_lengths)
        max_length = max(z_lengths)
        z_lengths = torch.LongTensor(z_lengths)
        if z_embed.is_cuda: z_lengths = z_lengths.cuda()        
        z_lengths, perm_idx = torch.sort(z_lengths, descending=True)
        z_embed = z_embed[perm_idx]
        init_hidden = (init_hidden[0][:,perm_idx,:],init_hidden[1][:,perm_idx,:])

        # RNN decoder
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(z_embed, z_lengths, batch_first=True)
        packed_output, (hidden,cell) = self.rnn(packed_input, init_hidden)
            # hidden is nlayer x batch_size x hidden_size (pair for hidden and cell)
        rnn_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output) # max_length x batch_size x hidden_size        

        # Reverse the sorting
        _, unperm_idx = perm_idx.sort(0)
        rnn_output = rnn_output[:,unperm_idx,:] # max_length x batch_size x hidden_size
        seq_len = z_lengths[unperm_idx].tolist()


        rule_context, rule_attn_weights = self.rule_attn.forward_mask(rnn_output.transpose(0,1), encoder_rule_outputs.transpose(0,1), encoder_rule_outputs.transpose(0,1), encoder_rule_lengths)

        attn_query = self.attn_concat(torch.cat((rule_context, rnn_output.transpose(0,1)), 2)) #TODO

        # Compute context vector using attention
        context, attn_weights = self.attn.forward_mask(attn_query, encoder_outputs.transpose(0,1), encoder_outputs.transpose(0,1), encoder_lengths)
        #context, attn_weights = self.attn.forward_mask(rnn_output.transpose(0,1), encoder_outputs.transpose(0,1), encoder_outputs.transpose(0,1), encoder_lengths)
        # context, attn_weights = self.attn(rnn_output.transpose(0,1), encoder_outputs.transpose(0,1), encoder_outputs.transpose(0,1))
            # context : batch_size x max_length x hidden_size
            # attn_weights : batch_size x max_length x max_input_length

        # Concatenate the context vector and RNN hidden state, and map to an output


        concat_input = torch.cat((rnn_output, rule_context.transpose(0,1), context.transpose(0,1)), 2) # max_length x batch_size x hidden_size*2
        concat_output = self.tanh(self.concat(concat_input)) # max_length x batch_size x hidden_size
        output = self.out(concat_output) # max_length x batch_size x output_size
        return output, seq_len
            # output : [unnormalized log-score] max_length x batch_size x output_size
            # seq_len : length of each output sequence
        
    def forward_seq(self, input, last_hidden, encoder_outputs, encoder_lengths, encoder_rule_outputs, encoder_rule_lengths):
        # Run batch decoder forward for a single time step (needed for generation)
        #  Each decoder step considers all of the encoder_outputs through attention.
        #  Attention retrieval is based on decoder hidden state (not cell state)
        #
        # Input
        #  input: LongTensor of length batch_size 
        #  last_hidden: previous decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        #  encoder_outputs: all encoder outputs, max_input_length x batch_size x embedding_size
        #  encoder_lengths : [batch_size list] real length of each encoder sequence
        #
        # Output
        #   output : unnormalized log-score, batch_size x output_size
        #   hidden : current decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        #   attn_weights : attention weights, batch_size x max_input_length 

        # Embed each input symbol
        batch_size = input.numel()
        embedding = self.embedding(input) # batch_size x hidden_size
        embedding = self.dropout(embedding)
        embedding = embedding.unsqueeze(0) # S=1 x batch_size x hidden_size

        rnn_output, hidden = self.rnn(embedding, last_hidden)
            # rnn_output is S=1 x batch_size x hidden_size
            # hidden is nlayer x batch_size x hidden_size (pair for hidden and cell)

        rule_context, rule_attn_weights = self.rule_attn.forward_mask(rnn_output.transpose(0,1), encoder_rule_outputs.transpose(0,1), encoder_rule_outputs.transpose(0,1), encoder_rule_lengths)
        #print("rule mask", encoder_rule_lengths)
        #print("rule", rule_attn_weights.size(), rule_attn_weights)
        
        attn_query = self.attn_concat(torch.cat((rule_context, rnn_output.transpose(0,1)), 2)) #TODO might want to make fancier if nothing works here
        #print(attn_query.shape)

        context, attn_weights = self.attn.forward_mask(attn_query, encoder_outputs.transpose(0,1), encoder_outputs.transpose(0,1), encoder_lengths)
        # context : batch_size x 1 x hidden_size
        # attn_weights : batch_size x 1 x max_input_length
        #print("mask", encoder_lengths)
        #print("attn", attn_weights.size(), attn_weights)

        # Concatenate the context vector and RNN hidden state, and map to an output
        rnn_output = rnn_output.squeeze(0) # batch_size x hidden_size
        context = context.squeeze(1) # batch_size x hidden_size
        rule_context = rule_context.squeeze(1)
        attn_weights = attn_weights.squeeze(1) # batch_size x max_input_length
        rule_attn_weights = rule_attn_weights.squeeze(1)        
        concat_input = torch.cat((rnn_output, rule_context, context), 1) # batch_size x hidden_size*2
        concat_output = self.tanh(self.concat(concat_input)) # batch_size x hidden_size
        output = self.out(concat_output) # batch_size x output_size        
        return output, hidden, attn_weights, rule_attn_weights
            # output : [unnormalized log-score] batch_size x output_size
            # hidden: pair of size [nlayer x batch_size x hidden_size] (pair for hidden and cell)
            # attn_weights: tensor of size (batch_size x max_input_length)

    def initHidden(self, encoder_message):
        # Populate the hidden variables with a message from the decoder. 
        # All layers, and both the hidden and cell vectors, are filled with the same message.
        #   message : batch_size x hidden_size tensor
        encoder_message = encoder_message.unsqueeze(0) # 1 x batch_size x hidden_size
        encoder_message = encoder_message.expand(self.nlayers,-1,-1).contiguous() # nlayers x batch_size x hidden_size tensor
        return (encoder_message, encoder_message)