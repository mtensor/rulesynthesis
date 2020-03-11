import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from masked_cross_entropy import sequence_mask


# Code adapted from:
#   https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

def describe_model(net):
    if type(net) is MetaNetRNN:
        print('EncoderMetaNet specs:')
        print(' nlayers=' + str(net.nlayers))
        print(' embedding_dim=' + str(net.embedding_dim))
        print(' dropout=' + str(net.dropout_p))
        print(' bi_encoder=' + str(net.bi_encoder))
        print(' n_input_symbols=' + str(net.input_size))
        print(' n_output_symbols=' + str(net.output_size))
        print('')
    elif type(net) is AttnDecoderRNN:
        print('AttnDecoderRNN specs:')
        print(' nlayers=' + str(net.nlayers))
        print(' hidden_size=' + str(net.hidden_size))
        print(' dropout=' + str(net.dropout_p))
        print(' n_output_symbols=' + str(net.output_size))
        print('')
    elif type(net) is DecoderRNN:
        print('DecoderRNN specs:')
        print(' nlayers=' + str(net.nlayers))
        print(' hidden_size=' + str(net.hidden_size))
        print(' dropout=' + str(net.dropout_p))
        print(' n_output_symbols=' + str(net.output_size))
        print("")
    elif type(net) is EncoderRNN or type(net) is WrapperEncoderRNN:
        print('EncoderRNN specs:')
        print(' bidirectional=' + str(net.bi))
        print(' nlayers=' + str(net.nlayers))
        print(' hidden_size=' + str(net.embedding_dim))
        print(' dropout=' + str(net.dropout_p))
        print(' n_input_symbols=' + str(net.input_size))
        print('')
    else:
        print('Network type not found...')



class MetaNetRNN(nn.Module):
    # Produces context-sensitive embedding of the query items. 
    # The embeddings are sensitive to the support items, which are stored in an external memory.
    #
    #  Architecture
    #   1) RNN encoder for input symbols in query and support items (either shared or separate)
    #   2) RNN encoder for output symbols in the support items only
    #   3) Key-value memory for embedding query items with context
    #   3) MLP to reduce the dimensionality of the context-sensitive embedding
    def __init__(self, embedding_dim, input_size, output_size, nlayers, dropout_p=0.1, bidirectional=True, tie_encoders=True):
        # Input
        #  embedding_dim : number of hidden units in RNN encoder, and size of all embeddings
        #  input_size : number of input symbols
        #  output_size : number of output symbols
        #  nlayers : number of hidden layers in RNN encoder
        #  dropout : dropout applied to symbol embeddings and RNNs
        #  tie_encoders : use the same encoder for the support and query items? (default=True)
        super(MetaNetRNN, self).__init__()
        self.nlayers = nlayers
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout_p
        self.bi_encoder = bidirectional
        self.attn = Attn()
        self.suppport_embedding = EncoderRNN(input_size, embedding_dim, nlayers, dropout_p, bidirectional)
        if tie_encoders:
            self.query_embedding = self.suppport_embedding
        else:    
            self.query_embedding = EncoderRNN(input_size, embedding_dim, nlayers, dropout_p, bidirectional)
        self.output_embedding = EncoderRNN(output_size, embedding_dim, nlayers, dropout_p, bidirectional)
        self.hidden = nn.Linear(embedding_dim*2,embedding_dim)
        self.tanh = nn.Tanh()

    def forward(self, sample):
        #   z_padded : LongTensor (n x max_length); list of n padded input sequences
        #   z_lengths : Python list (length n) for length of each padded input sequence 

        xs_padded = sample['xs_padded'] # support set input sequences; LongTensor (ns x max_xs_length)
        xs_lengths = sample['xs_lengths'] # ns list of lengths
        ys_padded = sample['ys_padded'] # support set output sequences; LongTensor (ns x max_ys_length)
        ys_lengths = sample['ys_lengths'] # ns list of lengths
        ns = xs_padded.size(0)
        xq_padded = sample['xq_padded'] # query set input sequences; LongTensor (nq x max_xq_length)
        xq_lengths = sample['xq_lengths'] # nq list of lengths
        nq = xq_padded.size(0)

        # Embed the input sequences for support and query set
        embed_xs,_ = self.suppport_embedding(xs_padded,xs_lengths) # ns x embedding_dim
        embed_xq,dict_embed_xq = self.query_embedding(xq_padded,xq_lengths) # nq x embedding_dim
        embed_xq_by_step = dict_embed_xq['embed_by_step'] # max_xq_length x nq x embedding_dim (embedding at each step)
        len_xq = dict_embed_xq['seq_len'] # len_xq is nq array with length of each sequence

        # Embed the output sequences for support set
        embed_ys,_ = self.output_embedding(ys_padded,ys_lengths) # ns x embedding_dim

        # Compute context based on key-value memory at each time step for queries
        max_xq_length = embed_xq_by_step.size(0) # for purpose of attention, this is the "batch_size"
        value_by_step, attn_by_step = self.attn(embed_xq_by_step, embed_xs.expand(max_xq_length,-1,-1), embed_ys.expand(max_xq_length, -1, -1))
            # value_by_step : max_xq_length x nq x embedding_dim
            # attn_by_step : max_xq_length x nq x ns
        concat_by_step = torch.cat((embed_xq_by_step,value_by_step),2) # max_xq_length x nq x embedding_dim*2
        context_by_step = self.tanh(self.hidden(concat_by_step)) # max_xq_length x nq x embedding_dim

        # Grab the last context for each query
        context_last = [context_by_step[len_xq[q]-1,q,:] for q in range(nq)] # list of 1D Tensors
        context_last = torch.stack(context_last, dim=0) # nq x embedding_dim

        #import pdb; pdb.set_trace()
        return context_last, {'embed_by_step' : context_by_step, 'attn_by_step' : attn_by_step, 'seq_len' : len_xq}
            # context_last : nq x embedding
            # embed_by_step: embedding at every step for each query [max_xq_length x nq x embedding_dim]
            # attn_by_step : attention over support items at every step for each query [max_xq_length x nq x ns]
            # seq_len : length of each query [nq list]

class EncoderRNN(nn.Module):
    # Embed a sequence of symbols using a LSTM.
    #
    # The RNN hidden vector (not cell vector) at each step is captured,
    #   for transfer to an attention-based decoder.
    #
    # Does not assume that sequences are sorted by length
    def __init__(self, input_size, embedding_dim, nlayers, dropout_p, bidirectional):
        # Input
        #  input_size : number of input symbols
        #  embedding_dim : number of hidden units in RNN encoder, and size of all embeddings        
        #  nlayers : number of hidden layers
        #  dropout : dropout applied to symbol embeddings and RNNs
        #  bidirectional : use a bidirectional LSTM instead and sum of the resulting embeddings
        super(EncoderRNN, self).__init__()
        self.nlayers = nlayers
        self.input_size = input_size
        self.embedding_dim = embedding_dim        
        self.dropout_p = dropout_p
        self.bi = bidirectional
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers=nlayers, dropout=dropout_p, bidirectional=bidirectional)

    def forward(self, z_padded, z_lengths):
        # Input 
        #   z_padded : LongTensor (n x max_length); list of n padded input sequences
        #   z_lengths : Python list (length n) for length of each padded input sequence    
        #import pdb; pdb.set_trace()    
        z_embed = self.embedding(z_padded) # n x max_length x embedding_size
        z_embed = self.dropout(z_embed) # n x max_length x embedding_size

        # Sort the sequences by length in descending order
        n = len(z_lengths)
        max_length = max(z_lengths)
        z_lengths = torch.LongTensor(z_lengths)
        if z_embed.is_cuda: z_lengths = z_lengths.cuda()
        z_lengths, perm_idx = torch.sort(z_lengths, descending=True)
        z_embed = z_embed[perm_idx]

        # RNN embedding
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(z_embed, z_lengths, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_input)
             # hidden is nlayers*num_directions x n x embedding_size
             # hidden and cell are unpacked, such that they stores the last hidden state for each unpadded sequence        
        hidden_by_step, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output) # max_length x n x embedding_size*num_directions

        # If biLSTM, sum the outputs for each direction
        if self.bi:
            hidden_by_step = hidden_by_step.view(max_length, n, 2, self.embedding_dim)
            hidden_by_step = torch.sum(hidden_by_step, 2) # max_length x n x embedding_size
            hidden = hidden.view(self.nlayers, 2, n, self.embedding_dim)
            hidden = torch.sum(hidden, 1) # nlayers x n x embedding_size
        hidden = hidden[-1,:,:] # n x embedding_size (grab the last layer)

        # Reverse the sorting
        _, unperm_idx = perm_idx.sort(0)
        hidden = hidden[unperm_idx,:] # n x embedding_size
        hidden_by_step = hidden_by_step[:,unperm_idx,:] # max_length x n x embedding_size
        seq_len = z_lengths[unperm_idx].tolist()

        return hidden, {"embed_by_step" : hidden_by_step, "seq_len" : seq_len}
                # hidden is (n x embedding_size); 
                # embed_by_step is (max_length x n x embedding_size)                
                # seq_len is tensor of length n

class WrapperEncoderRNN(EncoderRNN):
    # Wrapper for RNN encoder to behave like MetaNetRNN encoder.
    #  This isn't really doing meta-learning, since it is ignoring the support set entirely. 
    #  Instead, it allows us to train a standard sequence-to-sequence model, using the query set as the batch.
    def __init__(self, embedding_dim, input_size, output_size, nlayers, dropout_p=0.1, bidirectional=True, tie_encoders=True):
        super(WrapperEncoderRNN, self).__init__(input_size, embedding_dim, nlayers, dropout_p, bidirectional)
    def forward(self, sample):
        hidden, mydict = super(WrapperEncoderRNN, self).forward(sample['xq_padded'],sample['xq_lengths'])
        mydict['attn_by_step'] = [] # not applicable
        return hidden, mydict

class Attn(nn.Module):

    def __init__(self):
        super(Attn, self).__init__()

    def forward(self, Q, K, V):
        # Key-value memory which takes queries and retrieves weighted combinations of values
        #
        # Input
        #  Q : Matrix of queries; batch_size x n_queries x query_dim
        #  K : Matrix of keys; batch_size x n_memory x query_dim
        #  V : Matrix of values; batch_size x n_memory x value_dim
        #
        # Output
        #  R : soft-retrieval of values; batch_size x n_queries x value_dim
        #  attn_weights : soft-retrieval of values; batch_size x n_queries x n_memory
        query_dim = torch.tensor(float(Q.size(2)))
        if Q.is_cuda: query_dim = query_dim.cuda()
        attn_weights = torch.bmm(Q,K.transpose(1,2)) # batch_size x n_queries x n_memory
        attn_weights = torch.div(attn_weights, torch.sqrt(query_dim))
        attn_weights = F.softmax(attn_weights, dim=2) # batch_size x n_queries x n_memory
        R = torch.bmm(attn_weights,V) # batch_size x n_queries x value_dim
        return R, attn_weights

    def forward_mask(self, Q, K, V, memory_length):
        # Key-value memory which takes queries and retrieves weighted combinations of values
        #   This version masks out certain memories so you can differing numbers of memories per batch
        #
        # Input
        #  Q : Matrix of queries; batch_size x n_queries x query_dim
        #  K : Matrix of keys; batch_size x n_memory x query_dim
        #  V : Matrix of values; batch_size x n_memory x value_dim
        #  memory_length : [batch_size list] real number of keys in each batch
        #
        # Output
        #  R : soft-retrieval of values; batch_size x n_queries x value_dim
        #  attn_weights : soft-retrieval of values; batch_size x n_queries x n_memory


        #print("mem", memory_length)

        query_dim = torch.tensor(float(Q.size(2)))
        batch_size = K.size(0)
        assert(len(memory_length)==batch_size)
        memory_length = torch.LongTensor(memory_length)
        if Q.is_cuda: 
            memory_length = memory_length.cuda()
            query_dim = query_dim.cuda()
        attn_weights = torch.bmm(Q,K.transpose(1,2)) # batch_size x n_queries x n_memory



        #print("PRE attn_weights", attn_weights.size(), attn_weights)
        #added this condition for if memory lengths are all zero:
        if not any(memory_length):
            assert 0
            #mask = torch.zeros(len(memory_length),1).byte()
            #if memory_length.is_cuda: mask = mask.cuda()
        mask = sequence_mask(memory_length)# batch_size x n_memory
        #print("mask1\n", mask)
        mask = mask.unsqueeze(1).expand_as(attn_weights) # batch_size x n_queries x n_memory
        #print("mask2\n", mask)


        attn_weights[~mask] = float('-inf')
        #attn_weights = attn_weights + (1-mask).float()*-1e18 #i think this is wrong .... 

        #print("POST attn_weights\n", attn_weights)

        attn_weights = torch.div(attn_weights, torch.sqrt(query_dim))
        attn_weights = F.softmax(attn_weights, dim=2) # batch_size x n_queries x n_memory


        #print("SOFTMAX attn_weights\n", attn_weights)
        R = torch.bmm(attn_weights, V) # batch_size x n_queries x value_dim

        #import pdb; pdb.set_trace()
        return R, attn_weights

class FancyAttn(nn.Module):

    def __init__(self, q_dim, v_dim, n_heads=4):
        super(FancyAttn, self).__init__()

        assert q_dim % n_heads == 0
        assert v_dim % n_heads == 0

        self.n_heads = n_heads
        self.v_dim = v_dim
        self.q_dim = q_dim

        self.Q_lin = nn.Linear(q_dim, q_dim)
        self.K_lin = nn.Linear(q_dim, q_dim)
        self.V_lin = nn.Linear(v_dim, v_dim)

        self.d_q = q_dim/n_heads
        self.d_v = v_dim/n_heads


    def forward(self, Q, K, V):
        #
        # Input
        #  Q : Matrix of queries; batch_size x n_queries x query_dim
        #  K : Matrix of keys; batch_size x n_memory x query_dim
        #  V : Matrix of values; batch_size x n_memory x value_dim
        #
        # Output
        #  R : soft-retrieval of values; batch_size x n_queries x value_dim
        #  attn_weights : soft-retrieval of values; batch_size x n_queries x n_memory


        # Q is batch_size x n_heads x n_queries x d_q
        Q = self.Q_lin(Q).relu().view(Q.size(0), self.n_heads, Q.size(1), self.d_q) 
        K = self.K_lin(K).relu().view(K.size(0), self.n_heads, K.size(1), self.d_q) 
        V = self.V_lin(V).relu().view(V.size(0), self.n_heads, V.size(1), self.d_v) 



        query_dim = torch.tensor(float(self.d_q)) #Q.size(2)))


        if Q.is_cuda: query_dim = query_dim.cuda()

        #attn_weights = torch.bmm(Q,K.transpose(1,2)) # batch_size x n_queries x n_memory

        attn_weights = torch.matmul(Q, K.transpose(2,3)) #batch_size x n_heads x n_queries x n_memory?


        attn_weights = torch.div(attn_weights, torch.sqrt(query_dim))
        attn_weights = F.softmax(attn_weights, dim=3) # batch_size x n_queries x n_memory
        R = torch.matmul(attn_weights, V) # batch_size x n_queries x value_dim

        R = R.contiguous().view(R.size(0), -1, self.v_dim)
        return R, attn_weights

class AttnDecoderRNN(nn.Module):

    # One-step batch decoder with Luong et al. attention
    def __init__(self, hidden_size, output_size, nlayers, dropout_p=0.1, fancy_attn=False):
        # Input        
        #  hidden_size : number of hidden units in RNN, and embedding size for output symbols
        #  output_size : number of output symbols
        #  nlayers : number of hidden layers
        #  dropout_p : dropout applied to symbol embeddings and RNNs
        super(AttnDecoderRNN, self).__init__()
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
        else: 
            self.attn = Attn()

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, last_hidden, encoder_outputs):
        # Run batch decoder forward for a single time step.
        #  Each decoder step considers all of the encoder_outputs through attention.
        #  Attention retrieval is based on decoder hidden state (not cell state)
        #
        # Input
        #  input: LongTensor of length batch_size 
        #  last_hidden: previous decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        #  encoder_outputs: all encoder outputs, max_input_length x batch_size x embedding_size
        #
        # Output
        #   output : unnormalized output probabilities, batch_size x output_size
        #   hidden : current decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        #   attn_weights : attention weights, batch_size x max_input_length 

        # Embed each input symbol
        batch_size = input.numel()
        embedding = self.embedding(input) # batch_size x hidden_size
        embedding = self.dropout(embedding)
        embedding = embedding.unsqueeze(0) # S=1 x batch_size x hidden_size

        #import pdb; pdb.set_trace()
        rnn_output, hidden = self.rnn(embedding, last_hidden)
            # rnn_output is S=1 x batch_size x hidden_size
            # hidden is nlayer x batch_size x hidden_size (pair for hidden and cell)

        #print("encoder outputsh shape", encoder_outputs.shape)
        #import pdb; pdb.set_trace()

        context, attn_weights = self.attn(rnn_output.transpose(0,1), encoder_outputs.transpose(0,1), encoder_outputs.transpose(0,1))
            # context : batch_size x 1 x hidden_size
            # attn_weights : batch_size x 1 x max_input_length
        
        # Concatenate the context vector and RNN hidden state, and map to an output
        rnn_output = rnn_output.squeeze(0) # batch_size x hidden_size
        context = context.squeeze(1) # batch_size x hidden_size
        attn_weights = attn_weights.squeeze(1) # batch_size x max_input_length        
        concat_input = torch.cat((rnn_output, context), 1) # batch_size x hidden_size*2
        concat_output = self.tanh(self.concat(concat_input)) # batch_size x hidden_size
        output = self.out(concat_output) # batch_size x output_size
        return output, hidden, attn_weights
            # output : [unnormalized probabilities] batch_size x output_size
            # hidden: pair of size [nlayer x batch_size x hidden_size] (pair for hidden and cell)
            # attn_weights: tensor of size (batch_size x max_input_length)

    def initHidden(self, encoder_message):
        # Populate the hidden variables with a message from the decoder. 
        # All layers, and both the hidden and cell vectors, are filled with the same message.
        #   message : batch_size x hidden_size tensor
        encoder_message = encoder_message.unsqueeze(0) # 1 x batch_size x hidden_size
        encoder_message = encoder_message.expand(self.nlayers,-1,-1).contiguous() # nlayers x batch_size x hidden_size tensor
        return (encoder_message, encoder_message)

class DecoderRNN(nn.Module):

    # One-step simple batch RNN decoder
    def __init__(self, hidden_size, output_size, nlayers, dropout_p=0.1):
        # Input        
        #  hidden_size : number of hidden units in RNN, and embedding size for output symbols
        #  output_size : number of output symbols
        #  nlayers : number of hidden layers
        #  dropout_p : dropout applied to symbol embeddings and RNNs
        super(DecoderRNN, self).__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=nlayers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, last_hidden):
        # Run batch decoder forward for a single time step.
        #
        # Input
        #  input: LongTensor of length batch_size 
        #  last_hidden: previous decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        #
        # Output
        #   output : unnormalized output probabilities, batch_size x output_size
        #   hidden : current decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)

        # Embed each input symbol
        batch_size = input.numel()
        embedding = self.embedding(input) # batch_size x hidden_size
        embedding = self.dropout(embedding)
        embedding = embedding.unsqueeze(0) # S=1 x batch_size x hidden_size
        rnn_output, hidden = self.rnn(embedding, last_hidden)
            # rnn_output is S=1 x batch_size x hidden_size
            # hidden is nlayer x batch_size x hidden_size (pair for hidden and cell)
        rnn_output = rnn_output.squeeze(0) # batch_size x hidden_size
        output = self.out(rnn_output) # batch_size x output_size
        return output, hidden
            # output : [unnormalized probabilities] batch_size x output_size
            # hidden: pair of size [nlayer x batch_size x hidden_size] (pair for hidden and cell)

    def initHidden(self, encoder_message):
        # Populate the hidden variables with a message from the decoder. 
        # All layers, and both the hidden and cell vectors, are filled with the same message.
        #   message : batch_size x hidden_size tensor
        encoder_message = encoder_message.unsqueeze(0) # 1 x batch_size x hidden_size
        encoder_message = encoder_message.expand(self.nlayers,-1,-1).contiguous() # nlayers x batch_size x hidden_size tensor
        return (encoder_message, encoder_message)

# # DEBUG Key-value memory accessed at each step
# nstep = embed_xq_by_step.shape[0] # number of time steps
# context_by_step2 = torch.zeros((nstep,nq,self.embedding_dim)) # max_length x nq x embedding_dim
# attn_by_step2 = torch.zeros((nstep,nq,ns)) # max_length x nq x ns
# if embed_xs.is_cuda:
#     context_by_step2 = context_by_step2.cuda()
#     attn_by_step2 = attn_by_step2.cuda()
# for s in range(nstep): # step over time
#     slice_embed_xq = embed_xq_by_step[s,:,:] # ns x embedding_dim (query embedding at time s)
#     sims = torch.mm(embed_xs,torch.t(slice_embed_xq)) # ns x nq  (pairwise similarities)        
#     attn_weights = F.softmax(sims, dim=0) # ns x nq (normalize a query across support items/rows)
#     attn_weights = torch.t(attn_weights) # nq x ns
#     value = torch.mm(attn_weights,embed_ys) # nq x embedding_size (soft-retrieval of value)
#     value_query_combo = self.tanh(self.hidden(torch.cat((slice_embed_xq,value),1))) # nq x embedding_size
#     context_by_step2[s,:,:] = value_query_combo # nq x embedding_size
#     attn_by_step2[s,:,:] = attn_weights # nq x ns

# print(torch.all(torch.isclose(context_by_step,context_by_step2)))
# print(torch.all(torch.isclose(attn_by_step,attn_by_step2)))
# assert False


## DEBUG : can check this with block version of attention
# class Attn(nn.Module):
    
#     def __init__(self, method, hidden_size):
#         super(Attn, self).__init__()
        
#         self.method = method
#         self.hidden_size = hidden_size
        
#         if self.method == 'general':
#             self.attn = nn.Linear(self.hidden_size, hidden_size)

#         elif self.method == 'concat':
#             self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
#             self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

#     def forward(self, hidden, encoder_outputs):
#         # Input
#         #  hidden : 1 x batch_size x hidden_size
#         #  encoder_outputs : (max_input_length x batch_size x embedding_size) 

#         max_len = encoder_outputs.size(0)
#         this_batch_size = encoder_outputs.size(1) 

#         # Create variable to store attention energies
#         attn_energies = torch.zeros(this_batch_size, max_len) # B x S
#         if hidden.is_cuda: attn_energies = attn_energies.cuda()

#         # For each batch of encoder outputs
#         for b in range(this_batch_size):
#             # Calculate energy for each encoder output
#             for i in range(max_len):
#                 attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

#         # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
#         return F.softmax(attn_energies).unsqueeze(1)
    
#     def score(self, hidden, encoder_output):
        
#         if self.method == 'dot':
#             hidden = hidden.flatten()
#             encoder_output = encoder_output.flatten()
#             energy = hidden.dot(encoder_output)
#             return energy
        
#         elif self.method == 'general':
#             energy = self.attn(encoder_output)
#             energy = hidden.dot(energy)
#             return energy
        
#         elif self.method == 'concat':
#             energy = self.attn(torch.cat((hidden, encoder_output), 1))
#             energy = self.v.dot(energy)
#             return energy           
