import torch
from torch.nn import functional
from torch.autograd import Variable

# Code adapted from:
#   https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    # seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    length = torch.LongTensor(length)
    # length = Variable(torch.LongTensor(length))
    if target.is_cuda:
        length = length.cuda()
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    #print("logits\n", logits)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat,dim=1)
    #print("log probs flat\n", log_probs_flat)
    # target_flat: (batch * max_len, 1)
    #print("target\n", target)
    target_flat = target.view(-1, 1)
    #print("target_flat\n", target_flat)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    #print('losses_flat\n', losses_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    #print("pre mask losses:\n", losses)

    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    #print('mask', mask)
    #losses = losses * mask.float()
    losses[~mask] = 0
    #print('losses', losses)
    #loss = (losses.sum(1) / length.float()).mean()
    #print(loss == losses.sum() / length.float().sum())
    loss = losses.sum() / length.float().sum()
    #import pdb; pdb.set_trace()
    return loss


# def masked_cross_entropy(logits, target, length):
#     length = torch.LongTensor(length)
#     # length = Variable(torch.LongTensor(length))
#     if target.is_cuda:
#         length = length.cuda()
#     """
#     Args:
#         logits: A Variable containing a FloatTensor of size
#             (batch, max_len, num_classes) which contains the
#             unnormalized probability for each class.
#         target: A Variable containing a LongTensor of size
#             (batch, max_len) which contains the index of the true
#             class for each corresponding step.
#         length: A Variable containing a LongTensor of size (batch,)
#             which contains the length of each data in a batch.

#     Returns:
#         loss: An average loss value masked by the length.
#     """

#     # logits_flat: (batch * max_len, num_classes)
#     #logits_flat = logits.view(-1, logits.size(-1))
#     # log_probs_flat: (batch * max_len, num_classes)
#     log_probs = functional.log_softmax(logits,dim=2)
#     # target_flat: (batch * max_len, 1)
#     #target_flat = target.view(-1, 1)
#     # losses_flat: (batch * max_len, 1)
#     losses = -torch.gather(log_probs, dim=2, index=target)
#     # losses: (batch, max_len)
#     #losses = losses_flat.view(*target.size())
#     # mask: (batch, max_len)
#     mask = sequence_mask(sequence_length=length, max_len=target.size(1))
#     losses = losses * mask.float()
#     loss = losses.sum(1).mean()
#     #loss = losses.sum() / length.float().sum()
#     return loss