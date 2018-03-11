import torch
import torch.nn as nn

from ..functions.ctc_loss import CTCLossFunction


class CTCLoss(nn.Module):
    def __init__(self, reduce=True, after_softmax=False, blank_idx=0):
        super(CTCLoss, self).__init__()
        self._blank_index = blank_idx
        self._reduce = reduce
        self._after_softmax = after_softmax

    def forward(self, logits, targets, logits_lengths, targets_lengths):
        # seq_len, batch_size, alphabet_size = inputs.size()
        if self._after_softmax:
            logits_logsoftmax = torch.log(logits)
        else:
            logits_logsoftmax = nn.LogSoftmax(dim=2)(logits)

        loss = CTCLossFunction().apply(logits_logsoftmax, targets, logits_lengths, targets_lengths, self._blank_index)
        
        if self._reduce:
            return loss.sum()
        else:
            return loss
