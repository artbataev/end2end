import torch
import torch.nn as nn

from ..functions.ctc_loss import CTCLossFunction


class CTCLoss(nn.Module):
    def __init__(self, reduce=True, after_softmax=False, blank_idx=0):
        super(CTCLoss, self).__init__()
        self._blank_index = blank_idx
        self._reduce = reduce
        self._after_softmax = after_softmax

    def forward(self, inputs, targets_flat, input_sizes, targets_sizes):
        # seq_len, batch_size, alphabet_size = inputs.size()
        if self._after_softmax:
            inputs_logsoftmax = torch.log(inputs)
        else:
            inputs_logsoftmax = nn.LogSoftmax(dim=2)(inputs)

        loss = CTCLossFunction().apply(inputs_logsoftmax, targets_flat, input_sizes, targets_sizes, self._blank_index)
        # if DEBUG:
        #     print("q {} {} {} {}".format(inputs, inputs_logsoftmax, targets_flat, loss))
        if self._reduce:
            return loss.sum()
        else:
            return loss
