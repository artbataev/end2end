from ..functions.ctc_without_blank import CTCWithoutBlankLossFunction
import torch
import torch.nn as nn

class CTCWithoutBlankLoss(nn.Module):
    def __init__(self, reduce=True, after_softmax=False, space_idx=-1):
        super(CTCWithoutBlankLoss, self).__init__()
        self._space_idx = space_idx
        self._reduce = reduce
        self._after_softmax = after_softmax

    def forward(self, inputs, targets_flat, input_sizes, targets_sizes):
        # seq_len, batch_size, alphabet_size = inputs.size()
        if self._after_softmax:
            inputs_logsoftmax = torch.log(inputs)
        else:
            inputs_logsoftmax = nn.LogSoftmax(dim=2)(inputs)

        loss = CTCWithoutBlankLossFunction().apply(inputs_logsoftmax, targets_flat, input_sizes, targets_sizes, self._space_idx)
        if self._reduce:
            return loss.sum()
        else:
            return loss