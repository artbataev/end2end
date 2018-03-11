from ..functions.ctc_without_blank import CTCWithoutBlankLossFunction
import torch
import torch.nn as nn

class CTCWithoutBlankLoss(nn.Module):
    def __init__(self, reduce=True, after_softmax=False, space_idx=-1):
        super(CTCWithoutBlankLoss, self).__init__()
        self._space_idx = space_idx
        self._reduce = reduce
        self._after_softmax = after_softmax

    def forward(self, logits, targets, logits_lengths, targets_lengths):
        if self._after_softmax:
            logits_logsoftmax = torch.log(logits)
        else:
            logits_logsoftmax = nn.LogSoftmax(dim=2)(logits)

        loss = CTCWithoutBlankLossFunction().apply(logits_logsoftmax, targets, logits_lengths, targets_lengths, self._space_idx)
        if self._reduce:
            return loss.sum()
        else:
            return loss