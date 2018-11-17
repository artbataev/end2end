import torch.nn as nn
import torch.nn.functional as F

from ..functions.ctc_loss import CTCLossFunction


class CTCLoss(nn.Module):
    """
    Criterion to compute CTC Loss as described in ``http://www.cs.toronto.edu/~graves/icml_2006.pdf``

    :param size_average:
    :param reduce:
    :param after_logsoftmax:
    :param time_major: if logits are time major (or batch major), default ``True``
    :param blank_idx: id of blank label, default ``0``
    """
    def __init__(self, size_average=None, reduce=None, after_logsoftmax=False, time_major=True, blank_idx=0):
        super(CTCLoss, self).__init__()
        self._blank_index = blank_idx
        self._reduce = reduce
        self._size_average = size_average
        self._after_logsoftmax = after_logsoftmax
        self._time_major = time_major

    def forward(self, logits, targets, logits_lengths, targets_lengths):
        """
        Compute CTC Loss

        :param logits:
        :param targets:
        :param logits_lengths:
        :param targets_lengths:
        :return: tensor with CTC loss of shape ``(batch_size,)`` if ``reduce is None`` else of shape ``(1,)``
        """
        if self._after_logsoftmax:
            logits_logsoftmax = logits
        else:
            logits_logsoftmax = F.log_softmax(logits, dim=2)

        if self._time_major:
            logits_logsoftmax = logits_logsoftmax.permute(1, 0, 2)
        # shape of logits_logsoftmax now: batch_size, sequence_length, alphabet_size

        loss = CTCLossFunction().apply(logits_logsoftmax, targets, logits_lengths, targets_lengths, self._blank_index)
        
        if self._reduce is not None:
            if self._size_average:
                return loss.mean()
            else:
                return loss.sum()
        return loss
