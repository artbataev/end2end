import torch.nn as nn
import torch.nn.functional as F

from ..functions.ctc_loss import CTCLossFunction


class CTCLoss(nn.Module):
    """
    Criterion to compute CTC Loss as described in `<http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_

    :param size_average: if compute average loss (only if reduce is True)
    :param reduce: if compute mean or average loss (if None, returns full tensor of shape ``(batch_size,)``)
    :param after_logsoftmax: if logsoftmax is used before passing neural network outputs \n
        (else takes pure network outputs)
    :param time_major: if logits are time major (or batch major), default ``True``
    :param blank_idx: id of blank label, default ``0``
    """

    def __init__(self, size_average=None, reduce=None, after_logsoftmax=False, time_major=False, blank_idx=0):
        super(CTCLoss, self).__init__()
        self._blank_index = blank_idx
        self._reduce = reduce
        self._size_average = size_average
        self._after_logsoftmax = after_logsoftmax
        self._time_major = time_major

    def forward(self, logits, targets, logits_lengths, targets_lengths):
        """
        Computes CTC Loss

        :param logits: Float or Double Tensor (network output)
            of shape ``(sequence_length, batch_size, alphabet_size)`` if ``time_major`` is True,
            else of shape ``(batch_size, sequence_length, alphabet_size)``
        :param targets: Tensor with targets of shape ``(batch_size, targets_sequence_length)``
        :param logits_lengths: Tensor of shape ``(batch_size,)`` with lengths of sequences
        :param targets_lengths: Tensor of shape ``(batch_size,)`` with lengths of target sequences
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
