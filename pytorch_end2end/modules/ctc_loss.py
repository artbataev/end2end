import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from importlib import import_module

from pytorch_end2end.functions.forward_backward import ForwardBackwardLossFunction

if "DEBUG_E2E" in os.environ:
    module_base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    build_path = os.path.join(module_base, os.getenv("DEBUG_E2E"))
    sys.path.insert(0, build_path)


class ForwardBackwardLossBase(nn.Module):
    def __init__(self, size_average=None, reduce=None, after_logsoftmax=False, time_major=False, blank_idx=0):
        super().__init__()
        self._blank_idx = blank_idx
        self._reduce = reduce
        self._size_average = size_average
        self._after_logsoftmax = after_logsoftmax
        self._time_major = time_major
        self._engine = None

    def forward(self, logits, targets, logits_lengths, targets_lengths):
        """
        Computes Loss

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

        loss = ForwardBackwardLossFunction().apply(self._engine,
                                                   logits_logsoftmax,
                                                   targets,
                                                   logits_lengths,
                                                   targets_lengths)

        if self._reduce:
            if self._size_average:
                return loss.mean()
            else:
                return loss.sum()
        return loss


class CTCLoss(ForwardBackwardLossBase):
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
        super().__init__(size_average, reduce, after_logsoftmax, time_major, blank_idx)
        cpp_ctc_loss = import_module("cpp_ctc_loss")
        self._engine = cpp_ctc_loss.CTCLossEngine(self._blank_idx)


class GramCTCLoss(CTCLoss):
    """
   Criterion to compute Gram-CTC Loss as described in `<https://arxiv.org/abs/1703.00096>`_

   :param blank_idx: id of blank label, only ``0`` supported
   :param num_base_labels:
   :param total_labels:
   :param labels2ids: dict for mapping label_id -> base labels ids
   :param size_average: if compute average loss (only if reduce is True)
   :param reduce: if compute mean or average loss (if None, returns full tensor of shape ``(batch_size,)``)
   :param after_logsoftmax: if logsoftmax is used before passing neural network outputs \n
       (else takes pure network outputs)
   :param time_major: if logits are time major (or batch major), default ``True``
   """

    def __init__(self, blank_idx, num_base_labels, total_labels, label2ids,
                 size_average=None, reduce=None, after_logsoftmax=False, time_major=False):
        super().__init__(size_average, reduce, after_logsoftmax, time_major, blank_idx)
        if self._blank_idx != 0:
            raise NotImplementedError
        self._num_base_labels = num_base_labels
        self._total_labels = total_labels
        self._label2ids = label2ids
        cpp_gram_ctc_loss = import_module("cpp_gram_ctc_loss")
        self._engine = cpp_gram_ctc_loss.GramCTCLossEngine(self._blank_idx,
                                                           self._num_base_labels,
                                                           self._total_labels,
                                                           self._label2ids)
        raise NotImplementedError
