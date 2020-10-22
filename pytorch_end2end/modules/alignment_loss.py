import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pytorch_end2end.utils.alignment import get_alignment_3d


class AlignedTargetsLoss(nn.Module):
    def __init__(self, is_ctc, reduce=True, reduce_by_sequence=False):
        super().__init__()
        self._reduce = reduce
        self._is_ctc = is_ctc
        self._reduce_by_sequence = reduce_by_sequence

    def forward(self, logits, targets, logits_lengths, targets_lengths):
        """
        :param logits: batch_size * sequence_length * num_labels
        :param targets: batch_size * sequence_length, fill with -1 if ignored label
        :param logits_lengths: batch_size
        :param labels_lengths: batch_size
        :return:
        """
        logits_logsoftmax = nn.LogSoftmax(dim=-1)(logits)
        cur_device = logits_logsoftmax.get_device()
        targets_new = get_alignment_3d(logits_logsoftmax, targets, logits_lengths, targets_lengths,
                                       is_ctc=self._is_ctc)
        if logits_logsoftmax.is_cuda:
            targets_new = targets_new.cuda(cur_device)
        batch_size, sequence_length, _ = logits.size()

        if self._reduce:
            loss = F.nll_loss(logits_logsoftmax.view(batch_size * sequence_length, -1),
                              Variable(targets_new).view(batch_size * sequence_length), reduce=True)
        else:
            loss = F.nll_loss(logits_logsoftmax.view(batch_size * sequence_length, -1),
                              Variable(targets_new).view(batch_size * sequence_length),
                              reduce=False).view(batch_size, sequence_length)
            if self._reduce_by_sequence:
                if not logits_lengths.is_cuda:
                    logits_lengths = logits_lengths.cuda(cur_device)
                loss = loss.sum(dim=-1) / logits_lengths.float()
        return loss
