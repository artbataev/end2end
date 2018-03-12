import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..utils.alignment import get_alignment_3d


class AlignedTargetsLoss(nn.Module):
    def __init__(self, is_ctc, reduce=True):
        super().__init__()
        self._reduce = reduce
        self._is_ctc = is_ctc

    def forward(self, logits, targets, logits_lengths, targets_lengths):
        """
        :param logits: batch_size * sequence_length * num_labels
        :param targets: batch_size * sequence_length
        :param logits_lengths: batch_size
        :param labels_lengths: batch_size
        :return:
        """
        logits_logsoftmax = nn.LogSoftmax(dim=2)(logits)
        targets_new = get_alignment_3d(logits_logsoftmax, targets, logits_lengths, targets_lengths,
                                       is_ctc=self._is_ctc)
        if logits_logsoftmax.is_cuda:
            targets_new = targets_new.cuda(logits_logsoftmax.get_device())

        loss = F.nll_loss(logits_logsoftmax, Variable(targets_new), reduce=False)

        batch_size = logits_logsoftmax.size()[0]
        sequence_length = logits_logsoftmax.size()[1]
        mask = torch.ByteTensor(batch_size, sequence_length).ones_()
        for i in range(batch_size):
            mask[i, logits_lengths[i]:] = 0
        if self._reduce:
            loss = torch.masked_select(loss, mask).mean()
        else:
            loss *= mask.float()
        return loss
