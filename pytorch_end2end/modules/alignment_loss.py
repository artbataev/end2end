import torch.nn as nn
import torch.nn.functional as F

from pytorch_end2end.utils.alignment import get_alignment_3d


class AlignedTargetsLoss(nn.Module):
    def __init__(self, is_ctc, ignore_blank=False):
        super().__init__()
        self._is_ctc = is_ctc
        self._ignore_blank = ignore_blank

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        :param log_probs: batch_size * sequence_length * num_labels
        :param targets: batch_size * sequence_length, fill with -1 if ignored label
        :param input_lengths: batch_size
        :param target_lengths: batch_size
        :return:
        """
        cur_device = log_probs.get_device()
        targets_new = get_alignment_3d(log_probs, targets, input_lengths, target_lengths,
                                       is_ctc=self._is_ctc)
        if log_probs.is_cuda:
            targets_new = targets_new.cuda(cur_device)
        batch_size, sequence_length, _ = log_probs.shape
        if self._ignore_blank:
            targets_new[targets_new == 0] = -100
        loss = F.nll_loss(log_probs.reshape(batch_size * sequence_length, -1),
                          targets_new.reshape(batch_size * sequence_length),
                          reduction="none", ignore_index=-100).reshape(batch_size, sequence_length)
        loss = loss.sum(dim=-1) / input_lengths
        return loss
