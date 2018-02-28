import torch
import torch.nn as nn

from warpctc_pytorch import _CTC

class WarpCTCLoss(nn.Module):
    def __init__(self, reduce=False):
        super(WarpCTCLoss, self).__init__()
        self.ctc = _CTC.apply
        self.reduce = reduce

    def forward(self, logits, targets, logits_lengths, targets_lengths):
        """

        :param logits: batch_size * sequence_length * num_labels
        :param targets: batch_size * sequence_length
        :param logits_lengths: batch_size
        :param labels_lengths: batch_size
        :return:
        """

        batch_size = targets_lengths.size()[0]
        targets_lengths_cumsum = torch.cumsum(targets_lengths).data.numpy()
        targets_flat = torch.LongTensor(targets_lengths_cumsum[-1])
        for i in range(len(batch_size)):
            if i > 0:
                targets_flat[targets_lengths_cumsum[i - 1]: targets_lengths_cumsum[i]] = targets[i, :targets_lengths[i].data].data
            else:
                targets_lengths_cumsum[:targets_lengths_cumsum[i]] = targets[i, :targets_lengths[i].data].data

        # warp_ctc: logits: sequence_length * batch_size, targets: flat
        costs = self.ctc(logits.transpose(0, 1), targets_flat, logits_lengths, targets_lengths, self.size_average)
        if self.reduce:
            return costs.sum()
        return costs
