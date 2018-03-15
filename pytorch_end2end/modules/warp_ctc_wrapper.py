import torch
import torch.nn as nn


class WarpCTCLoss(nn.Module):
    def __init__(self, reduce=False):
        super(WarpCTCLoss, self).__init__()
        from warpctc_pytorch import _CTC
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

        # warp_ctc: logits: sequence_length * batch_size * num_labels, targets: flat
        batch_size = targets_lengths.size()[0]
        targets_flat = torch.cat([targets[i, :targets_lengths[i].data[0]] for i in range(batch_size)])

        costs = self.ctc(logits.transpose(0, 1), targets_flat.int(), logits_lengths.int(), targets_lengths.int())

        if self.reduce:
            return costs.sum()
        return costs
