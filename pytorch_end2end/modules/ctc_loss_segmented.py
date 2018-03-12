from .warp_ctc_wrapper import WarpCTCLoss
import torch.nn as nn
from ..utils.alignment import get_alignment_3d
import torch.nn.functional as F
from torch.autograd import Variable
import torch

class CTCLossSegmented(nn.Module):
    def __init__(self, space_idx, blank_idx=0, reduce=False):
        super().__init__()
        self.reduce = reduce
        self.space_idx = space_idx
        self.blank_idx = blank_idx
        self.ctc = WarpCTCLoss(reduce=False)

    def forward(self, logits, targets, logits_lengths, targets_lengths):
        logits_logsoftmax = F.log_softmax(logits, dim=2)
        targets_aligned = get_alignment_3d(logits_logsoftmax, targets, logits_lengths, targets_lengths, is_ctc=True)
        targets_argmax = torch.max(logits, dim=2)[1]


        batch_size = logits.size()[0]
        sequence_length = logits.size()[1]

        mask = torch.ByteTensor(batch_size, sequence_length).ones_()
        for i in range(batch_size):
            mask[i, logits_lengths[i]:] = 0

        targets_well_recognized = (targets_aligned == targets_argmax) * mask

        logits_new = []
        targets_new = []
        logits_lengths_new = []
        targets_lengths_new = []
        for i in range(batch_size):
            start = 0
            cnt_symbols = 0
            for t in range(logits_lengths[i]):

                # segment
                raise NotImplementedError

        logits_new = torch.cat(logits_new, dim=0) # ? pad
        targets_new = torch.cat(targets_new, dim=0) # ? pad
        logits_lengths_new = Variable(torch.LongTensor(logits_lengths_new), requires_grad=False)
        targets_lengths_new = Variable(torch.LongTensor(targets_lengths_new), requires_grad=False)

        loss = self.ctc(logits_new, targets_new, logits_lengths_new, targets_lengths_new)
        loss = loss.sum() / batch_size
        return loss
