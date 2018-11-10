import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .ctc_loss import CTCLoss as WarpCTCLoss
from ..utils.alignment import get_alignment_3d


class CTCLossSegmented(nn.Module):
    def __init__(self, space_idx, blank_idx=0, reduce=False, min_word_length=3):
        super().__init__()
        self.reduce = reduce
        self.space_idx = space_idx
        self.blank_idx = blank_idx
        self.ctc = WarpCTCLoss(reduce=False)
        self.min_word_length = min_word_length

    def forward(self, logits, targets, logits_lengths, targets_lengths):
        logits_logsoftmax = F.log_softmax(logits, dim=2)
        targets_aligned = get_alignment_3d(logits_logsoftmax, targets, logits_lengths, targets_lengths, is_ctc=True)
        predictions_argmax = torch.max(logits, dim=2)[1].data.cpu()

        batch_size = logits.size()[0]
        sequence_length = logits.size()[1]

        mask = torch.ByteTensor(batch_size, sequence_length)
        mask.fill_(1)
        for i in range(batch_size):
            start = logits_lengths.data[i]
            if start < sequence_length:
                mask[i, start:] = 0

        targets_well_recognized = (targets_aligned == predictions_argmax) * mask

        # cnt_well_recognized = targets_well_recognized.float().sum()
        # if cnt_well_recognized < targets_lengths.data.float().sum() * batch_size / 2:
        #     # our model is bad, do not try to segment
        #     return self.ctc(logits, targets, logits_lengths, targets_lengths)

        indices_to_segment = [[0, ] for _ in range(batch_size)]
        num_segments = 0
        for i in range(batch_size):
            start_space = -1
            all_word_well_recognized = True
            current_word_length=0
            last_word_char = -1
            last_word_char_is_blank = False
            for t in range(logits_lengths.data[i]):
                if targets_well_recognized[i, t] == 0:
                    all_word_well_recognized = False
                    continue

                if targets_aligned[i, t] == self.space_idx:
                    # create segment if found
                    if all_word_well_recognized and current_word_length >= self.min_word_length:
                        if start_space != -1 and indices_to_segment[i][-1] != start_space:
                            if start_space - indices_to_segment[i][-1] > 1:
                                num_segments += 1
                            indices_to_segment[i].append(start_space)
                            num_segments += 1
                        if t > 0:
                            indices_to_segment[i].append(t)
                            num_segments += 1
                            if t - start_space > 1:
                                num_segments += 1
                    start_space = t
                    all_word_well_recognized = True
                    current_word_length = 0
                    last_word_char = -1
                    last_word_char_is_blank = False
                elif targets_aligned[i, t] == self.blank_idx:
                    last_word_char_is_blank = True
                else: # not blank
                    if last_word_char_is_blank or targets_aligned[i, t] != last_word_char:
                        current_word_length += 1
                    last_word_char = targets_aligned[i, t]
                    last_word_char_is_blank = False

            if indices_to_segment[i][-1] != logits_lengths.data[i] - 1:
                indices_to_segment[i].append(logits_lengths.data[i] - 1)
                num_segments += 1

        assert num_segments >= batch_size
        if num_segments == batch_size:
            # no new segments found
            loss = self.ctc(logits, targets, logits_lengths, targets_lengths)
            return loss

        logits_new = Variable(torch.zeros(num_segments, logits.size()[1], logits.size()[2]), requires_grad=False)
        if logits.is_cuda:
            logits_new = logits_new.cuda(logits.get_device())
        targets_new = Variable(torch.LongTensor(num_segments, targets.size()[1]))
        batch_ids_new = []
        logits_lengths_new = []
        targets_lengths_new = []

        new_i = 0
        for i in range(batch_size):
            if len(indices_to_segment[i]) <= 2:  # use full segment
                logits_new[new_i] = logits[i]
                batch_ids_new.append(i)
                targets_new[new_i] = targets[i]
                targets_lengths_new.append(targets_lengths.data[i])
                logits_lengths_new.append(logits_lengths.data[i])
                new_i += 1
            else:
                for k, start in enumerate(indices_to_segment[i][:-1]):
                    if start != 0:
                        batch_ids_new.append(i)
                        targets_new[new_i, 0] = targets_aligned[i, start]
                        logits_new[new_i, 0:1] = logits[i, start:start + 1]
                        logits_lengths_new.append(1)
                        targets_lengths_new.append(1)
                        start += 1
                        new_i += 1
                    next = indices_to_segment[i][k + 1]
                    if k < len(indices_to_segment[i]) - 2:
                        next -= 1
                    if next < start:
                        continue
                    batch_ids_new.append(i)
                    logits_new[new_i, :(next-start + 1)] = logits[i, start:next+1]
                    current_targets = Variable(torch.LongTensor(
                        [c for c, _ in itertools.groupby(targets_aligned[i, start:next+1].tolist()) if
                         c != self.blank_idx]))
                    targets_new[new_i, :current_targets.size()[0]] = current_targets
                    logits_lengths_new.append(next - start + 1)
                    targets_lengths_new.append(current_targets.size()[0])
                    new_i += 1

        assert num_segments == len(batch_ids_new), "expected {}, get {} segments: {}".format(num_segments, len(batch_ids_new),
                                                                                             indices_to_segment)
        logits_lengths_new = Variable(torch.LongTensor(logits_lengths_new), requires_grad=False)
        targets_lengths_new = Variable(torch.LongTensor(targets_lengths_new), requires_grad=False)

        segmented_loss = self.ctc(logits_new, targets_new, logits_lengths_new, targets_lengths_new)
        loss = torch.zeros(batch_size)
        if logits.is_cuda:
            loss = loss.cuda(logits.get_device())
        loss = Variable(loss, requires_grad=False)
        for new_i, i in enumerate(batch_ids_new):
            loss[i] = segmented_loss[new_i] + loss[i]
        return loss
