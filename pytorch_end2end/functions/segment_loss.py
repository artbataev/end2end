import queue
import threading

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import numba

@numba.jit(nogil=True)
def _get_best_labeling_asg_1d(logits, targets):
    """
    :param logits: seq_len * output_dim, after LogSoftmax!
    :param targets:
    :return:
    """
    targets_len = targets.shape[0]
    prediction_len = logits.shape[0]
    best_labeling = np.zeros(prediction_len, dtype=np.int64)

    if prediction_len == 1:
        best_labeling[0] = targets[0]
        return best_labeling

    # forward
    alpha = np.zeros((targets_len, prediction_len), dtype=np.float64)
    path_alpha = np.zeros_like(alpha, dtype=np.int64)
    alpha[0, 0] = logits[0, targets[0]]
    for k in range(1, prediction_len):
        start = max(0, targets_len - (prediction_len - k))
        end = min(k + 1, targets_len)
        alpha[start:end, k] = alpha[start:end, k - 1]
        path_alpha[start:end, k] = np.arange(start, end)
        for i in range(start, end):
            current_label = targets[i]
            if i > 0:
                if alpha[i - 1, k - 1] > alpha[i, k]:
                    alpha[i, k] = alpha[i - 1, k - 1]
                    path_alpha[i, k] = i - 1
            alpha[i, k] += logits[k, current_label]  # log-scale, +

    i = targets_len - 1
    for k in range(prediction_len - 1, -1, -1):
        best_labeling[k] = targets[i]
        i = path_alpha[i, k]

    return best_labeling

@numba.jit(nogil=True)
def _get_best_labeling_ctc_1d(logits, targets):
    """
    with respect to blank symbol
    :param logits: seq_len * output_dim, after LogSoftmax!
    :param targets:
    :return:
    """
    blank = 0
    targets_len = targets.shape[0]
    prediction_len = logits.shape[0]
    extended_targets_len = targets_len * 2 + 1
    extended_targets = np.ones(extended_targets_len, dtype=int) * blank
    for i in range(targets_len):
        extended_targets[2 * i + 1] = targets[i]
    best_labeling = np.zeros(prediction_len, dtype=np.int64)

    if extended_targets_len == 1 or prediction_len == 1:
        if extended_targets_len == 1:  # only blank
            return best_labeling
        # prediction_len == 1
        best_labeling[0] = targets[0]
        return best_labeling

    # forward
    alpha = np.zeros((extended_targets_len, prediction_len), dtype=np.float64)
    path_alpha = np.zeros_like(alpha, dtype=np.int64)
    alpha[0, 0] = logits[0, extended_targets[0]]
    alpha[1, 0] = logits[0, extended_targets[1]]
    for k in range(1, prediction_len):
        start = max(0, extended_targets_len - 2 * (prediction_len - k))
        end = min(k * 2 + 2, extended_targets_len)
        alpha[start:end, k] = alpha[start:end, k - 1]
        path_alpha[start:end, k] = np.arange(start, end)
        for i in range(start, end):
            current_label = extended_targets[i]
            if i > 0:
                if alpha[i - 1, k - 1] > alpha[i, k]:
                    alpha[i, k] = alpha[i - 1, k - 1]
                    path_alpha[i, k] = i - 1
                if current_label != blank and i - 2 > 0 and extended_targets[i - 2] != current_label and alpha[
                    i - 2, k - 1] > alpha[i, k]:
                    alpha[i, k] = alpha[i - 2, k - 1]
                    path_alpha[i, k] = i - 2
            alpha[i, k] += logits[k, current_label]  # log-scale, +

    i = extended_targets_len - 1
    if alpha[i - 1, prediction_len - 1] > alpha[i, prediction_len - 1]:
        i = i - 1
    for k in range(prediction_len - 1, -1, -1):
        best_labeling[k] = extended_targets[i]
        i = path_alpha[i, k]

    return best_labeling


def get_best_labelings_3d(logits_logsoftmax, targets_flat, input_sizes, targets_sizes, is_ctc=True):
    targets_np = targets_flat.cpu().data.numpy()
    seq_len, batch_size, _ = logits_logsoftmax.size()
    targets_sizes_end = np.cumsum(targets_sizes.cpu().data.numpy())
    targets_sizes_start = targets_sizes_end - targets_sizes.cpu().data.numpy()

    logits_logsoftmax_numpy = logits_logsoftmax.cpu().data.numpy()
    inputs_sizes_numpy = input_sizes.cpu().data.numpy()

    que = queue.Queue()
    threads = []
    if is_ctc:
        func_1d = _get_best_labeling_ctc_1d
    else:
        func_1d = _get_best_labeling_asg_1d

    for i in range(batch_size):
        t = threading.Thread(target=lambda q, i, *args: q.put((i, func_1d(*args))),
                             args=(que, i, logits_logsoftmax_numpy[:inputs_sizes_numpy[i], i, :],
                                   targets_np[targets_sizes_start[i]:targets_sizes_end[i]]))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    targets_sizes_new_end = np.cumsum(input_sizes.cpu().data.numpy())
    targets_sizes_new_start = targets_sizes_new_end - input_sizes.cpu().data.numpy()
    targets_new = torch.LongTensor(int(targets_sizes_new_end[-1])).zero_()
    while not que.empty():
        i, best_labeling = que.get()
        targets_new[targets_sizes_new_start[i]: targets_sizes_new_end[i]] = torch.LongTensor(best_labeling)

    return Variable(targets_new)


class HardtargetsLoss(nn.Module):
    def __init__(self, is_ctc, reduce=True):
        super().__init__()
        self._reduce = reduce
        self._is_ctc = is_ctc

    def forward(self, inputs, targets_flat, input_sizes, targets_sizes):
        """
        :param inputs: seq_length * batch_size x output_dim
        :param targets_flat:
        :param input_sizes:
        :param targets_sizes:
        :return:
        """
        logits_logsoftmax = nn.LogSoftmax(dim=2)(inputs)
        batch_size = targets_sizes.size()[0]
        targets_flat_new = get_best_labelings_3d(logits_logsoftmax, targets_flat, input_sizes, targets_sizes,
                                                  is_ctc=self._is_ctc)
        logits_logsoftmax_flat = torch.cat([logits_logsoftmax[:input_sizes.data[i], i, :] for i in range(batch_size)])
        if logits_logsoftmax.is_cuda:
            targets_flat_new = targets_flat_new.cuda(logits_logsoftmax.get_device())
        loss = nn.NLLLoss(reduce=self._reduce, size_average=False)(logits_logsoftmax_flat, targets_flat_new) / targets_flat.size()[0] * batch_size
        # loss2 = nn.CrossEntropyLoss(reduce=self._reduce)(torch.cat([inputs[:input_sizes.data[i], i, :] for i in range(batch_size)]), targets_flat_new)
        return loss
