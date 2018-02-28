import queue
import threading

import numba
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function, Variable

from .utils import log_sum_exp


@numba.jit(nogil=True)
def _asg_loss(logits, targets, space_idx):
    """
    http://www.cs.toronto.edu/~graves/icml_2006.pdf
    :param logits: numpy array, sequence_len * num_labels
    :param targets: numpy array, target labels
    :param blank: blank index
    :return: loss (float), gradient (same shape as logits)
    """
    targets_len = targets.shape[0]
    prediction_len = logits.shape[0]
    num_labels = logits.shape[1]

    using_additional_spaces = False
    if targets_len == 0 or (targets_len == 1 and targets[0] == space_idx):
        extended_targets_len = 1
        extended_targets = np.ones(extended_targets_len, dtype=np.int64) * space_idx
    elif space_idx == -1:
        extended_targets_len = targets_len
        extended_targets = targets
    else:
        using_additional_spaces = True
        extended_targets_len = targets_len + 2
        extended_targets = np.ones(extended_targets_len, dtype=np.int64) * space_idx
        extended_targets[1:extended_targets_len - 1] = targets
    # alpha and beta computation
    # forward - alpha
    log_alpha = np.zeros((extended_targets_len, prediction_len))
    log_alpha[:] = -np.inf  # numba bugfix instead of log_alpha.fill(-np.inf)
    if prediction_len > 1 or extended_targets_len == 1:
        log_alpha[0, 0] = logits[0, extended_targets[0]]
    if extended_targets_len > 1 and using_additional_spaces:
        log_alpha[1, 0] = logits[0, extended_targets[1]]
    for t in range(1, prediction_len):  # timesteps
        start = max(0, extended_targets_len - prediction_len + t - 1) if using_additional_spaces \
            else max(0, extended_targets_len - prediction_len + t)
        end = min(t + 2, extended_targets_len) if using_additional_spaces else min(t + 1, extended_targets_len)
        log_alpha[start:end, t] = log_alpha[start:end, t - 1]
        for j in range(start, end):
            current_label = extended_targets[j]
            if j > 0:
                log_alpha[j, t] = log_sum_exp(log_alpha[j, t], log_alpha[j - 1, t - 1])
            log_alpha[j, t] += logits[t, current_label]
    if extended_targets_len > 1 and using_additional_spaces:
        loss_forward = log_sum_exp(log_alpha[extended_targets_len - 1, prediction_len - 1],
                                   log_alpha[extended_targets_len - 2, prediction_len - 1])
    else:
        loss_forward = log_alpha[extended_targets_len - 1, prediction_len - 1]

    # backward - beta
    log_beta = np.zeros((extended_targets_len, prediction_len))
    log_beta[:] = -np.inf  # numba bugfix instead of log_beta.fill(-np.inf)
    if prediction_len > 1 or extended_targets_len == 1:
        log_beta[extended_targets_len - 1, prediction_len - 1] = 0
    if extended_targets_len > 1 and using_additional_spaces:
        log_beta[extended_targets_len - 2, prediction_len - 1] = 0
    for t in range(prediction_len - 2, -1, -1):  # timesteps
        start = max(0, extended_targets_len - prediction_len + t - 1) if using_additional_spaces \
            else max(0, extended_targets_len - prediction_len + t)
        end = min(t + 2, extended_targets_len) if using_additional_spaces else min(t + 1, extended_targets_len)
        for j in range(start, end):
            log_beta[j, t] = log_beta[j, t + 1] + logits[t + 1, extended_targets[j]]
            if j < extended_targets_len - 1:
                log_beta[j, t] = log_sum_exp(log_beta[j, t],
                                             log_beta[j + 1, t + 1] + logits[t + 1, extended_targets[j + 1]])

    grad = np.zeros_like(logits)
    alpha_beta = log_alpha + log_beta

    for t in range(prediction_len):
        prob_sum = np.zeros(num_labels)
        prob_sum[:] = -np.inf
        for i in range(extended_targets_len):
            current_label = extended_targets[i]
            prob_sum[current_label] = log_sum_exp(prob_sum[current_label], alpha_beta[i, t])
        for l in range(num_labels):
            negative_term = np.exp(prob_sum[l] - loss_forward)
            grad[t, l] = np.exp(logits[t, l]) - negative_term
    return -loss_forward, grad


def _asg_3d_loss(inputs, targets_flat, input_sizes, targets_sizes, space_idx=-1):
    batch_size = len(input_sizes)
    grads = np.zeros_like(inputs)

    targets_sizes_end = np.cumsum(targets_sizes)
    targets_sizes_start = targets_sizes_end - targets_sizes

    losses = np.zeros(batch_size)

    # parallel computation, threading - because gil is released with numba.jit(nogil=True)
    que = queue.Queue()
    threads = []
    for i in range(batch_size):
        t = threading.Thread(target=lambda q, i, *args: q.put((i, _asg_loss(*args))),
                             args=(que, i, inputs[:input_sizes[i], i, :],
                                   targets_flat[targets_sizes_start[i]:targets_sizes_end[i]], space_idx))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    while not que.empty():
        i, (loss, grad) = que.get()
        grads[:input_sizes[i], i, :] = grad
        losses[i] = loss

    # iterative computation
    # for i in range(batch_size):
    #     loss, grad = _ctc_loss(inputs[:input_sizes[i], i, :], targets_flat[targets_sizes_start[i]:targets_sizes_end[i]])
    #     grads[:input_sizes[i], i, :] = grad
    #     losses[i] = loss

    return losses, grads


class CTCWithoutBlankLossFunction(Function):
    @staticmethod
    def forward(ctx, inputs, targets_flat, input_sizes, targets_sizes, space_idx=-1):
        # inputs: expected shape of seqLength x batchSize x alphabet_size, after logsoftmax!
        loss, grads = _asg_3d_loss(inputs.cpu().numpy(), targets_flat.cpu().numpy(),
                                   input_sizes.cpu().numpy(), targets_sizes.cpu().numpy(), space_idx)
        ctx.grads = torch.FloatTensor(grads)  # save for backward not works!
        if inputs.is_cuda:
            return torch.FloatTensor(loss).cuda(inputs.get_device())
        return torch.FloatTensor(loss)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param grad_output: [batch_size]
        :return:
        """
        loss_grads = Variable(ctx.grads)
        if grad_output.is_cuda:
            loss_grads = loss_grads.cuda(grad_output.get_device())
        grad = loss_grads.contiguous() * grad_output.contiguous().view(1, -1, 1)
        return grad, None, None, None, None



if __name__ == "__main__":
    from torch.autograd import gradcheck

    # gradchek takes a tuple of tensor as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    # alphabet_size = 30
    # max_targets_len = 50
    # max_sequence_len = 100
    # batch_size = 2

    alphabet_size = 5
    max_targets_len = 100
    max_sequence_len = 200
    # max_targets_len = 5
    # max_sequence_len = 6
    batch_size = 1

    np.random.seed(523)

    targets_sizes = np.random.randint(1, max_targets_len + 1, batch_size)
    inputs_sizes = targets_sizes + np.random.randint(0, (max_sequence_len - max_targets_len) + 1, batch_size)
    inputs = np.random.randn(max_sequence_len, batch_size, alphabet_size)
    # expected shape seqLength x batchSize x alphabet_size

    sum_target_len = np.sum(targets_sizes)
    targets_flat = (np.random.rand(sum_target_len) * alphabet_size).astype(np.int64)

    input = (nn.LogSoftmax(dim=2)(Variable(torch.FloatTensor(inputs), requires_grad=True)),
             Variable(torch.LongTensor(targets_flat), requires_grad=False),
             Variable(torch.LongTensor(inputs_sizes), requires_grad=False),
             Variable(torch.LongTensor(targets_sizes), requires_grad=False))
    # ASGLoss(space_idx=1)(*input)
    print(CTCWithoutBlankLossFunction.apply(*input).data[0])
    test = gradcheck(CTCWithoutBlankLossFunction.apply, input)  # , atol=1e-5, rtol=1e-5)
    print(test)
