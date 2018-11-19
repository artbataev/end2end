import queue
import threading

import numba
import numpy as np
import torch
from torch.autograd import Function

from .utils import log_sum_exp
import os
import sys
module_base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
build_path = os.path.join(module_base, "cmake-build-debug")
build_path2 = os.path.join(module_base, "build")
if os.path.exists(build_path2):
    sys.path.append(build_path2)
else:
    sys.path.append(build_path)
import cpp_ctc_loss

@numba.jit(nogil=True)
def _ctc_loss_np(logits, targets, blank_idx=0):
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
    extended_targets_len = targets_len * 2 + 1
    extended_targets = np.ones(extended_targets_len, dtype=np.int64) * blank_idx
    extended_targets[1::2] = targets

    # alpha and beta computation
    # forward - alpha
    log_alpha = np.zeros((extended_targets_len, prediction_len))
    log_alpha[:] = -np.inf  # numba bugfix instead of log_alpha.fill(-np.inf)
    if prediction_len > 1 or extended_targets_len == 1:
        log_alpha[0, 0] = logits[0, extended_targets[0]]
    if extended_targets_len > 1:
        log_alpha[1, 0] = logits[0, extended_targets[1]]
    for t in range(1, prediction_len):  # timesteps
        start = max(0, extended_targets_len - 2 * (prediction_len - t))
        end = min(t * 2 + 2, extended_targets_len)
        log_alpha[start:end, t] = log_alpha[start:end, t - 1]
        for j in range(start, end):
            current_label = extended_targets[j]
            if j > 0:
                log_alpha[j, t] = log_sum_exp(log_alpha[j, t], log_alpha[j - 1, t - 1])
                if current_label != blank_idx and j - 2 >= 0 and extended_targets[j - 2] != current_label:
                    log_alpha[j, t] = log_sum_exp(log_alpha[j, t], log_alpha[j - 2, t - 1])
            log_alpha[j, t] += logits[t, current_label]
    if extended_targets_len > 1:
        loss_forward = log_sum_exp(log_alpha[extended_targets_len - 1, prediction_len - 1],
                                   log_alpha[extended_targets_len - 2, prediction_len - 1])
    else:
        loss_forward = log_alpha[extended_targets_len - 1, prediction_len - 1]

    # backward - beta
    log_beta = np.zeros((extended_targets_len, prediction_len))
    log_beta[:] = -np.inf  # numba bugfix instead of log_beta.fill(-np.inf)
    if prediction_len > 1 or extended_targets_len == 1:
        log_beta[extended_targets_len - 1, prediction_len - 1] = 0
    if extended_targets_len > 1:
        log_beta[extended_targets_len - 2, prediction_len - 1] = 0
    for t in range(prediction_len - 2, -1, -1):  # timesteps
        start = max(0, extended_targets_len - 2 * (prediction_len - t))
        end = min(t * 2 + 2, extended_targets_len)
        for j in range(start, end):
            current_label = extended_targets[j]
            log_beta[j, t] = log_beta[j, t + 1] + logits[t + 1, extended_targets[j]]
            if j < extended_targets_len - 1:
                log_beta[j, t] = log_sum_exp(log_beta[j, t],
                                             log_beta[j + 1, t + 1] + logits[t + 1, extended_targets[j + 1]])
                if current_label != blank_idx and j + 2 < extended_targets_len and extended_targets[
                    j + 2] != current_label:
                    log_beta[j, t] = log_sum_exp(log_beta[j, t], log_beta[j + 2, t + 1] + logits[
                        t + 1, extended_targets[j + 2]])

    alpha_beta = log_alpha + log_beta

    prob_sum = np.zeros((prediction_len, num_labels))
    prob_sum[:] = -np.inf
    for i in range(extended_targets_len):
        current_label = extended_targets[i]
        prob_sum[:, current_label] = log_sum_exp(prob_sum[:, current_label], alpha_beta[i, :])
    negative_term = prob_sum - loss_forward
    grad = np.exp(logits) - np.exp(negative_term)

    return -loss_forward, grad


def _ctc_3d_loss_np(logits, targets, logits_lengths, targets_length, blank_idx=0):
    batch_size = len(targets_length)
    grads = np.zeros_like(logits)

    losses = np.zeros(batch_size)

    # parallel computation, threading - because gil is released with numba.jit(nogil=True)
    # equivalent iterative computation is:
    # for i in range(batch_size):
    #     loss, grad = _ctc_loss_np(logits[:logits_lengths[i], i, :], targets[i, :targets_length[i]], blank_idx)
    #     grads[:logits_lengths[i], i, :] = grad
    #     losses[i] = loss
    que = queue.Queue()
    threads = []
    for i in range(batch_size):
        t = threading.Thread(target=lambda q, i, *args: q.put((i, _ctc_loss_np(*args))),
                             args=(que, i, logits[i, :logits_lengths[i], :],
                                   targets[i, :targets_length[i]], blank_idx))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    while not que.empty():
        i, (loss, grad) = que.get()
        grads[i, :logits_lengths[i], :] = grad
        losses[i] = loss

    return losses, grads


class CTCLossFunction(Function):
    @staticmethod
    def forward(ctx, logits, targets, logits_lengths, targets_lengths, blank_idx=0):
        """
        Computes and returns CTC Loss, stores grads for backward computation

        :param ctx: storage for inner computations (to use in backward method)
        :param logits: Float or Double Tensor of shape [batch_size, sequence_length, alphabet_size]
        :param targets: Tensor with targets of shape [batch_size, targets_sequence_length]
        :param logits_lengths: Tensor of shape [batch_size] with lenghts of sequences
        :param targets_lengths: Tensor of shape [batch_size] with lengths of target sequences
        :param blank_idx: id of blank label, default 0
        :return: tensor with loss of shape [batch_size]
        """
        tensor_type = logits.dtype
        ctc_engine = cpp_ctc_loss.CTCLossWrapper(blank_idx)
        loss, grads = ctc_engine.ctc_loss_forward(logits, targets, logits_lengths, targets_lengths)
        # loss, grads = _ctc_3d_loss_np(logits.detach().cpu().numpy(), targets.cpu().numpy(),
        #                            logits_lengths.cpu().numpy(), targets_lengths.cpu().numpy(), blank_idx)
        # loss = torch.tensor(loss, dtype=tensor_type)
        # grads = torch.tensor(grads, dtype=tensor_type)  # save for backward not works!
        # if logits.is_cuda:
        #     loss = loss.cuda(logits.get_device())
        ctx.grads = grads
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes backward for CTC Loss

        :param grad_output: [batch_size]
        :return: gradient for logits, None for other inputs (targets, logits_lengths, etc.: see forward method)
        """
        loss_grads = ctx.grads
        loss_grads.requires_grad_()
        if grad_output.is_cuda:
            loss_grads = loss_grads.cuda(grad_output.get_device())
        grad = loss_grads.contiguous() * grad_output.contiguous().view(-1, 1, 1)
        return grad, None, None, None, None
