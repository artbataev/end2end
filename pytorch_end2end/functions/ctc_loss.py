import os
import sys

from torch.autograd import Function

if "DEBUG" in os.environ:
    module_base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    build_path = os.path.join(module_base, "cmake-build-debug")
    sys.path.append(build_path)

import cpp_ctc_loss


class CTCLossFunction(Function):
    @staticmethod
    def forward(ctx, logits, targets, logits_lengths, targets_lengths, blank_idx=0):
        """
        Computes and returns CTC Loss, stores grads for backward computation

        :param ctx: storage for inner computations (to use in backward method)
        :param logits: Float or Double Tensor of shape [batch_size, sequence_length, alphabet_size]
        :param targets: Tensor with targets of shape [batch_size, targets_sequence_length]
        :param logits_lengths: Tensor of shape [batch_size] with lengths of sequences
        :param targets_lengths: Tensor of shape [batch_size] with lengths of target sequences
        :param blank_idx: id of blank label, default 0
        :return: tensor with loss of shape [batch_size]
        """
        ctc_engine = cpp_ctc_loss.CTCLossEngine(blank_idx)
        loss, grads = ctc_engine.compute(logits, targets, logits_lengths, targets_lengths)
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
