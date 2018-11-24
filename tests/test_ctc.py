"""
Tests are based on the torch7 bindings for warp-ctc. Reference numbers are also obtained from the tests.
file taken from https://github.com/SeanNaren/warp-ctc
originals (not pytorch) https://github.com/baidu-research/warp-ctc
test with
>> python -m tests.test_ctc
"""
import unittest

import torch
import torch.nn.functional as F
from pytorch_end2end import CTCLoss
from torch.autograd.gradcheck import gradcheck
import numpy as np


class TestCTCLoss(unittest.TestCase):
    places = 5

    @staticmethod
    def run_grads(probs, targets, input_lengths, target_lengths,
                  after_logsoftmax=True, blank_idx=0,
                  print_compare_grads=False):
        """

        :param probs: batch_size * sequence_length * alphabet_dim
        :param targets: batch_size * sequence_length
        :param input_lengths:
        :param target_lengths:
        :param after_logsoftmax:
        :param blank_idx:
        :return:
        """
        probs = probs.permute(1, 0, 2)  # sequence_length * batch_size * alphabet_dim: make time major
        if not after_logsoftmax:
            log_probs = F.log_softmax(probs, dim=-1)
        else:
            log_probs = probs
        ctc_loss = CTCLoss(reduce=True, size_average=False, after_logsoftmax=True, time_major=True, blank_idx=blank_idx)
        log_probs = torch.Tensor(log_probs).requires_grad_()
        cost = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        cost.backward()
        cpu_cost = cost.item()
        if torch.cuda.is_available():
            log_probs = torch.Tensor(log_probs).cuda().requires_grad_()
            cost = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            cost.backward()
            gpu_cost = cost.item()
        else:
            gpu_cost = cpu_cost
        grads = log_probs.grad

        if print_compare_grads:
            from torch.nn import CTCLoss as CTCLossNative
            ctc_loss_native = CTCLossNative(blank=blank_idx, reduction="sum")
            log_probs_2 = torch.Tensor(log_probs.detach()).requires_grad_()
            cost_native = ctc_loss_native(log_probs_2, targets, input_lengths, target_lengths)
            cost_native.backward()
            grads_native = log_probs_2.grad
            print("=" * 70)
            print("GRADS Native\n {}".format(grads_native))
            print("=" * 30)
            print("Your GRADS \n {}".format(grads))
            print("=" * 70)
        return cpu_cost, gpu_cost

    # @unittest.skip("")
    def test_simple(self):
        probs = torch.FloatTensor([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]).contiguous()
        targets = torch.IntTensor([[1, 2]])
        target_lengths = torch.IntTensor([2])
        input_lengths = torch.IntTensor([2])
        cpu_cost, gpu_cost = self.run_grads(probs, targets, input_lengths, target_lengths, after_logsoftmax=False)
        expected_cost = 2.4628584384918
        self.assertAlmostEqual(cpu_cost, gpu_cost, self.places)
        self.assertAlmostEqual(cpu_cost, expected_cost, self.places)

    # @unittest.skip("")
    def test_medium(self):
        probs = torch.FloatTensor([
            [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
        ]).transpose(0, 1).contiguous()

        targets = torch.IntTensor([[1, 2],
                                   [1, 2]])
        target_lengths = torch.IntTensor([2, 2])
        input_lengths = torch.IntTensor([2, 2])
        cpu_cost, gpu_cost = self.run_grads(probs, targets, input_lengths, target_lengths, after_logsoftmax=False)
        expected_cost = 6.0165174007416
        self.assertAlmostEqual(cpu_cost, gpu_cost, self.places)
        self.assertAlmostEqual(cpu_cost, expected_cost, self.places)

    # def test_medium_stability(self):
    #     strange test from https://github.com/baidu-research/warp-ctc: maybe incorrect
    #     multiplier = 200
    #     probs = torch.FloatTensor([
    #         [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
    #         [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
    #     ]).transpose(0, 1).contiguous() * multiplier
    #
    #     targets = torch.IntTensor([[1, 2], [1, 2]])
    #     target_lengths = torch.IntTensor([2, 2])
    #     input_lengths = torch.IntTensor([2, 2])
    #     cpu_cost, gpu_cost = run_grads(probs, targets, input_lengths, target_lengths, after_logsoftmax=False)
    #     expected_cost = 199.96618652344
    #     self.assertAlmostEqual(cpu_cost, gpu_cost, self.places)
    #     self.assertAlmostEqual(cpu_cost, expected_cost, self.places)

    # @unittest.skip("")
    def test_empty_label(self):
        probs = torch.FloatTensor([
            [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
        ]).transpose(0, 1).contiguous()

        targets = torch.IntTensor([[1, 2], [0, 0]])
        target_lengths = torch.IntTensor([2, 0])
        input_lengths = torch.IntTensor([2, 2])
        cpu_cost, gpu_cost = self.run_grads(probs, targets, input_lengths, target_lengths, after_logsoftmax=False)
        expected_cost = 6.416517496109
        self.assertAlmostEqual(cpu_cost, gpu_cost, self.places)
        self.assertAlmostEqual(cpu_cost, expected_cost, self.places)

    # # TODO: add tests from https://github.com/tensorflow/tensorflow/blob/4ac1956698e0e02f3051da65cdf453601555ed4a/tensorflow/python/kernel_tests/ctc_loss_op_test.py
    # @unittest.skip("")
    def test_tf_1(self):
        # blank in tensorflow - last! here - first!
        probs = torch.FloatTensor([
            [[0.0260553, 0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857],
             [0.010436, 0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609],
             [0.0037688, 0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882],
             [0.00331533, 0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545],
             [0.00623107, 0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441]]
        ]).contiguous()

        targets = torch.IntTensor([[1, 2, 3, 2, 1]])
        target_lengths = torch.IntTensor([5])
        input_lengths = torch.IntTensor([5])
        cpu_cost, gpu_cost = self.run_grads(torch.log(probs), targets, input_lengths, target_lengths,
                                            after_logsoftmax=True, blank_idx=5)
        expected_cost = 3.34211
        self.assertAlmostEqual(cpu_cost, gpu_cost, self.places)
        self.assertAlmostEqual(cpu_cost, expected_cost, self.places)

    # @unittest.skip("")
    def test_tf_2(self):
        # blank in tensorflow - last! here - first!

        probs = torch.FloatTensor([
            [[0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
             [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549],
             [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456],
             [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345],
             [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]]
        ]).contiguous()

        targets = torch.IntTensor([[0, 1, 1, 0]])
        target_lengths = torch.IntTensor([4])
        input_lengths = torch.IntTensor([5])
        cpu_cost, gpu_cost = self.run_grads(torch.log(probs), targets, input_lengths, target_lengths,
                                            after_logsoftmax=True, blank_idx=5)
        expected_cost = 5.42262
        self.assertAlmostEqual(cpu_cost, gpu_cost, self.places)
        self.assertAlmostEqual(cpu_cost, expected_cost, self.places)

    # @unittest.skip("not implemented")
    def test_gradient(self):
        alphabet_size = 5
        max_targets_len = 10
        max_sequence_len = 20
        batch_size = 2

        np.random.seed(678)  # fix random seed
        targets_lengths = np.random.randint(low=1, high=max_targets_len + 1, size=batch_size)
        logits_lengths = targets_lengths + np.random.randint(
            low=0, high=(max_sequence_len - max_targets_len + 1), size=batch_size)
        logits = np.random.randn(batch_size, max_sequence_len, alphabet_size + 1)
        # 1 - blank, full_alphabet: alphabet_size + 1
        targets = (1 + np.random.rand(batch_size, np.max(targets_lengths)) * alphabet_size).astype(np.int64)

        targets_lengths = torch.LongTensor(targets_lengths)
        logits_lengths = torch.LongTensor(logits_lengths)
        targets = torch.LongTensor(targets)
        logits = torch.DoubleTensor(logits).requires_grad_()

        input_ = (logits, targets, logits_lengths, targets_lengths)

        test_result = gradcheck(CTCLoss(blank_idx=0, time_major=False, after_logsoftmax=False),
                                input_, eps=1e-6, atol=1e-4)
        self.assertTrue(test_result, "Gradient Invalid")


if __name__ == '__main__':
    unittest.main()
