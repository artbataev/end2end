"""
Tests are based on the torch7 bindings for warp-ctc. Reference numbers are also obtained from the tests.
file taken from https://github.com/SeanNaren/warp-ctc
originals (not pytorch) https://github.com/baidu-research/warp-ctc
test with
>> python -m tests.test_ctc
"""
import unittest

import torch
from torch.autograd import Variable

from pytorch_end2end.modules.ctc_loss import CTCLoss

places = 5
DELIMITER = "=" * 50 + "\n"


def run_grads(label_sizes, labels, probs, sizes, after_softmax=False, blank_idx=0, check_baidu_grads=False):
    ctc_loss = CTCLoss(reduce=True, after_softmax=after_softmax, blank_idx=blank_idx)
    probs = Variable(probs, requires_grad=True)
    cost = ctc_loss(probs, labels, sizes, label_sizes)
    cost.backward()
    cpu_cost = cost.data[0]
    if torch.cuda.is_available():
        probs = Variable(probs.data.cuda(), requires_grad=True)
        cost = ctc_loss(probs, labels, sizes, label_sizes)
        cost.backward()
        gpu_cost = cost.data[0]
    else:
        gpu_cost = cpu_cost
    grads = probs.grad

    if check_baidu_grads and blank_idx == 0:
        from pytorch_end2end.modules.warp_ctc_wrapper import WarpCTCLoss as CTCLoss_Baidu
        probs = Variable(probs.data, requires_grad=True)
        cost = CTCLoss_Baidu(reduce=True)(probs, labels, sizes, label_sizes)
        cost.backward()
        grads_baidu = probs.grad
        print(DELIMITER * 3 + "GRADS BAIDU\n {}".format(grads_baidu) + DELIMITER + "Your GRADS \n {}".format(
            grads) + DELIMITER * 3)

    print(grads.view(grads.size(0) * grads.size(1), grads.size(2)))
    return cpu_cost, gpu_cost


class TestCases(unittest.TestCase):
    def test_simple(self):
        probs = torch.FloatTensor([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]).contiguous()
        labels = Variable(torch.IntTensor([[1, 2]]))
        label_sizes = Variable(torch.IntTensor([2]))
        sizes = Variable(torch.IntTensor([2]))
        cpu_cost, gpu_cost = run_grads(label_sizes, labels, probs, sizes, check_baidu_grads=True)
        expected_cost = 2.4628584384918
        self.assertEqual(cpu_cost, gpu_cost)
        self.assertAlmostEqual(cpu_cost, expected_cost, places)

    def test_medium(self):
        probs = torch.FloatTensor([
            [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
        ]).transpose(0, 1).contiguous()

        labels = Variable(torch.IntTensor([[1, 2],
                                           [1, 2]]))
        label_sizes = Variable(torch.IntTensor([2, 2]))
        sizes = Variable(torch.IntTensor([2, 2]))
        cpu_cost, gpu_cost = run_grads(label_sizes, labels, probs, sizes, check_baidu_grads=True)
        expected_cost = 6.0165174007416
        self.assertEqual(cpu_cost, gpu_cost)
        self.assertAlmostEqual(cpu_cost, expected_cost, places)

    # def test_medium_stability(self):
    #     multiplier = 200
    #     probs = torch.FloatTensor([
    #         [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
    #         [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
    #     ]).transpose(0, 1).contiguous() * multiplier
    #
    #     labels = Variable(torch.IntTensor([[1, 2], [1, 2]]))
    #     label_sizes = Variable(torch.IntTensor([2, 2]))
    #     sizes = Variable(torch.IntTensor([2, 2]))
    #     cpu_cost, gpu_cost = run_grads(label_sizes, labels, probs, sizes, check_baidu_grads=True)
    #     expected_cost = 199.96618652344
    #     self.assertEqual(cpu_cost, gpu_cost)
    #     self.assertAlmostEqual(cpu_cost, expected_cost, places)

    def test_empty_label(self):
        probs = torch.FloatTensor([
            [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
        ]).transpose(0, 1).contiguous()

        labels = Variable(torch.IntTensor([[1, 2], [0, 0]]))
        label_sizes = Variable(torch.IntTensor([2, 0]))
        sizes = Variable(torch.IntTensor([2, 2]))
        cpu_cost, gpu_cost = run_grads(label_sizes, labels, probs, sizes)
        expected_cost = 6.416517496109
        self.assertEqual(cpu_cost, gpu_cost)
        self.assertAlmostEqual(cpu_cost, expected_cost, places)

    # # TODO: add tests from https://github.com/tensorflow/tensorflow/blob/4ac1956698e0e02f3051da65cdf453601555ed4a/tensorflow/python/kernel_tests/ctc_loss_op_test.py
    def test_tf_1(self):
        # blank in tensorflow - last! here - first!
        probs = torch.FloatTensor([
            [[0.0260553, 0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857],
             [0.010436, 0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609],
             [0.0037688, 0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882],
             [0.00331533, 0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545],
             [0.00623107, 0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441]]
        ]).contiguous()

        labels = Variable(torch.IntTensor([[1, 2, 3, 2, 1]]))
        label_sizes = Variable(torch.IntTensor([5]))
        sizes = Variable(torch.IntTensor([5]))
        cpu_cost, gpu_cost = run_grads(label_sizes, labels, probs, sizes, after_softmax=True)
        expected_cost = 3.34211
        self.assertAlmostEqual(cpu_cost, gpu_cost, 5)
        self.assertAlmostEqual(cpu_cost, expected_cost, places)

    def test_tf_2(self):
        # blank in tensorflow - last! here - first!

        probs = torch.FloatTensor([
            [[0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
             [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549],
             [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456],
             [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345],
             [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]]
        ]).contiguous()

        labels = Variable(torch.IntTensor([[0, 1, 1, 0]]))
        label_sizes = Variable(torch.IntTensor([4]))
        sizes = Variable(torch.IntTensor([5]))
        cpu_cost, gpu_cost = run_grads(label_sizes, labels, probs, sizes, after_softmax=True, blank_idx=5)
        expected_cost = 5.42262
        self.assertAlmostEqual(cpu_cost, gpu_cost, 5)
        self.assertAlmostEqual(cpu_cost, expected_cost, places)


if __name__ == '__main__':
    unittest.main()
