import unittest

import torch

from pytorch_end2end import CTCBeamSearchDecoder


class TestCTCDecoder(unittest.TestCase):
    def test_greedy_simple(self):
        logits = torch.FloatTensor([[[1, 2, 4, 3, 10],
                                     [2, 1, 3, 8, 1]]]).contiguous()
        logits_lengths = torch.LongTensor([2])
        blank_idx = 0
        labels = ["_", "a", "b", "c", "d"]
        correct_targets = torch.LongTensor([[4, 3]])
        correct_targets_lengths = torch.LongTensor([2])
        correct_sentences = ["dc", ]
        decoder = CTCBeamSearchDecoder(beam_width=1, blank_idx=blank_idx, labels=labels, time_major=False)
        decoded_targets, decoded_targets_lengths, decoded_sentences = decoder.decode(logits,
                                                                                     logits_lengths=logits_lengths)

        self.assertListEqual(decoded_targets_lengths.numpy().tolist(), correct_targets_lengths.numpy().tolist())
        self.assertListEqual(decoded_targets.numpy().tolist(), correct_targets.numpy().tolist())
        self.assertListEqual(decoded_sentences, correct_sentences)


if __name__ == "__main__":
    unittest.main()
