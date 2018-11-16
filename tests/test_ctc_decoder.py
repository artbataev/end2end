import unittest

import torch
import os
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

    def test_simple_lm(self):
        blank_idx = 0
        labels = ["_", "a", "b", "c", "d"]
        decoder = CTCBeamSearchDecoder(
            beam_width=1, blank_idx=blank_idx,
            labels=labels, time_major=False,
            lm_path=os.path.join(os.path.dirname(__file__), "librispeech_data", "librispeech_3-gram_pruned.3e-7.arpa.gz"))
        decoder.print_scores_for_sentence(["mother", "washed", "the", "frame"])
        print("=" * 50)
        decoder.print_scores_for_sentence(["mother".upper(), "washed".upper(), "the".upper(), "frame".upper()])
        print("=" * 50)
        print("=" * 50)
        decoder = CTCBeamSearchDecoder(
            beam_width=1, blank_idx=blank_idx,
            labels=labels, time_major=False,
            lm_path=os.path.join(os.path.dirname(__file__), "librispeech_data", "librispeech_3-gram_pruned.3e-7.arpa.gz"),
            case_sensitive=True)
        decoder.print_scores_for_sentence(["mother", "washed", "the", "frame"])
        print("=" * 50)
        decoder.print_scores_for_sentence(["mother".upper(), "washed".upper(), "the".upper(), "frame".upper()])


if __name__ == "__main__":
    unittest.main()
