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

    @unittest.skip("")
    def test_simple_lm(self):
        blank_idx = 0
        labels = ["_", "a", "b", "c", "d"]
        decoder = CTCBeamSearchDecoder(
            beam_width=1, blank_idx=blank_idx,
            labels=labels, time_major=False,
            lm_path=os.path.join(os.path.dirname(__file__), "librispeech_data",
                                 "librispeech_3-gram_pruned.3e-7.arpa.gz"))
        decoder.print_scores_for_sentence(["mother", "washed", "the", "frame"])
        print("=" * 50)
        decoder.print_scores_for_sentence(["mother".upper(), "washed".upper(), "the".upper(), "frame".upper()])
        print("=" * 50)
        print("=" * 50)
        decoder = CTCBeamSearchDecoder(
            beam_width=1, blank_idx=blank_idx,
            labels=labels, time_major=False,
            lm_path=os.path.join(os.path.dirname(__file__), "librispeech_data",
                                 "librispeech_3-gram_pruned.3e-7.arpa.gz"),
            case_sensitive=True)
        decoder.print_scores_for_sentence(["mother", "washed", "the", "frame"])
        print("=" * 50)
        decoder.print_scores_for_sentence(["mother".upper(), "washed".upper(), "the".upper(), "frame".upper()])

    def test_with_probs(self):
        """
        This test is from https://github.com/PaddlePaddle/DeepSpeech/blob/develop/decoders/tests/test_decoders.py
        :return:
        """
        labels = ["\'", ' ', 'a', 'b', 'c', 'd', '_']
        decoder = CTCBeamSearchDecoder(beam_width=50, blank_idx=6, time_major=False, labels=labels)
        probs_seq1 = [
            [0.06390443, 0.21124858, 0.27323887, 0.06870235, 0.0361254, 0.18184413, 0.16493624],
            [0.03309247, 0.22866108, 0.24390638, 0.09699597, 0.31895462, 0.0094893, 0.06890021],
            [0.218104, 0.19992557, 0.18245131, 0.08503348, 0.14903535, 0.08424043, 0.08120984],
            [0.12094152, 0.19162472, 0.01473646, 0.28045061, 0.24246305, 0.05206269, 0.09772094],
            [0.1333387, 0.00550838, 0.00301669, 0.21745861, 0.20803985, 0.41317442, 0.01946335],
            [0.16468227, 0.1980699, 0.1906545, 0.18963251, 0.19860937, 0.04377724, 0.01457421]
        ]
        probs_seq2 = [
            [0.08034842, 0.22671944, 0.05799633, 0.36814645, 0.11307441, 0.04468023, 0.10903471],
            [0.09742457, 0.12959763, 0.09435383, 0.21889204, 0.15113123, 0.10219457, 0.20640612],
            [0.45033529, 0.09091417, 0.15333208, 0.07939558, 0.08649316, 0.12298585, 0.01654384],
            [0.02512238, 0.22079203, 0.19664364, 0.11906379, 0.07816055, 0.22538587, 0.13483174],
            [0.17928453, 0.06065261, 0.41153005, 0.1172041, 0.11880313, 0.07113197, 0.04139363],
            [0.15882358, 0.1235788, 0.23376776, 0.20510435, 0.00279306, 0.05294827, 0.22298418]
        ]
        log_probs = torch.log(torch.FloatTensor([probs_seq1, probs_seq2]))
        expected_greedy_result = ["ac'bdc", "b'da"]
        expected_beam_search_result = ['acdc', "b'a"]
        _, _, greedy_result = decoder.decode_greedy(log_probs)
        self.assertListEqual(greedy_result, expected_greedy_result)

        _, _, beam_search_result = decoder.decode(log_probs)
        self.assertListEqual(beam_search_result, expected_beam_search_result)
        # print(beam_search_result)

        # from pytorch_end2end import CTCLoss
        # criterion = CTCLoss(blank_idx=6, time_major=False, after_logsoftmax=True)
        # loss = criterion(log_probs, torch.LongTensor([[2, 4, 5, 4], [3, 0, 2, 0]]), torch.LongTensor([5, 5]), torch.LongTensor([4, 3]))
        # print(loss)
        # loss = criterion(log_probs, torch.LongTensor([[2, 4, 5, 4], [3, 0, 5, 2]]), torch.LongTensor([5, 5]), torch.LongTensor([4, 4]))
        # print(loss)



if __name__ == "__main__":
    unittest.main()
