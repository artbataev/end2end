import os

import torch


class CTCDecoderError(Exception):
    pass


class CTCBeamSearchDecoder:
    def __init__(self, beam_width=100, blank_idx=0, time_major=True, labels=None, lm_path=None, alpha=0.0, beta=0.0):
        """

        :param beam_width:
        :param blank_idx:
        :param time_major:
        :param labels:
        :param lm_path:
        :param alpha:
        :param beta:
        """
        self._beam_width = beam_width
        self._blank_idx = blank_idx
        self._labels = labels
        self._lm_path = os.path.abspath(lm_path) if lm_path is not None else None
        self._alpha = alpha
        self._beta = beta
        self._time_major = time_major

        self._check_params()

    def _check_params(self):
        # TODO: Check all params
        if self._lm_path is not None:
            if self._labels is None:
                raise CTCDecoderError("To decode with language model you should pass labels")
            if not os.path.isfile(self._lm_path):
                raise CTCDecoderError("Can't find a model: {}".format(self._lm_path))

    def decode(self, logits, logits_lengths=None):
        if self._beam_width == 1:
            return self.decode_greedy(logits, logits_lengths)
        raise NotImplementedError("Beam search is not implemented")

    def decode_greedy(self, logits, logits_lengths=None):
        """

        :param logits:
        :param logits_lengths:
        :return: decoded_targets
                 decoded_targets_lengths
                 decoded_sentences
        """
        if self._time_major:
            logits = logits.transpose(1, 0)  # batch_size * sequence_length * alphabet_size
        logits = logits.detach().cpu()
        batch_size = logits.size()[0]
        max_sequence_length = logits.size()[1]
        if logits_lengths is None:
            logits_lengths = torch.zeros(batch_size, dtype=torch.int).fill_(max_sequence_length)
        else:
            logits_lengths = logits_lengths.cpu()

        argmax_logits = torch.argmax(logits, dim=-1)
        decoded_targets = torch.zeros_like(argmax_logits)
        decoded_targets_lengths = torch.zeros_like(logits_lengths)
        decoded_sentences = []

        for i in range(batch_size):
            prev_symbol = self._blank_idx
            current_len = 0
            decoded_sentences.append("")
            for j in range(logits_lengths[i]):
                current_symbol = argmax_logits[i, j]
                if current_symbol != self._blank_idx:
                    if prev_symbol != current_symbol:
                        decoded_targets[i, current_len] = current_symbol
                        current_len += 1
                        if self._labels is not None:
                            decoded_sentences[i] += self._labels[current_symbol]
                prev_symbol = current_symbol
            decoded_targets_lengths[i] = current_len

        return decoded_targets, decoded_targets_lengths, decoded_sentences
