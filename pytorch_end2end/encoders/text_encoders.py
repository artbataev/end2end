# -*- coding: utf-8 -*-
import itertools
import string
import torch
import numpy as np

# replace("\u0130", "i").replace("\u0049", "\u0131") for Turkish

class CTCEncoder:
    def __init__(self, allowed_chars=string.ascii_lowercase + " " + "'", to_lower=str.casefold):
        self.allowed_chars = allowed_chars
        self.num_symbols = len(allowed_chars) + 1
        self._char2num = {c: i + 1 for i, c in enumerate(self.allowed_chars)}
        self._num2char = dict(zip(self._char2num.values(), self._char2num.keys()))
        self._num2char[0] = ""  # "silence" character
        self._to_lower = to_lower
        self._blank_idx = 0

    def encode(self, text):
        if self._to_lower is not None:
            text = self._to_lower(text)
        text = " ".join(text.split())  # removing tabs, additional spaces, etc
        return [self._char2num[c] for c in text if c in self._char2num]

    def clean(self, text):
        if self._to_lower is not None:
            text = self._to_lower(text)
        text = " ".join(text.split())  # removing tabs, additional spaces, etc
        return "".join(c for c in text if c in self._char2num)

    def decode_argmax(self, logits, sequence_lengths=None):
        """

        :param logits: batch_size * sequence_length * num_labels
        :param sequence_lengths:
        :return:
        """
        labels = torch.max(logits, dim=2)[1]
        labels = labels.cpu()
        batch_size = labels.size()[0]
        max_sequence_length = labels.size()[1] if sequence_lengths is None else torch.max(sequence_lengths)
        decoded_sequences = np.zeros(labels.size(), dtype=np.int64)
        decoded_lengths = np.zeros(batch_size, dtype=np.int64)
        for i in range(batch_size):
            cur_sequence_length = sequence_lengths[i] if sequence_lengths is not None else max_sequence_length
            # assert cur_sequence_length > 0
            t = 0
            while labels[i, t] == self._blank_idx and t < cur_sequence_length:
                t += 1
            while t < cur_sequence_length:
                cur_label = labels[i, t]
                if cur_label != self._blank_idx and (t == 0 or cur_label != labels[i, t-1]):
                    decoded_sequences[i, decoded_lengths[i]] = cur_label
                    decoded_lengths[i] += 1

        results_str = []
        for i in range(batch_size):
            results_str.append("".join(self._num2char[idx] for idx in decoded_sequences[i, :decoded_lengths[i]]))

        max_length = decoded_lengths.max()
        return torch.LongTensor(decoded_sequences[:, :max_length]), torch.LongTensor(decoded_lengths), results_str

    def decode_list(self, text_int):
        return "".join(self._num2char[num] for num, _ in itertools.groupby(text_int))  # collapse repeated

    def decode_list_full(self, text_int):
        return "".join(self._num2char[num] for num in text_int)


class CollapseCTCEncoder(CTCEncoder):
    """Collapse all repeated characters"""

    def encode(self, text):
        if self._to_lower is not None:
            text = self._to_lower(text)
        text = " ".join(text.split())  # removing tabs, additional spaces, etc
        return [self._char2num[c] for c, _ in itertools.groupby(text) if c in self._char2num]


class ASGEncoder:
    """
    http://arxiv.org/abs/1609.03193
    Encoder for Auto Segmentation Criterion (and CTC without blank)
    """

    def __init__(self, allowed_chars=" " + string.ascii_lowercase + "'", to_lower=str.casefold):
        self.allowed_chars = allowed_chars
        self.num_symbols = len(allowed_chars) + 1  # "2" and ? "3" characters
        self._char2num = {c: i for i, c in enumerate(self.allowed_chars)}
        self._num2char = dict(zip(self._char2num.values(), self._char2num.keys()))
        self._num2char[len(allowed_chars)] = "2"
        # self._num2char[len(allowed_chars) + 1] = "3"
        self._to_lower = to_lower
        self._space_idx = self._char2num[" "]


    def encode(self, text):
        if self._to_lower is not None:
            text = self._to_lower(text)
        text = " ".join(text.split())  # removing tabs, additional spaces, etc
        encoded = [self._char2num[c] for c in text if c in self._char2num]
        for i, c in enumerate(encoded):
            if i > 0 and encoded[i - 1] == c:
                encoded[i] = self.num_symbols - 1  # last = "2"
        return encoded

    def clean(self, text):
        if self._to_lower is not None:
            text = self._to_lower(text)
        text = " ".join(text.split())  # removing tabs, additional spaces, etc
        return "".join(c for c in text if c in self._char2num)

    def greedy_decode_int(self, labels):
        result = [label for label, _ in itertools.groupby(labels)]
        start = 0
        end = len(result)
        if len(result) > 1 and result[0] == 0:
            start += 1
        if len(result[start:]) > 1 and result[-1] == 0:
            end -= 1
        return result[start:end]

    def decode_list(self, text_int):
        return "".join(self._num2char[num] for num, _ in itertools.groupby(text_int)).strip()  # collapse repeated, remove first/last space

    def decode_full(self, text_int):
        return "".join(self._num2char[num] for num in text_int)
