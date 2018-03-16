# -*- coding: utf-8 -*-
import itertools
import string

import numpy as np
import torch


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

    def decode_argmax(self, logits, sequence_lengths=None, convert_to_str=True):
        """

        :param logits: batch_size * sequence_length * num_labels
        :param sequence_lengths:
        :return:
        """
        labels = torch.max(logits, dim=2)[1]
        labels = labels.cpu()
        batch_size = labels.size()[0]
        max_sequence_length = labels.size()[1] if sequence_lengths is None else torch.max(sequence_lengths)
        decoded_sequences = np.zeros((batch_size, max_sequence_length), dtype=np.int64)
        decoded_lengths = np.zeros(batch_size, dtype=np.int64)
        for i in range(batch_size):
            cur_sequence_length = sequence_lengths[i] if sequence_lengths is not None else max_sequence_length
            # assert cur_sequence_length > 0
            t = 0
            while t < cur_sequence_length and labels[i, t] == self._blank_idx:
                t += 1
            while t < cur_sequence_length:
                cur_label = labels[i, t]
                if cur_label != self._blank_idx and (t == 0 or cur_label != labels[i, t - 1]):
                    decoded_sequences[i, decoded_lengths[i]] = cur_label
                    decoded_lengths[i] += 1
                t += 1

        results_str = []
        if convert_to_str:
            for i in range(batch_size):
                results_str.append("".join(self._num2char[idx] for idx in decoded_sequences[i, :decoded_lengths[i]]))

        max_length = decoded_lengths.max() or 1
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
        self.num_symbols = len(allowed_chars)
        self._char2num = {c: i for i, c in enumerate(self.allowed_chars)}
        self._num2char = dict(zip(self._char2num.values(), self._char2num.keys()))
        self._num2char[self.num_symbols] = "2"
        self._double_idx = self.num_symbols
        self._num2char[self.num_symbols + 1] = "3"
        self._triple_idx = self.num_symbols + 1
        self.num_symbols += 2
        self._to_lower = to_lower
        self._space_idx = self._char2num[" "]

    def encode(self, text):
        if self._to_lower is not None:
            text = self._to_lower(text)
        text = " ".join(text.split())  # removing tabs, additional spaces, etc
        encoded = []
        for c, group in itertools.groupby(text):
            encoded.append(self._char2num[c])
            num = len(list(group))
            if num == 2:
                encoded.append(self._double_idx)
            elif num > 2:
                encoded.append(self._triple_idx)

        return encoded

    def clean(self, text):
        if self._to_lower is not None:
            text = self._to_lower(text)
        text = " ".join(text.split())  # removing tabs, additional spaces, etc
        return "".join(c for c in text if c in self._char2num)

    def decode_argmax(self, logits, sequence_lengths=None, convert_to_str=True):
        """

        :param logits: batch_size * sequence_length * num_labels
        :param sequence_lengths:
        :return:
        """
        labels = torch.max(logits, dim=2)[1]
        labels = labels.cpu()
        batch_size = labels.size()[0]
        max_sequence_length = labels.size()[1] if sequence_lengths is None else torch.max(sequence_lengths)
        decoded_sequences = np.zeros((batch_size, max_sequence_length), dtype=np.int64)
        decoded_lengths = np.zeros(batch_size, dtype=np.int64)
        for i in range(batch_size):
            cur_sequence_length = sequence_lengths[i] if sequence_lengths is not None else max_sequence_length
            # assert cur_sequence_length > 0
            t = 0
            while t < cur_sequence_length and labels[i, t] == self._space_idx:
                t += 1
            while t < cur_sequence_length:
                cur_label = labels[i, t]
                if t == 0 or cur_label != labels[i, t - 1]:
                    if cur_label == self._double_idx:
                        if decoded_lengths[i] > 0:
                            decoded_sequences[i, decoded_lengths[i]] = decoded_sequences[i, decoded_lengths[i] - 1]
                            decoded_lengths[i] += 1
                    elif cur_label == self._triple_idx:
                        if decoded_lengths[i] > 0:
                            decoded_sequences[i, decoded_lengths[i]] = decoded_sequences[i, decoded_lengths[i] - 1]
                            decoded_sequences[i, decoded_lengths[i] + 1] = decoded_sequences[i, decoded_lengths[i] - 1]
                            decoded_lengths[i] += 2
                    else:
                        decoded_sequences[i, decoded_lengths[i]] = cur_label
                        decoded_lengths[i] += 1
                t += 1
            if decoded_lengths[i] == 0:
                decoded_lengths[i] = 1
                decoded_sequences[i, 0] = self._space_idx

        results_str = []
        if convert_to_str:
            for i in range(batch_size):
                results_str.append(self.decode_list_full(decoded_sequences[i, :decoded_lengths[i]]).strip())

        max_length = decoded_lengths.max() or 1
        return torch.LongTensor(decoded_sequences[:, :max_length]), torch.LongTensor(decoded_lengths), results_str

    def decode_list(self, text_int):
        return "".join(self._num2char[num] for num, _ in
                       itertools.groupby(text_int)).strip()  # collapse repeated, remove first/last space

    def decode_list_full(self, text_int):
        text_int_transformed = []
        for num in text_int:
            if num == self._double_idx:
                if len(text_int_transformed) > 0:
                    text_int_transformed.append(text_int_transformed[-1])
            elif num == self._triple_idx:
                if len(text_int_transformed) > 0:
                    text_int_transformed.append(text_int_transformed[-1])
                    text_int_transformed.append(text_int_transformed[-1])
            else:
                text_int_transformed.append(num)
        return "".join(self._num2char[num] for num in text_int_transformed)
