# -*- coding: utf-8 -*-
import itertools
import string


class CTCEncoder:
    def __init__(self, allowed_chars=string.ascii_lowercase + " " + "'", to_lower=True):
        self.allowed_chars = allowed_chars
        self.num_symbols = len(allowed_chars) + 1
        self._char2num = {c: i + 1 for i, c in enumerate(self.allowed_chars)}
        self._num2char = dict(zip(self._char2num.values(), self._char2num.keys()))
        self._num2char[0] = ""  # "silence" character
        self._to_lower = to_lower

    def encode(self, text):
        if self._to_lower:
            text = text.lower()
        return [self._char2num[c] for c in text if c in self._char2num]

    def clean(self, text):
        if self._to_lower:
            text = text.lower()
        return "".join(c for c in text if c in self._char2num)

    def greedy_decode_int(self, labels):
        result = [label for label, _ in itertools.groupby(labels) if label != 0]
        return result

    def decode(self, text_int):
        return "".join(self._num2char[num] for num, _ in itertools.groupby(text_int))  # collapse repeated

    def decode_full(self, text_int):
        return "".join(self._num2char[num] for num in text_int)

    def decode_full_show_silence(self, text_int):
        return "".join(self._num2char[num] if (num != self.num_symbols - 1) else "_" for num in text_int)

    def decode_texts_sparse(self, indices, values, shape):
        decoded = ["" for _ in range(shape[0])]
        for index, value in zip(indices.tolist(), values.tolist()):
            decoded[index[0]] += self._num2char[value]
        return decoded


class CollapseCTCEncoder(CTCEncoder):
    """Collapse all repeated characters"""

    def encode(self, text):
        if self._to_lower:
            text = text.lower()
        return [self._char2num[c] for c, _ in itertools.groupby(text) if c in self._char2num]


class ASGEncoder:
    """
    http://arxiv.org/abs/1609.03193
    Encoder for Auto Segmentation Criterion (and CTC without blank)
    """

    def __init__(self, allowed_chars=" " + string.ascii_lowercase + "'", to_lower=True):
        self.allowed_chars = allowed_chars
        self.num_symbols = len(allowed_chars) + 1  # "2" and ? "3" characters
        self._char2num = {c: i for i, c in enumerate(self.allowed_chars)}
        self._num2char = dict(zip(self._char2num.values(), self._char2num.keys()))
        self._num2char[len(allowed_chars)] = "2"
        # self._num2char[len(allowed_chars) + 1] = "3"
        self._to_lower = to_lower

    def encode(self, text):
        if self._to_lower:
            text = text.strip().casefold()
            text = " ".join(text.split())  # removing tags, additional spaces, etc
        encoded = [self._char2num[c] for c in text if c in self._char2num]
        for i, c in enumerate(encoded):
            if i > 0 and encoded[i - 1] == c:
                encoded[i] = self.num_symbols - 1  # last = "2"
        return encoded

    def clean(self, text):
        if self._to_lower:
            text = text.lower()
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

    def decode(self, text_int):
        return "".join(self._num2char[num] for num, _ in itertools.groupby(text_int)).strip()  # collapse repeated, remove first/last space

    def decode_full(self, text_int):
        return "".join(self._num2char[num] for num in text_int)

    def decode_texts_sparse(self, indices, values, shape):
        decoded = ["" for _ in range(shape[0])]
        for index, value in zip(indices.tolist(), values.tolist()):
            decoded[index[0]] += self._num2char[value]
        return decoded
