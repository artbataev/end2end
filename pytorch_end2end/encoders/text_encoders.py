# -*- coding: utf-8 -*-
import itertools
import string

import numpy as np


class CTCEncoder:
    """
    Simple CTC-encoder for text

    """
    def __init__(self, characters, blank_id=0, transform_fn=str.upper):
        self.blank_id = blank_id
        self.transform_fn = transform_fn
        self.char2id = dict()
        idx = 0
        for c in characters:
            if idx == blank_id:
                idx += 1
            self.char2id[c] = idx
            idx += 1
        self.id2char = dict(zip(self.char2id.values(), self.char2id.keys()))
        self.id2char[self.blank_id] = ""
        self.num_symbols = len(self.id2char)

    def clean(self, text):
        clean_text = "".join(c for c in self.transform_fn(text) if c in self.char2id)
        return clean_text

    def encode(self, text):
        encoded_text = np.array([self.char2id[c] for c in self.transform_fn(text) if c in self.char2id])
        return encoded_text

    def decode(self, ids_list):
        text = "".join(self.id2char[idx] for idx, _ in itertools.groupby(ids_list) if idx != self.blank_id)
        return text

    def decode_pure(self, ids_list):
        text = "".join(self.id2char[idx] for idx in ids_list)
        return text


class ASGEncoder:
    """
    http://arxiv.org/abs/1609.03193
    Encoder for Auto Segmentation Criterion (and CTC without blank)
    """

    def __init__(self, allowed_chars=" " + string.ascii_lowercase + "'", to_lower=str.casefold):
        raise NotImplementedError
