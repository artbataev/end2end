import os
import sys
from collections import namedtuple

import torch

if "DEBUG_E2E" in os.environ:
    module_base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    build_path = os.path.join(module_base, os.getenv("DEBUG_E2E"))
    sys.path.append(build_path)

import cpp_ctc_decoder


class CTCDecoderError(Exception):
    pass


DecoderResults = namedtuple("DecoderResults", ["decoded_targets",
                                               "decoded_targets_lengths",
                                               "decoded_sentences"])


class CTCDecoder:
    """
    Decoder class to perform CTC decoding

    :param beam_width: width of beam (number of stored hypotheses), default ``100``. \n
        If ``1``, decoder always perform greedy (argmax) decoding
    :param blank_idx: id of blank label, default ``0``
    :param time_major: if logits are time major (else batch major)
    :param labels: list of strings with labels (including blank symbol), e.g. ``["_", "a", "b", "c"]``
    :param lm_path: path to language model (ARPA format or gzipped ARPA)
    :param lmwt: language model weight, default ``1.0``, makes sense only if language model is present
    :param wip: word insertion penalty, default ``1.0``, makes sense only if labels are present
    :param oov_penalty: penalty for each oov word, default ``-10.0``
    :param case_sensitive: obtain language model scores with respect to case, default ``False``
    """

    def __init__(self, beam_width=100, blank_idx=0, time_major=False, labels=None,
                 lm_path=None, lmwt=1.0, wip=1.0,
                 oov_penalty=-10,
                 case_sensitive=True):
        self._beam_width = beam_width
        self._blank_idx = blank_idx
        self._labels = labels or []
        self._lm_path = os.path.abspath(lm_path) if lm_path else ""
        self._lmwt = lmwt
        self._wip = wip
        self._oov_penalty = oov_penalty
        self._time_major = time_major
        self._case_sensitive = case_sensitive

        self._check_params()

        self._decoder = cpp_ctc_decoder.CTCDecoder(self._blank_idx, self._beam_width,
                                                   self._labels,
                                                   self._lm_path, self._lmwt, self._wip, self._oov_penalty,
                                                   self._case_sensitive)

    def _check_params(self):
        # TODO: Check all params
        if self._lm_path:
            if self._labels is None:
                raise CTCDecoderError("To decode with language model you should pass labels")
            if not os.path.isfile(self._lm_path):
                raise CTCDecoderError("Can't find a model: {}".format(self._lm_path))

    def decode(self, logits, logits_lengths=None):
        """
        Performs prefix beam search decoding as described in `<https://arxiv.org/abs/1408.2873>`_

        :param logits: tensor with neural network outputs after logsoftmax \n
            of shape ``(sequence_length, batch_size, alphabet_size)`` if ``time_major`` \n
            else of shape ``(batch_size, sequence_length, alphabet_size)``
        :param logits_lengths: default ``None``
        :return: ``namedtuple(decoded_targets, decoded_targets_lengths, decoded_sentences)`` \n
            decoded_targets:
                tensor with result targets of shape ``(batch_size, sequence_length)``,
                doesn't contain blank symbols \n
            decoded_targets_length:
                tensor with lengths of decoded targets \n
            decoded_sentences:
                list of strings, shape ``(batch_size)``.
                If ``labels are None``, list of empty string is returned. \n
        """
        if self._beam_width == 1:
            return self.decode_greedy(logits, logits_lengths)

        if self._time_major:
            logits = logits.transpose(1, 0)  # batch_size * sequence_length * alphabet_size
        logits = logits.detach().cpu()
        batch_size = logits.size()[0]
        max_sequence_length = logits.size()[1]
        if logits_lengths is None:
            logits_lengths = torch.zeros(batch_size, dtype=torch.int).fill_(max_sequence_length)
        else:
            logits_lengths = logits_lengths.cpu()

        decoded_targets, decoded_targets_lengths, decoded_sentences = self._decoder.decode(
            logits_=logits,
            logits_lengths_=logits_lengths)

        return DecoderResults(decoded_targets, decoded_targets_lengths, decoded_sentences)

    def _print_scores_for_sentence(self, words):
        self._decoder.print_scores_for_sentence(words)

    def decode_greedy(self, logits, logits_lengths=None):
        """
        Performs greedy (argmax) decoding

        :param logits: tensor with neural network outputs after logsoftmax \n
            of shape ``(sequence_length, batch_size, alphabet_size)`` if ``time_major`` \n
            else of shape ``(batch_size, sequence_length, alphabet_size)``
        :param logits_lengths: default ``None``
        :return: ``(decoded_targets, decoded_targets_lengths, decoded_sentences)`` \n
            decoded_targets:
                tensor with result targets of shape ``(batch_size, sequence_length)``,
                doesn't contain blank symbols \n
            decoded_targets_length:
                tensor with lengths of decoded targets \n
            decoded_sentences:
                list of strings, shape ``(batch_size)``.
                If ``labels are None``, list of empty string is returned. \n
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

        decoded_targets, decoded_targets_lengths, decoded_sentences = self._decoder.decode_greedy(
            logits_=logits,
            logits_lengths_=logits_lengths)

        return DecoderResults(decoded_targets, decoded_targets_lengths, decoded_sentences)
