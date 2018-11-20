import os

import torch

import sys

module_base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

build_path = os.path.join(module_base, "cmake-build-debug")
build_path2 = os.path.join(module_base, "build")
if os.path.exists(build_path2):
    sys.path.append(build_path2)
else:
    sys.path.append(build_path)

import cpp_ctc_decoder


class CTCDecoderError(Exception):
    pass


class CTCBeamSearchDecoder:
    """
        Decoder class to perform CTC Beam Search

        :param beam_width: width of beam (number of stored hypotheses), default ``100``. \n
            If ``1``, decoder always perform greedy (argmax) decoding
        :param blank_idx: id of blank label, default ``0``
        :param time_major: if logits are time major (else batch major)
        :param labels: list of strings with labels (including blank symbol), e.g. ``["_", "a", "b", "c"]``
        :param lm_path: path to language model (ARPA format or gzipped ARPA)
        :param alpha: acoustic (original network) model weight, makes sense only if language model is present
        :param beta: language model weight
        :param case_sensitive: obtain language model scores with respect to case, default ``False``
        """
    def __init__(self, beam_width=100, blank_idx=0, time_major=True, labels=None,
                 lm_path=None, alpha=1.0, beta=1.0,
                 case_sensitive=False):
        self._beam_width = beam_width
        self._blank_idx = blank_idx
        self._labels = labels or []
        self._lm_path = os.path.abspath(lm_path) if lm_path else ""
        self._alpha = alpha
        self._beta = beta
        self._time_major = time_major
        self._case_sensitive = case_sensitive

        self._check_params()

        self._decoder = cpp_ctc_decoder.CTCDecoder(self._blank_idx, self._beam_width,
                                                   self._labels, self._lm_path, self._case_sensitive)

    def _check_params(self):
        # TODO: Check all params
        if self._lm_path:
            if self._labels is None:
                raise CTCDecoderError("To decode with language model you should pass labels")
            if not os.path.isfile(self._lm_path):
                raise CTCDecoderError("Can't find a model: {}".format(self._lm_path))

    def decode(self, logits, logits_lengths=None):
        """
        Perform decoding

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
        if self._beam_width == 1:
            return self.decode_greedy(logits, logits_lengths)
        raise NotImplementedError("Beam search is not implemented")

    def print_scores_for_sentence(self, words):
        self._decoder.print_scores_for_sentence(words)

    def decode_greedy(self, logits, logits_lengths=None):
        """
        Perform greedy (argmax) decoding

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
            logits=logits,
            logits_lengths=logits_lengths)

        return decoded_targets, decoded_targets_lengths, decoded_sentences
