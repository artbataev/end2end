import os

class CTCDecoderError(Exception):
    pass

class CTCBeamSearchDecoder:
    def __init__(self, beam_width=100, blank_idx=0, labels=None, lm_path=None, alpha=0.0, beta=0.0):
        self._beam_width = beam_width
        self._blank_idx = blank_idx
        self._labels = labels
        self._lm_path = os.path.abspath(lm_path)
        self._alpha = alpha
        self._beta = beta

        self._check_params()

    def _check_params(self):
        # TODO: Check all params
        if self._lm_path is not None:
            if self._labels is None:
                raise CTCDecoderError("To decode with language model you should pass labels")
            if not os.path.isfile(self._lm_path):
                raise CTCDecoderError("Can't find a model: {}".format(self._lm_path))

    def decode(self, logits, logits_lengths=None):
        raise NotImplementedError("Decoder is not yet implemented")

