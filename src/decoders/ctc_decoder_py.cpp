// Copyright 2019 Vladimir Bataev

#include "decoders/ctc_decoder.h"

PYBIND11_MODULE(cpp_ctc_decoder, m) {
  namespace py = pybind11;
  using pybind11::literals::operator""_a;
  py::class_<CTCDecoder>(m, "CTCDecoder")
      .def(py::init<int,
                    int,
                    std::vector<std::string>,
                    std::string,
                    double,
                    double,
                    double,
                    bool>(),
           "blank_idx"_a,
           "beam_width_"_a = 100,
           "labels"_a = std::vector<std::string>{},
           "lm_path"_a = "",
           "lmwt_"_a = 1.0,
           "wip_"_a = 0.0,
           "oov_penalty_"_a = -1000.0,
           "case_sensitive"_a = false)
      .def("decode_greedy",
           &CTCDecoder::decode_greedy,
           "Decode greedy",
           "logits_"_a,
           "logits_lengths_"_a)
      .def("decode",
           &CTCDecoder::decode,
           "Decode greedy",
           "logits_"_a,
           "logits_lengths_"_a)
      .def("print_scores_for_sentence",
           &CTCDecoder::print_scores_for_sentence,
           "Print scores",
           "words"_a);
}
