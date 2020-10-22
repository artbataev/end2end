// Copyright 2019 Vladimir Bataev

#include "losses/gram_ctc_loss.h"

PYBIND11_MODULE(cpp_gram_ctc_loss, m) {
  namespace py = pybind11;
  using pybind11::literals::operator""_a;
  py::class_<GramCTCLossEngine>(m, "GramCTCLossEngine")
      .def(py::init<int, int, int, std::unordered_map<int, std::vector<int>>>(),
          "blank_idx"_a,
          "num_base_labels"_a,
          "total_labels"_a,
          "label2ids"_a)
      .def("compute",
           &GramCTCLossEngine::compute,
           "CTC loss forward and grads",
           "logits"_a,
           "targets"_a,
           "logits_lengths"_a,
           "targets_lengths"_a);
}
