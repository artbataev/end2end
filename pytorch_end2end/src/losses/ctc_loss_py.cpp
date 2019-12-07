// Copyright 2019 Vladimir Bataev

#include "losses/ctc_loss.h"

PYBIND11_MODULE(cpp_ctc_loss, m) {
  namespace py = pybind11;
  using pybind11::literals::operator""_a;
  py::class_<CTCLossEngine>(m, "CTCLossEngine")
      .def(py::init<int>(), "blank_idx"_a)
      .def("compute", &CTCLossEngine::compute, "CTC loss forward and grads",
           "logits"_a, "targets"_a, "logits_lengths"_a, "targets_lengths"_a);
}
