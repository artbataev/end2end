// Copyright 2019 Vladimir Bataev

#pragma once

#include <torch/extension.h>

class CTCLossEngine {
 public:
  explicit CTCLossEngine(int blank_idx_);

  std::tuple<at::Tensor, at::Tensor> compute(const at::Tensor& logits,
                                             const at::Tensor& targets,
                                             const at::Tensor& logits_lengths,
                                             const at::Tensor& targets_lengths);

 private:
  void compute_2d(const torch::Tensor& logits_2d,
                  const torch::TensorAccessor<int64_t, 1>& targets_1d_a,
                  int seq_len, int targets_len, int batch_i,
                  torch::Tensor& losses, torch::Tensor& grads);

  int blank_idx;
};

PYBIND11_MODULE(cpp_ctc_loss, m) {
  namespace py = pybind11;
  using namespace pybind11::literals;
  py::class_<CTCLossEngine>(m, "CTCLossEngine")
      .def(py::init<int>(), "blank_idx"_a)
      .def("compute", &CTCLossEngine::compute, "CTC loss forward and grads",
           "logits"_a, "targets"_a, "logits_lengths"_a, "targets_lengths"_a);
}
