#pragma once

#include <torch/extension.h>

std::tuple <at::Tensor,
at::Tensor> ctc_loss_forward(const at::Tensor& logits, const at::Tensor& logits_lengths);

PYBIND11_MODULE(cpp_ctc_loss, m) {
    namespace py = pybind11;
    using namespace pybind11::literals;
    m.def("ctc_loss_forward", &ctc_loss_forward, "CTC Loss forward pass and gradient computation");
}
