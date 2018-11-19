#pragma once

#include <torch/extension.h>

class CTCLossWrapper {
public:
    explicit CTCLossWrapper(int blank_idx_);

    std::tuple<
            at::Tensor,
            at::Tensor
    > ctc_loss_forward(
            const at::Tensor& logits,
            const at::Tensor& targets,
            const at::Tensor& logits_lengths,
            const at::Tensor& targets_lengths);

private:
    void _ctc_loss_forward_2d(
            const torch::Tensor& logits,
            const torch::Tensor& targets,
            int sequence_length, int targets_len,
            int batch_i,
            torch::Tensor& losses,
            torch::Tensor& grads);

    int blank_idx;
};

PYBIND11_MODULE(cpp_ctc_loss, m) {
    namespace py = pybind11;
    using namespace pybind11::literals;
    py::class_<CTCLossWrapper>(m, "CTCLossWrapper").
            def(py::init<int>(), "blank_idx"_a).
            def("ctc_loss_forward", &CTCLossWrapper::ctc_loss_forward, "CTC loss forward and grads", "logits"_a,
                "targets"_a, "logits_lengths"_a, "targets_lengths"_a);
}
