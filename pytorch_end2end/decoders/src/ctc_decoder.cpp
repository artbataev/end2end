#include <vector>
#include <torch/extension.h>

namespace py = pybind11;

std::vector<at::Tensor> decode_greedy(const at::Tensor& logits, const at::Tensor& logits_lengths, const int blank_idx) {
    // collapse repeated, remove blank
    auto argmax_logits = logits.argmax(-1);
    auto decoded_targets = at::zeros_like(argmax_logits);
    auto decoded_targets_lengths = at::zeros_like(logits_lengths);
    auto batch_size = logits_lengths.size(0);

    for (int i = 0; i < batch_size; i++) {
        auto prev_symbol = blank_idx;
        auto current_len = 0;
        for (int j = 0; j < logits_lengths[i].item<int>(); j++) {
            const auto current_symbol = argmax_logits[i][j].item<int>();
            if (current_symbol != blank_idx && prev_symbol != current_symbol) {
                decoded_targets[i][current_len] = current_symbol;
                current_len += 1;
            }
            prev_symbol = current_symbol;
        }
        decoded_targets_lengths[i] = current_len;
    }

    return {decoded_targets,
            decoded_targets_lengths};
}

PYBIND11_MODULE(cpp_ctc_decoder, m) {
    m.def("decode_greedy", &decode_greedy, "Decode greedy",
          py::arg("logits"), py::arg("logits_lengths"), py::arg("blank_idx"));
}
