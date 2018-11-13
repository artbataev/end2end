#include <vector>
#include <string>
#include <torch/extension.h>

namespace py = pybind11;
using namespace pybind11::literals;

std::tuple<at::Tensor, at::Tensor, std::vector<std::string>> decode_greedy(
        const at::Tensor& logits,
        const at::Tensor& logits_lengths,
        const int blank_idx,
        const std::vector<std::string>& labels = {}) {
    // collapse repeated, remove blank
    auto argmax_logits = logits.argmax(-1);
    auto decoded_targets = at::zeros_like(argmax_logits);
    auto decoded_targets_lengths = at::zeros_like(logits_lengths);
    auto batch_size = logits_lengths.size(0);

    std::vector<std::string> decoded_sentences{};
    decoded_sentences.reserve(static_cast<size_t>(batch_size));

    for (int i = 0; i < batch_size; i++) {
        auto prev_symbol = blank_idx;
        auto current_len = 0;
        decoded_sentences.emplace_back("");
        for (int j = 0; j < logits_lengths[i].item<int>(); j++) {
            const auto current_symbol = argmax_logits[i][j].item<int>();
            if (current_symbol != blank_idx && prev_symbol != current_symbol) {
                decoded_targets[i][current_len] = current_symbol;
                if (!labels.empty())
                    decoded_sentences[i] += labels[current_symbol];
                current_len += 1;
            }
            prev_symbol = current_symbol;
        }
        decoded_targets_lengths[i] = current_len;
    }

    return {decoded_targets,
            decoded_targets_lengths,
            decoded_sentences};
}

PYBIND11_MODULE(cpp_ctc_decoder, m) {
    m.def("decode_greedy", &decode_greedy, "Decode greedy",
          "logits"_a, "logits_lengths"_a, "blank_idx"_a, "labels"_a=std::vector<std::string>{});
}
