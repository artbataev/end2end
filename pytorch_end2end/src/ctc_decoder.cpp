#include <vector>
#include <string>
#include <iostream>
#include <torch/extension.h>
#include "lm/model.hh"


namespace py = pybind11;


class CTCDecoder {
    using lm_state = lm::ngram::State;
public:
    CTCDecoder(int blank_idx_,
               std::vector<std::string> labels_ = {},
               const std::string& lm_path = "") :
            blank_idx(blank_idx_),
            labels(std::move(labels_)) {
        if (!lm_path.empty()) {
            std::unique_ptr<lm::ngram::Model> lm_model_(new lm::ngram::Model(lm_path.c_str()));
            lm_model = std::move(lm_model_);
        } else
            lm_model = nullptr;
    }

    void print_scores_for_sentence(std::vector<std::string> words) {
        if (lm_model == nullptr)
            return;
        lm_state state(lm_model->BeginSentenceState()), out_state;
        auto& vocab = lm_model->GetVocabulary();
        for (const auto& word: words) {
            std::cout << lm_model->Score(state, vocab.Index(word), out_state) << '\n';
            state = out_state;
        }
    }

    std::tuple<
            at::Tensor,
            at::Tensor,
            std::vector<std::string>
    > decode_greedy(
            const at::Tensor& logits,
            const at::Tensor& logits_lengths) {
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

private:
    int blank_idx;
    std::vector<std::string> labels;
    std::unique_ptr<lm::ngram::Model> lm_model;
};


PYBIND11_MODULE(cpp_ctc_decoder, m) {
    using namespace pybind11::literals;
    py::class_<CTCDecoder>(m, "CTCDecoder").
            def(py::init<int, std::vector<std::string>, std::string>(),
                "blank_idx"_a,
                "labels"_a = std::vector<std::string>{},
                "lm_path"_a = "").
            def("decode_greedy", &CTCDecoder::decode_greedy, "Decode greedy", "logits"_a, "logits_lengths"_a).
            def("print_scores_for_sentence", &CTCDecoder::print_scores_for_sentence, "Print scores", "words"_a);
}
