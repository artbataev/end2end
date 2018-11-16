#include <vector>
#include <string>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <torch/extension.h>
#include "lm/model.hh"
#include "lm/enumerate_vocab.hh"


namespace py = pybind11;
using word2index_t = std::unordered_map<std::string, lm::WordIndex>;

class CustomEnumerateVocab : public lm::EnumerateVocab {
public:
    CustomEnumerateVocab() = default;

    void Add(lm::WordIndex index, const StringPiece& str) override {
        word2index[{str.data(), str.length()}] = index;
    };

    word2index_t get_word2index() {
        return word2index;
    }

private:
    word2index_t word2index;
};

std::string str_to_lower(const std::string& str) {
    auto result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

class CTCDecoder {
    using lm_state = lm::ngram::State;
public:
    CTCDecoder(int blank_idx_,
               std::vector<std::string> labels_ = {},
               const std::string& lm_path = "", bool case_sensitive_ = false) :
            blank_idx(blank_idx_),
            labels(std::move(labels_)), case_sensitive(case_sensitive_) {
        if (!lm_path.empty()) {
            CustomEnumerateVocab enumerate_vocab;
            lm::ngram::Config config;
            config.enumerate_vocab = &enumerate_vocab;
            std::unique_ptr<lm::ngram::ProbingModel> lm_model_(
                    dynamic_cast<lm::ngram::ProbingModel *>(
                            lm::ngram::LoadVirtual(lm_path.c_str(), config, lm::ngram::PROBING)));
            lm_model = std::move(lm_model_);
            if (case_sensitive)
                word2index = enumerate_vocab.get_word2index();
            else
                for (const auto& elem: enumerate_vocab.get_word2index())
                    word2index[str_to_lower(elem.first)] = elem.second;
        } else
            lm_model = nullptr;
    }

    lm::WordIndex get_idx(const std::string& word) {
        auto word_to_find = case_sensitive ? word : str_to_lower(word);
        if (word2index.count(word_to_find) > 0)
            return word2index.at(word_to_find);
        return lm_model->GetVocabulary().NotFound();
    }

    void print_scores_for_sentence(std::vector<std::string> words) {
        if (lm_model == nullptr)
            return;
        lm_state state(lm_model->BeginSentenceState()), out_state;
        for (const auto& word: words) {
            std::cout << word << " " << get_idx(word) << " " << lm_model->GetVocabulary().Index(word) << " "
                      << lm_model->Score(state, get_idx(word), out_state) << "\n";
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
    bool case_sensitive;
    std::vector<std::string> labels;
    std::unique_ptr<lm::ngram::ProbingModel> lm_model;
    word2index_t word2index;
};


PYBIND11_MODULE(cpp_ctc_decoder, m) {
    using namespace pybind11::literals;
    py::class_<CTCDecoder>(m, "CTCDecoder").
            def(py::init<int, std::vector<std::string>, std::string, bool>(),
                "blank_idx"_a,
                "labels"_a = std::vector<std::string>{},
                "lm_path"_a = "",
                "case_sensitive"_a = false).
            def("decode_greedy", &CTCDecoder::decode_greedy, "Decode greedy", "logits"_a, "logits_lengths"_a).
            def("print_scores_for_sentence", &CTCDecoder::print_scores_for_sentence, "Print scores", "words"_a);
}
