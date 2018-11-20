#include <vector>
#include <string>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <torch/extension.h>
#include "lm/model.hh"
#include "lm/enumerate_vocab.hh"
#include "ctc_decoder.h"


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

CTCDecoder::CTCDecoder(int blank_idx_, int beam_width_ = 100,
                       std::vector<std::string> labels_ = {},
                       const std::string& lm_path = "", bool case_sensitive_ = false) :
        blank_idx(blank_idx_),
        beam_width{beam_width_},
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

lm::WordIndex CTCDecoder::get_idx(const std::string& word) {
    if (case_sensitive)
        return lm_model->GetVocabulary().Index(word);
    auto word_to_find = str_to_lower(word);
    if (word2index.count(word_to_find) > 0)
        return word2index.at(word_to_find);
    return lm_model->GetVocabulary().NotFound();
}

void CTCDecoder::print_scores_for_sentence(std::vector<std::string> words) {
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
> CTCDecoder::decode(const at::Tensor& logits,
         const at::Tensor& logits_lengths) {
    auto decoded_targets_lengths = at::zeros_like(logits_lengths);
    auto batch_size = logits_lengths.size(0);

    std::vector<std::string> decoded_sentences{static_cast<size_t>(batch_size), ""};
    std::vector<std::vector<int>> decoded_indices_vec{static_cast<size_t>(batch_size), std::vector<int>{}};

    int current_sequence_len = 0;
    for(int i = 0; i < batch_size; i++) {
        std::tie(decoded_indices_vec[i], current_sequence_len, decoded_sentences[i]) = decode_sentence(
                logits[i], 0); // logits_lengths[i].item<int>()
        decoded_targets_lengths[i] = current_sequence_len;
    }

    auto max_sequence_len = 1; // decoded_targets_lengths.max().item<long long>();
    auto decoded_indices = at::zeros({batch_size, max_sequence_len}, logits_lengths.options());
    for(int i = 0; i < batch_size; i++) {
        for (int l = 0; l < 1 /*decoded_targets_lengths[i].item<int>()*/; l++) {
            decoded_indices[i][l] = decoded_indices_vec[i][l];
        }
    }
    return {decoded_indices, decoded_targets_lengths, decoded_sentences};
}

struct Hypothesis {
    // path to prev
    // last
};

std::tuple<std::vector<int>, int, std::string> CTCDecoder::decode_sentence(const at::Tensor& logits_2d, int sequence_len) {
    auto logits_a = logits_2d.accessor<float, 2>();
//    auto alphabet_size = static_cast<int>(logits_2d.size(1));
    std::vector<Hypothesis> hyps;
    std::vector<Hypothesis> new_hyps;
    hyps.reserve(static_cast<int>(beam_width));
    for(int l = 0; l < sequence_len; l++) {
        std::swap(hyps, new_hyps);
        new_hyps.clear();


    }
    return {{0}, 0, "_"};
}

std::tuple<
        at::Tensor,
        at::Tensor,
        std::vector<std::string>
> CTCDecoder::decode_greedy(
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
