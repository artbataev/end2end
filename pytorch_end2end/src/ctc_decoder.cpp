#include <vector>
#include <string>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <torch/extension.h>
#include "lm/model.hh"
#include "lm/enumerate_vocab.hh"
#include "math_utils.h"
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
    for (int i = 0; i < batch_size; i++) {
        std::tie(decoded_indices_vec[i], current_sequence_len, decoded_sentences[i]) = decode_sentence(
                logits[i], logits_lengths[i].item<int>());
        decoded_targets_lengths[i] = current_sequence_len;
    }

    auto max_sequence_len = decoded_targets_lengths.max().item<int64_t>();
    auto decoded_indices = at::zeros({batch_size, max_sequence_len}, logits_lengths.options());
    for (int i = 0; i < batch_size; i++) {
        for (int l = 0; l < decoded_targets_lengths[i].item<int>(); l++) {
            decoded_indices[i][l] = decoded_indices_vec[i][l];
        }
    }
    return {decoded_indices, decoded_targets_lengths, decoded_sentences};
}


std::string CTCDecoder::indices2str(const std::vector<int>& char_ids) {
    std::string result;
    result.reserve(char_ids.size());
    if (!labels.empty()) {
        for (const auto& char_id: char_ids)
            result += labels[char_id];
    }
    return result;
}

std::string CTCDecoder::indices2str(const at::Tensor& char_ids, int len) {
    std::string result;
    result.reserve(static_cast<size_t>(len));
    if (!labels.empty()) {
        for (int i = 0; i < len; i++)
            result += labels[char_ids[i].item<int>()];
    }
    return result;
}

using prefix_key_t = std::vector<int>;

class Prefix {
public:
    Prefix() : prob_last_b{-INFINITY}, prob_last_nb{-INFINITY} {};

//    Prefix(double prob_last_b_, double prob_last_nb_) : prob_last_b{prob_last_b_}, prob_last_nb{prob_last_nb_} {}

    double prob_last_b;
    double prob_last_nb;

    double get_full_prob() {
        return log_sum_exp(prob_last_nb, prob_last_b);
    }
};


std::tuple<std::vector<int>, int, std::string>
CTCDecoder::decode_sentence(const at::Tensor& logits_2d, int sequence_len) {
    // Prefix beam search: https://arxiv.org/pdf/1408.2873.pdf
    auto logits_a = logits_2d.accessor<float, 2>(); // TODO: convert to double?
    auto alphabet_size = static_cast<int>(logits_2d.size(1));

    std::map<prefix_key_t, Prefix> prev_prefixes;
    prev_prefixes[{blank_idx}] = {0, -INFINITY};
    std::map<prefix_key_t, Prefix> next_prefixes;

    for (int l = 0; l < sequence_len; l++) {
        // generate new hypotheses
        for (const auto& prev_prefix: prev_prefixes) {
            auto& prefix_key = prev_prefix.first;
            auto& prefix_weights = prev_prefix.second;
            auto last_char_id = prefix_key[prefix_key.size() - 1];
            auto cur_log_prob_blank = logits_a[l][blank_idx];
            for (int char_id = 0; char_id < alphabet_size; char_id++) {
                auto cur_log_prob = logits_a[l][char_id];
                if (char_id == blank_idx) {
                    next_prefixes[prefix_key].prob_last_b = log_sum_exp(
                            next_prefixes[prefix_key].prob_last_b,
                            cur_log_prob + log_sum_exp(prefix_weights.prob_last_b, prefix_weights.prob_last_nb));
                } else {
                    auto new_prefix_key{prefix_key};
                    new_prefix_key.emplace_back(char_id);
                    if (char_id == last_char_id) {
                        next_prefixes[prefix_key].prob_last_nb = log_sum_exp(
                                next_prefixes[prefix_key].prob_last_nb, cur_log_prob + prefix_weights.prob_last_b);
                        next_prefixes[new_prefix_key].prob_last_nb = log_sum_exp(
                                next_prefixes[new_prefix_key].prob_last_nb, cur_log_prob + prefix_weights.prob_last_b);
                    } else {
                        next_prefixes[new_prefix_key].prob_last_nb = log_sum_exp(
                                next_prefixes[new_prefix_key].prob_last_nb,
                                cur_log_prob + log_sum_exp(prefix_weights.prob_last_b, prefix_weights.prob_last_nb));
                    }
                    if (next_prefixes.at(new_prefix_key).prob_last_b == -INFINITY &&
                        prev_prefixes.count(new_prefix_key) > 0)
                        next_prefixes.at(new_prefix_key).prob_last_b =
                                cur_log_prob_blank + prev_prefixes.at(new_prefix_key).get_full_prob();

                    if (next_prefixes.at(new_prefix_key).prob_last_nb == -INFINITY &&
                        prev_prefixes.count(new_prefix_key) > 0)
                        next_prefixes.at(new_prefix_key).prob_last_nb =
                                cur_log_prob + prev_prefixes.at(new_prefix_key).prob_last_nb;
                }
            }
        }
        std::vector<prefix_key_t> next_prefixes_keys;
        next_prefixes_keys.reserve(next_prefixes.size());
        for (const auto& prefix: next_prefixes)
            next_prefixes_keys.emplace_back(prefix.first);
        if (next_prefixes.size() > beam_width) {
            std::sort(next_prefixes_keys.begin(), next_prefixes_keys.end(),
                      [&next_prefixes](const prefix_key_t& lhs, const prefix_key_t& rhs) {
                          return next_prefixes.at(lhs).get_full_prob() > next_prefixes.at(rhs).get_full_prob();
                      });
            prev_prefixes.clear();
            for (int i = 0; i < beam_width; i++)
                prev_prefixes[next_prefixes_keys[i]] = next_prefixes[next_prefixes_keys[i]];
        } else {
            std::swap(prev_prefixes, next_prefixes);
        }
        next_prefixes.clear();
    }

    std::vector<prefix_key_t> prev_prefixes_keys;
    prev_prefixes_keys.reserve(prev_prefixes_keys.size());
    for (const auto& prefix: prev_prefixes)
        prev_prefixes_keys.emplace_back(prefix.first);
    std::sort(prev_prefixes_keys.begin(), prev_prefixes_keys.end(),
              [&prev_prefixes](const prefix_key_t& lhs, const prefix_key_t& rhs) {
                  return prev_prefixes.at(lhs).get_full_prob() > prev_prefixes.at(rhs).get_full_prob();
              });

    std::vector<int> result_sequence{prev_prefixes_keys[0].begin() + 1, prev_prefixes_keys[0].end()};
    std::string result_sequence_str = indices2str(result_sequence);
//    for(int i = 0; i < std::min(10, static_cast<int>(prev_prefixes_keys.size())); i++) {
//        const auto& prefix = prev_prefixes_keys[i];
//        std::cout << indices2str({prefix.begin() + 1, prefix.end()}) << " " << prev_prefixes[prefix].get_full_prob() << "\n";
//    }

    return {result_sequence, static_cast<int>(result_sequence.size()), result_sequence_str};
}

std::tuple<
        at::Tensor,
        at::Tensor,
        std::vector<std::string>
> CTCDecoder::decode_greedy(
        const at::Tensor& logits,
        const at::Tensor& logits_lengths) {
    // collapse repeated, remove blank
    auto batch_size = logits_lengths.size(0);
    auto argmax_logits = logits.argmax(-1);
    auto decoded_targets = at::zeros_like(argmax_logits);
    auto decoded_targets_lengths = at::zeros_like(logits_lengths);
    std::vector<std::string> decoded_sentences(static_cast<size_t>(batch_size), "");

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
        decoded_sentences[i] = indices2str(decoded_targets[i], current_len);
        decoded_targets_lengths[i] = current_len;
    }

    return {decoded_targets,
            decoded_targets_lengths,
            decoded_sentences};
}
