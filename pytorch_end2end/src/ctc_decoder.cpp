#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <iostream>
#include <memory>

// pytorch
#include <torch/extension.h>

// kenlm
#include "lm/model.hh"
#include "lm/enumerate_vocab.hh"

#include "math_utils.h"
#include "ctc_decoder.h"
#include "threadpool.h"


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
                       const std::string& lm_path = "", bool case_sensitive_ = false, double lmwt_ = 1.0) :
        blank_idx(blank_idx_),
        beam_width{beam_width_},
        space_idx{-1},
        lmwt{lmwt_},
        oov_score{-10000}, // TODO: tune
        labels(std::move(labels_)), case_sensitive(case_sensitive_) {
    if (!labels.empty()) {
        space_idx = static_cast<int>(std::distance(labels.begin(), std::find(labels.begin(), labels.end(), " ")));
        if (space_idx >= labels.size())
            space_idx = -1;
    }
    if (!lm_path.empty()) {
        auto enumerate_vocab = new CustomEnumerateVocab{};
        lm::ngram::Config config;
        config.enumerate_vocab = enumerate_vocab;
        std::unique_ptr<lm::ngram::ProbingModel> lm_model_(
                dynamic_cast<lm::ngram::ProbingModel *>(
                        lm::ngram::LoadVirtual(lm_path.c_str(), config, lm::ngram::PROBING)));
        lm_model = std::move(lm_model_);
        if (case_sensitive)
            word2index = enumerate_vocab->get_word2index();
        else
            for (const auto& elem: enumerate_vocab->get_word2index())
                word2index[str_to_lower(elem.first)] = elem.second;
        delete enumerate_vocab;
    } else {
        lm_model = nullptr;
        lmwt = 0;
    }
}

lm::WordIndex CTCDecoder::get_idx(const std::string& word) {
    if (case_sensitive)
        return lm_model->GetVocabulary().Index(word);
    auto word_to_find = str_to_lower(word);
    if (word2index.count(word_to_find) > 0)
        return word2index.at(word_to_find);
    return lm_model->GetVocabulary().NotFound();
}

double CTCDecoder::get_score_for_sentence(std::vector<std::string> words) {
    if (lm_model == nullptr)
        return oov_score;
    double result = 1.0;
    lm_state_t state(lm_model->BeginSentenceState()), out_state;
    for (const auto& word: words) {
        auto word_idx = get_idx(word);
        if (word_idx == 0)
            return oov_score;
        result = lm_model->Score(state, word_idx, out_state);
        state = out_state;
    }
    return result;
}

double CTCDecoder::get_score_for_sentence(const std::vector<int>& sentence) {
    std::vector<std::string> words;
    std::string word;
    for (const auto& c_id: sentence) {
        if (c_id == blank_idx) // first
            continue;
        if (c_id != space_idx)
            word += labels[c_id];
        else if (!word.empty()) {
            words.emplace_back(word);
            word = "";
        }
    }
    if (!word.empty())
        words.emplace_back(word);
    if (!words.empty())
        return get_score_for_sentence(words);
    return oov_score;
}

bool CTCDecoder::is_empty_sentence(const std::vector<int>& sentence) {
    for (const auto& c: sentence)
        if (c != blank_idx && c != space_idx)
            return false;
    return true;
}

void CTCDecoder::print_scores_for_sentence(std::vector<std::string> words) {
    if (lm_model == nullptr)
        return;
    lm_state_t state(lm_model->BeginSentenceState()), out_state;
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

    {
        ThreadPool pool{static_cast<size_t>(batch_size)};
        for (int i = 0; i < batch_size; i++) {
            pool.add_task([this, &logits, &logits_lengths, i,
                                  &decoded_sentences, &decoded_indices_vec, &decoded_targets_lengths] {
                int current_sequence_len = 0;
                std::tie(decoded_indices_vec[i], current_sequence_len, decoded_sentences[i]) = decode_sentence(
                        logits[i], logits_lengths[i].item<int>());
                decoded_targets_lengths[i] = current_sequence_len;
            });
        }
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

class Prefix : public std::enable_shared_from_this<Prefix> {
public:
    Prefix() : cur_prob_blank{-INFINITY},
               cur_prob_not_blank{-INFINITY},
               prev_prob_blank{-INFINITY},
               prev_prob_not_blank{-INFINITY},
               last_char{0},
               next_data{},
               parent{nullptr},
               lm_weight{-20000} {}

    double cur_prob_blank;
    double cur_prob_not_blank;
    double prev_prob_blank;
    double prev_prob_not_blank;
    int last_char;
    double lm_weight;
    std::shared_ptr<Prefix> parent;
    std::map<int, std::weak_ptr<Prefix>> next_data;

    std::vector<int> get_sentence() {
        std::vector<int> result;
        auto cur_char = last_char;
        auto cur_parent = parent;
        while (last_char >= 0 && cur_parent != nullptr) {
            result.emplace_back(cur_char);
            cur_char = cur_parent->last_char;
            cur_parent = cur_parent->parent;
        }
        return {result.rbegin(), result.rend()};
    }

    std::pair<std::shared_ptr<Prefix>, bool> get_next(int char_id) {
        if (next_data.count(char_id) > 0)
            if (auto next_prefix = next_data.at(char_id).lock())
                return {next_prefix, false};

        auto new_prefix = std::make_shared<Prefix>();
        next_data[char_id] = new_prefix;
        new_prefix->last_char = char_id;
        new_prefix->lm_weight = lm_weight;
        new_prefix->parent = shared_from_this();
        return {new_prefix, true};
    }

    double get_prev_full_prob() const {
        return log_sum_exp(prev_prob_not_blank, prev_prob_blank);
    }

    double get_prev_full_prob_with_lmwt() const {
        return get_prev_full_prob() + lm_weight;
    }
};


std::tuple<std::vector<int>, int, std::string>
CTCDecoder::decode_sentence(const at::Tensor& logits_2d, int sequence_len) {
    // Prefix beam search: https://arxiv.org/pdf/1408.2873.pdf
    auto logits_a = logits_2d.accessor<float, 2>(); // TODO: convert to double?
    auto alphabet_size = static_cast<int>(logits_2d.size(1));

    // NB: prob - log-probabilities
    std::vector<std::shared_ptr<Prefix>> prefixes;
    auto init_prefix = std::make_shared<Prefix>();
    init_prefix->prev_prob_blank = 0.0;
    prefixes.emplace_back(init_prefix);

    // at every timestep
    for (int l = 0; l < sequence_len; l++) {
        auto cur_log_prob_blank = logits_a[l][blank_idx];

        std::vector<std::shared_ptr<Prefix>> new_prefixes;
        // for every character
        for (int char_id = 0; char_id < alphabet_size; char_id++) {
            auto cur_prob = logits_a[l][char_id];
            // for every prefix
            for (auto& prefix: prefixes) {
                if (char_id == blank_idx) {
                    prefix->cur_prob_blank = log_sum_exp(prefix->cur_prob_blank,
                                                         cur_prob + prefix->get_prev_full_prob());
                } else {
                    auto new_prefix_res = prefix->get_next(char_id);
                    auto new_prefix = new_prefix_res.first;
                    auto is_new = new_prefix_res.second;
                    if (is_new)
                        new_prefixes.emplace_back(new_prefix);

                    if (char_id == prefix->last_char) { // repeated character
                        new_prefix->cur_prob_not_blank = log_sum_exp(
                                new_prefix->cur_prob_not_blank,
                                cur_prob + prefix->prev_prob_blank);
                        prefix->cur_prob_not_blank = log_sum_exp(
                                prefix->cur_prob_not_blank,
                                cur_prob + prefix->prev_prob_not_blank);
                    } else {
                        new_prefix->cur_prob_not_blank = log_sum_exp(
                                new_prefix->cur_prob_not_blank,
                                cur_prob + prefix->get_prev_full_prob());

                        // lm weight
                        auto new_prefix_sentence = new_prefix->get_sentence();
                        // TODO: score only if char_id == space_idx
                        if (lm_model != nullptr && !is_empty_sentence(new_prefix_sentence)) {
                            new_prefix->lm_weight = get_score_for_sentence(new_prefix_sentence) * lmwt;
                        }
                    }
                }
            } // end for: prefixes
        } // end for: characters
//        std::cout << "step: " << l << std::endl;
        prefixes.reserve(prefixes.size() + new_prefixes.size());
        prefixes.insert(prefixes.end(), new_prefixes.begin(), new_prefixes.end());
        for (auto& prefix: prefixes) {
            prefix->prev_prob_blank = prefix->cur_prob_blank;
            prefix->prev_prob_not_blank = prefix->cur_prob_not_blank;
            prefix->cur_prob_blank = -INFINITY;
            prefix->cur_prob_not_blank = -INFINITY;

//            std::cout << "\"" << indices2str(prefix->get_sentence()) << "\": " << std::exp(prefix->get_prev_full_prob())
//                      << ", " << prefix->get_prev_full_prob()
//                      << " | " << prefix->lm_weight << " " << prefix->prev_prob_blank << " "
//                      << prefix->prev_prob_not_blank << "\n";
        }

        std::sort(prefixes.begin(), prefixes.end(),
                  [](const std::shared_ptr<Prefix>& lhs, const std::shared_ptr<Prefix>& rhs) {
                      return lhs->get_prev_full_prob_with_lmwt() > rhs->get_prev_full_prob_with_lmwt();
                  });


        if (prefixes.size() > beam_width)
            prefixes.resize(static_cast<size_t>(beam_width));
    } // end for: timestep

//    std::cout << "=======================" << std::endl;
//    std::cout << "=======================" << std::endl;
    std::vector<int> result_sequence{prefixes[0]->get_sentence()};
    std::string result_sequence_str = indices2str(result_sequence);
    for (int i = 0; i < std::min(20, static_cast<int>(prefixes.size())); i++) {
        const auto& prefix = prefixes[i];
//        std::cout << "\"" << indices2str(prefix->get_sentence()) << "\": " << std::exp(prefix->get_prev_full_prob())
//                  << ", " << prefix->get_prev_full_prob()
//                  << " | " << prefix->lm_weight << " " << prefix->prev_prob_blank << " "
//                  << prefix->prev_prob_not_blank << "\n";
    }
//    std::cout << "=======================" << std::endl;

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

    {
        ThreadPool pool{static_cast<size_t>(batch_size)};
        for (int i = 0; i < batch_size; i++) {
            pool.add_task([this, &argmax_logits, &logits_lengths, i,
                                  &decoded_targets, &decoded_targets_lengths, &decoded_sentences] {
                auto prev_symbol = blank_idx;
                auto current_len = 0;
                for (int j = 0; j < logits_lengths[i].item<int>(); j++) {
                    const auto current_symbol = argmax_logits[i][j].item<int>();
                    if (current_symbol != blank_idx && prev_symbol != current_symbol) {
                        decoded_targets[i][current_len] = current_symbol;
                        current_len++;
                    }
                    prev_symbol = current_symbol;
                }
                decoded_sentences[i] = indices2str(decoded_targets[i], current_len);
                decoded_targets_lengths[i] = current_len;
            });
        }
    }


    return {decoded_targets,
            decoded_targets_lengths,
            decoded_sentences};
}
