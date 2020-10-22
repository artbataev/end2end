// Copyright 2019 Vladimir Bataev

#include "decoders/ctc_decoder.h"

#include <torch/extension.h>  // pytorch

#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

#include "lm/model.hh"  // kenlm
#include "utils/math_utils.h"
#include "utils/threadpool.h"

const double kLogE10 = std::log(10.0);

class CustomEnumerateVocab : public lm::EnumerateVocab {
 public:
  CustomEnumerateVocab() = default;

  void Add(lm::WordIndex index, const StringPiece& str) override {
    word2index[{str.data(), str.length()}] = index;
  };

  word2index_t get_word2index() { return word2index; }

 private:
  word2index_t word2index;
};

std::string str_to_lower(const std::string& str) {
  auto result = str;
  std::transform(result.begin(), result.end(), result.begin(), ::tolower);
  return result;
}

CTCDecoder::CTCDecoder(int blank_idx_,
                       int beam_width_ = 100,
                       std::vector<std::string> labels_ = {},
                       const std::string& lm_path = "",
                       double lmwt_ = 1.0,
                       double wip_ = 0.0,
                       double oov_penalty_ = -1000.0,
                       bool case_sensitive_ = false)
    : blank_id(blank_idx_),
      beam_width{beam_width_},
      space_id{-1},
      lmwt{lmwt_},
      wip{wip_},
      oov_penalty{oov_penalty_},
      labels(std::move(labels_)),
      case_sensitive(case_sensitive_),
      lm_model{nullptr} {
  if (!labels.empty()) {
    space_id = static_cast<int>(std::distance(
        labels.begin(), std::find(labels.begin(), labels.end(), " ")));
    if (space_id >= labels.size()) space_id = -1;
  }
  if (!lm_path.empty()) {
    auto enumerate_vocab = CustomEnumerateVocab{};
    lm::ngram::Config config;
    config.enumerate_vocab = &enumerate_vocab;
    lm_model = std::unique_ptr<lm::ngram::Model>(static_cast<lm::ngram::Model*>(
        lm::ngram::LoadVirtual(lm_path.c_str(), config, lm::ngram::PROBING)));
    if (case_sensitive) {
      word2index = enumerate_vocab.get_word2index();
    } else {
      for (const auto& elem : enumerate_vocab.get_word2index())
        word2index[str_to_lower(elem.first)] = elem.second;
    }
  } else {
    lmwt = 0;
  }
}

lm::WordIndex CTCDecoder::get_idx(const std::string& word) const {
  if (case_sensitive) return lm_model->GetVocabulary().Index(word);
  auto word_to_find = str_to_lower(word);
  if (word2index.count(word_to_find) > 0) return word2index.at(word_to_find);
  return lm_model->GetVocabulary().NotFound();
}

lm::WordIndex CTCDecoder::get_idx(const std::vector<int>& word_int) const {
  std::stringstream word_s;
  for (const auto c : word_int) word_s << labels[c];
  return get_idx(word_s.str());
}

double CTCDecoder::get_score_for_sentence(
    std::vector<std::string> words) const {
  if (lm_model == nullptr) return 0;

  double result = 0;
  double penalty = 0;
  lm_state_t state(lm_model->BeginSentenceState()), out_state;
  for (const auto& word : words) {
    auto word_idx = get_idx(word);
    result += lm_model->BaseScore(&state, word_idx, &out_state);  // ? Score
    state = out_state;
    if (word_idx == 0) penalty += oov_penalty;
  }
  return result / kLogE10 + penalty;
}

double CTCDecoder::get_score_for_sentence(
    const std::vector<int>& sentence) const {
  if (lm_model == nullptr) return 0;

  std::vector<std::string> words;
  std::string word;
  for (const auto& c_id : sentence) {
    if (c_id == blank_id)  // first
      continue;

    if (c_id != space_id) {
      word += labels[c_id];
    } else if (!word.empty()) {
      words.emplace_back(word);
      word = "";
    }
  }
  if (!word.empty()) words.emplace_back(word);
  if (!words.empty()) return get_score_for_sentence(words);
  throw std::logic_error("incorrect scoring: empty sentence");
}

double CTCDecoder::get_score_for_sentence(
    const std::vector<std::vector<int>>& words_int) const {
  if (lm_model == nullptr) return 0;

  std::vector<std::string> words{words_int.size(), ""};
  std::string word;
  for (int i = 0; i < words_int.size(); i++) {
    for (const auto& c_id : words_int[i]) words[i] += labels[c_id];
  }
  if (!words.empty()) return get_score_for_sentence(words);
  throw std::logic_error("incorrect scoring: empty sentence");
}

void CTCDecoder::print_scores_for_sentence(
    std::vector<std::string> words) const {
  if (lm_model == nullptr) return;
  lm_state_t state(lm_model->BeginSentenceState()), out_state;
  for (const auto& word : words) {
    std::cout << word << " " << get_idx(word) << " "
              << lm_model->GetVocabulary().Index(word) << " "
              << lm_model->Score(state, get_idx(word), out_state) << "\n";
    state = out_state;
  }
}

std::tuple<at::Tensor, at::Tensor, std::vector<std::string>> CTCDecoder::decode(
    const at::Tensor& logits_, const at::Tensor& logits_lengths_) {
  const auto work_device = torch::kCPU;

  auto logits = logits_.detach().to(work_device).to(torch::kDouble);
  auto logits_lengths = logits_lengths_.to(work_device).to(torch::kLong);
  auto logits_a = logits.accessor<double, 3>();
  auto logits_lengths_a = logits_lengths.accessor<int64_t, 1>();

  auto decoded_targets_lengths = at::zeros_like(logits_lengths);
  auto decoded_targets_lengths_a =
      decoded_targets_lengths.accessor<int64_t, 1>();
  auto batch_size = logits_lengths.size(0);

  std::vector<std::string> decoded_sentences{static_cast<size_t>(batch_size),
                                             ""};
  std::vector<std::vector<int>> decoded_indices_vec{
      static_cast<size_t>(batch_size), std::vector<int>{}};

  {
    ThreadPool pool{static_cast<size_t>(batch_size)};
    for (int i = 0; i < batch_size; i++) {
      pool.add_task([this,
                     &logits_a,
                     &logits_lengths_a,
                     i,
                     &decoded_sentences,
                     &decoded_indices_vec,
                     &decoded_targets_lengths_a] {
        int current_sequence_len = 0;
        std::tie(decoded_indices_vec[i],
                 current_sequence_len,
                 decoded_sentences[i]) =
            decode_sentence(logits_a[i], logits_lengths_a[i]);
        decoded_targets_lengths_a[i] = current_sequence_len;
      });
    }
  }

  auto max_sequence_len = decoded_targets_lengths.max().item<int64_t>();
  auto decoded_indices =
      at::zeros({batch_size, max_sequence_len}, logits_lengths.options());
  for (int i = 0; i < batch_size; i++) {
    for (int l = 0; l < decoded_targets_lengths_a[i]; l++) {
      decoded_indices[i][l] = decoded_indices_vec[i][l];
    }
  }
  return {decoded_indices, decoded_targets_lengths, decoded_sentences};
}

std::string CTCDecoder::indices2str(const std::vector<int>& char_ids) {
  std::string result;
  result.reserve(char_ids.size());
  if (!labels.empty()) {
    for (const auto& char_id : char_ids) result += labels[char_id];
  }
  return result;
}

std::string CTCDecoder::indices2str(
    const at::TensorAccessor<int64_t, 1>& char_ids, int len) {
  std::string result;
  result.reserve(static_cast<size_t>(len));
  if (!labels.empty()) {
    for (int i = 0; i < len; i++) result += labels[char_ids[i]];
  }
  return result;
}

std::shared_ptr<CTCDecoder::Prefix> CTCDecoder::get_initial_prefix() const {
  auto prefix = std::make_shared<CTCDecoder::Prefix>();
  prefix->prev_prob_blank = 0.0;
  if (lm_model != nullptr) {
    prefix->lm_state_before_last = lm_model->BeginSentenceState();
    prefix->lm_state = lm_model->BeginSentenceState();
  }
  return prefix;
}

std::vector<int> CTCDecoder::Prefix::get_sentence() {
  std::vector<int> result;
  result.reserve(static_cast<size_t>(num_words) * 6);
  result.emplace_back(last_char);
  auto& prev = parent;
  while (prev != nullptr) {
    if (prev->parent != nullptr)  // blank
      result.emplace_back(prev->last_char);
    prev = prev->parent;
  }

  std::reverse(result.begin(), result.end());
  return result;
}

std::pair<std::shared_ptr<CTCDecoder::Prefix>, bool>
CTCDecoder::get_next_prefix(std::shared_ptr<CTCDecoder::Prefix>& prefix,
                            const int char_id) const {
  if (prefix->next_data.count(char_id) > 0)
    if (auto next_prefix = prefix->next_data.at(char_id).lock())
      return {next_prefix, false};

  auto new_prefix = std::make_shared<Prefix>();
  prefix->next_data[char_id] = new_prefix;
  new_prefix->last_char = char_id;
  new_prefix->num_words = prefix->num_words;
  auto is_new_word = char_id != space_id &&
                     (prefix->num_words == 0 || prefix->last_char == space_id);
  if (is_new_word) {
    new_prefix->num_words++;
  }

  if (lm_model != nullptr) {
    if (is_new_word) {
      new_prefix->last_word = {
          char_id,
      };
      auto word_idx = get_idx(new_prefix->last_word);
      new_prefix->lm_state_before_last = prefix->lm_state;
      new_prefix->lm_score_before_last = prefix->lm_score;

      new_prefix->lm_score +=
          new_prefix->lm_score_before_last +
          lm_model->BaseScore(&new_prefix->lm_state_before_last,
                              word_idx,
                              &new_prefix->lm_state) /
              kLogE10;

      new_prefix->num_oov_words_before_last = prefix->num_oov_words;
      new_prefix->num_oov_words = prefix->num_oov_words + (word_idx == 0);
    } else if (char_id != space_id) {
      new_prefix->last_word = prefix->last_word;
      new_prefix->last_word.emplace_back(char_id);
      auto word_idx = get_idx(new_prefix->last_word);
      new_prefix->lm_state_before_last = prefix->lm_state_before_last;
      new_prefix->lm_score_before_last = prefix->lm_score_before_last;

      new_prefix->lm_score +=
          new_prefix->lm_score_before_last +
          lm_model->BaseScore(&new_prefix->lm_state_before_last,
                              word_idx,
                              &new_prefix->lm_state) /
              kLogE10;
      new_prefix->num_oov_words_before_last = prefix->num_oov_words_before_last;
      new_prefix->num_oov_words =
          new_prefix->num_oov_words_before_last + (word_idx == 0);

    } else {  // char_id == space_id, do nothing, just copy
      new_prefix->last_word = prefix->last_word;
      new_prefix->lm_score = prefix->lm_score;
      new_prefix->lm_score_before_last = prefix->lm_score_before_last;
      new_prefix->num_oov_words = prefix->num_oov_words;
      new_prefix->num_oov_words_before_last = prefix->num_oov_words_before_last;
      new_prefix->lm_state = prefix->lm_state;
      new_prefix->lm_state_before_last = prefix->lm_state_before_last;
    }
  }

  new_prefix->parent = prefix;
  return {new_prefix, true};
}

double CTCDecoder::get_prev_full_prob_with_lmwt(
    const CTCDecoder::Prefix* prefix) const {
  return prefix->get_prev_full_prob() + prefix->lm_score * lmwt -
         prefix->num_words * wip + prefix->num_oov_words * oov_penalty;
}

CTCDecoder::Prefix::Prefix()
    : prob_blank{-INFINITY},
      prob_not_blank{-INFINITY},
      prev_prob_blank{-INFINITY},
      prev_prob_not_blank{-INFINITY},
      last_char{-1},
      next_data{},
      parent{nullptr},
      lm_score{0},
      num_words{0},
      num_oov_words{0},
      num_oov_words_before_last{0} {}

double CTCDecoder::Prefix::get_prev_full_prob() const {
  return log_sum_exp(prev_prob_not_blank, prev_prob_blank);
}

void CTCDecoder::Prefix::next_step() {
  prev_prob_blank = prob_blank;
  prev_prob_not_blank = prob_not_blank;
  prob_blank = -INFINITY;
  prob_not_blank = -INFINITY;
}

void CTCDecoder::Prefix::repr(
    std::function<std::string(const std::vector<int>&)> indices2str) {
  std::cout << "\"" << indices2str(get_sentence()) << "\": \n";
  std::cout << "words: " << num_words
            << " | prob: " << std::exp(get_prev_full_prob()) << ", "
            << get_prev_full_prob();
  std::cout << " | lm_score: " << lm_score << "\n";
}

std::tuple<std::vector<int>, int, std::string> CTCDecoder::decode_sentence(
    const at::TensorAccessor<double, 2>& logits_a, int sequence_len) {
  // Prefix beam search: https://arxiv.org/pdf/1408.2873.pdf
  auto alphabet_size = static_cast<int>(logits_a.size(1));

  // NB: prob - log-probabilities
  std::vector<std::shared_ptr<Prefix>> prefixes;
  prefixes.emplace_back(get_initial_prefix());

  // at every timestep
  std::vector<std::shared_ptr<Prefix>> new_prefixes;

  for (int l = 0; l < sequence_len; l++) {
    auto cur_log_prob_blank = logits_a[l][blank_id];

    new_prefixes.reserve(prefixes.size() * alphabet_size);
    // for every character
    for (int char_id = 0; char_id < alphabet_size; char_id++) {
      auto cur_prob = logits_a[l][char_id];
      // for every prefix
      for (auto& prefix : prefixes) {
        if (char_id == blank_id) {
          prefix->prob_blank = log_sum_exp(
              prefix->prob_blank, cur_prob + prefix->get_prev_full_prob());
        } else {
          auto new_prefix_res = get_next_prefix(prefix, char_id);
          auto& new_prefix = new_prefix_res.first;
          auto& is_new = new_prefix_res.second;
          if (is_new) new_prefixes.emplace_back(new_prefix);

          if (char_id == prefix->last_char) {  // repeated character
            new_prefix->prob_not_blank = log_sum_exp(
                new_prefix->prob_not_blank, cur_prob + prefix->prev_prob_blank);
            prefix->prob_not_blank = log_sum_exp(
                prefix->prob_not_blank, cur_prob + prefix->prev_prob_not_blank);
          } else {
            new_prefix->prob_not_blank =
                log_sum_exp(new_prefix->prob_not_blank,
                            cur_prob + prefix->get_prev_full_prob());
          }
        }
      }  // end for: prefixes
    }    // end for: characters

    prefixes.reserve(prefixes.size() + new_prefixes.size());
    prefixes.insert(prefixes.end(),
                    std::make_move_iterator(new_prefixes.begin()),
                    std::make_move_iterator(new_prefixes.end()));
    new_prefixes.resize(0);

    for (auto& prefix : prefixes) prefix->next_step();

    if (prefixes.size() > beam_width) {
      std::nth_element(prefixes.begin(),
                       std::next(prefixes.begin(), beam_width),
                       prefixes.end(),
                       [this](const std::shared_ptr<Prefix>& lhs,
                              const std::shared_ptr<Prefix>& rhs) {
                         return get_prev_full_prob_with_lmwt(lhs.get()) >
                                get_prev_full_prob_with_lmwt(rhs.get());
                       });
      prefixes.resize(static_cast<size_t>(beam_width));
    }
  }  // end for: timestep

  std::sort(prefixes.begin(),
            prefixes.end(),
            [this](const std::shared_ptr<Prefix>& lhs,
                   const std::shared_ptr<Prefix>& rhs) {
              return get_prev_full_prob_with_lmwt(lhs.get()) >
                     get_prev_full_prob_with_lmwt(rhs.get());
            });

  //    for (auto& prefix: prefixes) {
  //        prefix->repr([this](const std::vector<int>& s) {
  //            return indices2str(s);
  //        });
  //    }

  //    std::cout << "-->" << get_score_for_sentence(prefixes[0]->sentence) <<
  //    "\n";

  std::vector<int> result_sequence{prefixes[0]->get_sentence()};
  std::string result_sequence_str = indices2str(result_sequence);

  return {result_sequence,
          static_cast<int>(result_sequence.size()),
          result_sequence_str};
}

std::tuple<at::Tensor, at::Tensor, std::vector<std::string>>
CTCDecoder::decode_greedy(const at::Tensor& logits_,
                          const at::Tensor& logits_lengths_) {
  // collapse repeated, remove blank
  constexpr auto work_device = torch::kCPU;

  const auto logits_lengths = logits_lengths_.to(work_device).to(torch::kLong);
  const auto batch_size = logits_lengths.size(0);
  const auto argmax_logits = logits_.argmax(-1).to(work_device);
  auto decoded_targets = at::zeros_like(argmax_logits);
  auto decoded_targets_lengths = at::zeros_like(logits_lengths);
  std::vector<std::string> decoded_sentences(static_cast<size_t>(batch_size),
                                             "");

  const auto logits_lengths_a = logits_lengths.accessor<int64_t, 1>();
  auto argmax_logits_a = argmax_logits.accessor<int64_t, 2>();
  auto decoded_targets_a = decoded_targets.accessor<int64_t, 2>();
  auto decoded_targets_lengths_a =
      decoded_targets_lengths.accessor<int64_t, 1>();

  {
    ThreadPool pool{static_cast<size_t>(batch_size)};
    for (int i = 0; i < batch_size; i++) {
      pool.add_task([this,
                     &argmax_logits_a,
                     &logits_lengths_a,
                     i,
                     &decoded_targets_a,
                     &decoded_targets_lengths_a,
                     &decoded_sentences] {
        auto prev_symbol = blank_id;
        auto current_len = 0;
        for (int j = 0; j < logits_lengths_a[i]; j++) {
          const auto current_symbol = argmax_logits_a[i][j];
          if (current_symbol != blank_id && prev_symbol != current_symbol) {
            decoded_targets_a[i][current_len] = current_symbol;
            current_len++;
          }
          prev_symbol = current_symbol;
        }
        decoded_sentences[i] = indices2str(decoded_targets_a[i], current_len);
        decoded_targets_lengths_a[i] = current_len;
      });
    }
  }

  return {decoded_targets, decoded_targets_lengths, decoded_sentences};
}
