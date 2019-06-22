// Copyright 2019 Vladimir Bataev

#pragma once

#include <torch/extension.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <map>

#include "lm/model.hh"

using word2index_t = std::unordered_map<std::string, lm::WordIndex>;
using lm_state_t = lm::ngram::State;

class CTCDecoder {
 public:
  CTCDecoder(int blank_idx_, int beam_width_, std::vector<std::string> labels_,
             const std::string& lm_path, double lmwt_, double wip_,
             double oov_penalty_, bool case_sensitive_);

  lm::WordIndex get_idx(const std::string& word) const;
  lm::WordIndex get_idx(const std::vector<int>& word_int) const;

  void print_scores_for_sentence(std::vector<std::string> words) const;

  std::tuple<at::Tensor, at::Tensor, std::vector<std::string>> decode_greedy(
      const at::Tensor& logits_, const at::Tensor& logits_lengths_);

  std::tuple<at::Tensor, at::Tensor, std::vector<std::string>> decode(
      const at::Tensor& logits_, const at::Tensor& logits_lengths_);

 private:
  int blank_id;
  int space_id;
  int beam_width;
  bool case_sensitive;
  double lmwt;
  double wip;
  double oov_penalty;
  std::vector<std::string> labels;
  std::unique_ptr<lm::ngram::ProbingModel> lm_model;
  word2index_t word2index;

  std::tuple<std::vector<int>, int, std::string> decode_sentence(
      const at::TensorAccessor<double, 2>& logits, int sequence_len);

  std::string indices2str(const std::vector<int>& char_ids);

  std::string indices2str(const at::TensorAccessor<int64_t, 1>& char_ids,
                          int len);

  double get_score_for_sentence(std::vector<std::string> words) const;

  double get_score_for_sentence(const std::vector<int>& sentence) const;

  double get_score_for_sentence(
      const std::vector<std::vector<int>>& words_int) const;

  struct Prefix : public std::enable_shared_from_this<Prefix> {
    Prefix();

    double prob_blank;
    double prob_not_blank;
    double prev_prob_blank;
    double prev_prob_not_blank;
    int last_char;
    double lm_score;
    double lm_score_before_last;
    int num_words;
    int num_oov_words;
    int num_oov_words_before_last;
    std::vector<int> last_word{};
    lm_state_t lm_state_before_last;
    lm_state_t lm_state;
    std::vector<std::vector<int>> words{};
    std::shared_ptr<Prefix> parent;
    std::map<int, std::weak_ptr<Prefix>> next_data;

    std::vector<int> get_sentence();

    double get_prev_full_prob() const;

    void next_step();

    void repr(std::function<std::string(const std::vector<int>&)> indices2str);
  };

  std::shared_ptr<CTCDecoder::Prefix> get_initial_prefix() const;

  std::pair<std::shared_ptr<CTCDecoder::Prefix>, bool> get_next_prefix(
      std::shared_ptr<CTCDecoder::Prefix>& prefix, int char_id) const;
  double get_prev_full_prob_with_lmwt(const CTCDecoder::Prefix* prefix) const;
};

PYBIND11_MODULE(cpp_ctc_decoder, m) {
  namespace py = pybind11;
  using namespace pybind11::literals;
  py::class_<CTCDecoder>(m, "CTCDecoder")
      .def(py::init<int, int, std::vector<std::string>, std::string, double,
                    double, double, bool>(),
           "blank_idx"_a, "beam_width_"_a = 100,
           "labels"_a = std::vector<std::string>{}, "lm_path"_a = "",
           "lmwt_"_a = 1.0, "wip_"_a = 0.0, "oov_penalty_"_a = -1000.0,
           "case_sensitive"_a = false)
      .def("decode_greedy", &CTCDecoder::decode_greedy, "Decode greedy",
           "logits_"_a, "logits_lengths_"_a)
      .def("decode", &CTCDecoder::decode, "Decode greedy", "logits_"_a,
           "logits_lengths_"_a)
      .def("print_scores_for_sentence", &CTCDecoder::print_scores_for_sentence,
           "Print scores", "words"_a);
}
