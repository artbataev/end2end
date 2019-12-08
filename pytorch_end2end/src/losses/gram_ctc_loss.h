// Copyright 2019 Vladimir Bataev

#pragma once

#include <unordered_map>

#include "losses/forward_backward.h"

class GramCTCLossEngine : public ForwardBackwardBase {
 public:
  GramCTCLossEngine(
      int blank_idx,
      int num_base_labels,
      int total_labels,
      const std::unordered_map<int, std::vector<int>>& label2ids);

 private:
  void compute_2d(const torch::Tensor& logits_2d,
                  const torch::TensorAccessor<int64_t, 1>& targets_1d_a,
                  int seq_len,
                  int targets_len,
                  int batch_i,
                  torch::Tensor& losses,
                  torch::Tensor& grads) const override;
  int64_t get_hash(const std::vector<int>& labels) const;

  int blank_idx_;
  int num_base_labels_;
  int total_labels_;
  std::unordered_map<int64_t, int> ids2label_;
};
