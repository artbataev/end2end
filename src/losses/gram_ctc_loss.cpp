// Copyright 2019 Vladimir Bataev

#include "losses/gram_ctc_loss.h"

GramCTCLossEngine::GramCTCLossEngine(
    int blank_idx,
    int num_base_labels,
    int total_labels,
    const std::unordered_map<int, std::vector<int>>& label2ids)
    : blank_idx_{blank_idx},
      num_base_labels_{num_base_labels},
      total_labels_{total_labels} {
  size_t max_order = 1;
  for (const auto& elem : label2ids) {
    max_order = std::max(max_order, elem.second.size());
    const auto label_id = elem.first;
    const auto label_hash = get_hash(elem.second);
    ids2label_[label_hash] = label_id;
  }
}

int64_t GramCTCLossEngine::get_hash(const std::vector<int>& labels) const {
  int64_t result = 0;
  for (const auto& label_id : labels) {
    result = result * num_base_labels_ + label_id;
  }
  return result;
}

void GramCTCLossEngine::compute_2d(
    const torch::Tensor& logits_2d,
    const torch::TensorAccessor<int64_t, 1>& targets_1d_a,
    int seq_len,
    int targets_len,
    int batch_i,
    torch::Tensor& losses,
    torch::Tensor& grads) const {}
