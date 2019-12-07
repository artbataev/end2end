// Copyright 2019 Vladimir Bataev

#pragma once

#include <torch/extension.h>

#include <tuple>

#include "losses/forward_backward.h"

class CTCLossEngine : public ForwardBackwardBase {
 public:
  explicit CTCLossEngine(int blank_idx_);

 private:
  void compute_2d(const torch::Tensor& logits_2d,
                  const torch::TensorAccessor<int64_t, 1>& targets_1d_a,
                  int seq_len, int targets_len, int batch_i,
                  torch::Tensor& losses, torch::Tensor& grads) override;

  int blank_idx;
};
