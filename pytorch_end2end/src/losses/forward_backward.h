// Copyright 2019 Vladimir Bataev

#pragma once
#include <torch/extension.h>

class ForwardBackwardBase {
 public:
  virtual void compute_2d(const torch::Tensor& logits_2d,
                          const torch::TensorAccessor<int64_t, 1>& targets_1d_a,
                          int seq_len,
                          int targets_len,
                          int batch_i,
                          torch::Tensor& losses,
                          torch::Tensor& grads) = 0;
  virtual ~ForwardBackwardBase() = default;
  std::tuple<at::Tensor, at::Tensor> compute(const at::Tensor& logits,
                                             const at::Tensor& targets,
                                             const at::Tensor& logits_lengths,
                                             const at::Tensor& targets_lengths);
};
