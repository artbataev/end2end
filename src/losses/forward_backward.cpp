// Copyright 2019 Vladimir Bataev

#include "losses/forward_backward.h"

#include "utils/threadpool.h"

std::tuple<at::Tensor, at::Tensor> ForwardBackwardBase::compute(
    const at::Tensor& logits_,
    const at::Tensor& targets_,
    const at::Tensor& logits_lengths_,
    const at::Tensor& targets_lengths_) {
  const auto src_device = logits_.device();
  constexpr auto work_device = torch::kCPU;

  const auto logits = logits_.to(work_device).to(torch::kDouble).detach();
  const auto targets = targets_.to(work_device).to(torch::kLong);
  const auto logits_lengths = logits_lengths_.to(work_device).to(torch::kLong);
  const auto targets_lengths =
      targets_lengths_.to(work_device).to(torch::kLong);

  //  const auto logits_a = logits.accessor<double, 3>();
  const auto targets_a = targets.accessor<int64_t, 2>();
  const auto logits_lengths_a = logits_lengths.accessor<int64_t, 1>();
  const auto targets_lengths_a = targets_lengths.accessor<int64_t, 1>();

  const auto batch_size = logits_lengths.size(0);

  const auto options = torch::TensorOptions()
                           .dtype(logits.dtype())
                           .layout(torch::kStrided)
                           .device(work_device)
                           .requires_grad(false);
  auto losses = torch::zeros(batch_size, options);
  auto grads = torch::zeros_like(logits);

  {
    ThreadPool pool{static_cast<size_t>(batch_size)};
    for (int i = 0; i < batch_size; ++i) {
      auto seq_len = logits_lengths_a[i];
      auto targets_len = targets_lengths_a[i];
      pool.add_task([this,
                     &logits,
                     &targets_a,
                     i,
                     seq_len,
                     targets_len,
                     &losses,
                     &grads] {
        compute_2d(
            logits[i], targets_a[i], seq_len, targets_len, i, losses, grads);
      });
    }
  }

  losses = losses.to(src_device).to(logits_.dtype());
  grads = grads.to(src_device).to(logits_.dtype());

  return {losses, grads};
}
