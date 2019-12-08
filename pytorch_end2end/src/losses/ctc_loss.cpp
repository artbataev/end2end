// Copyright 2019 Vladimir Bataev

#include "losses/ctc_loss.h"

#include <algorithm>
#include <limits>

#include "utils/math_utils.h"

using scalar_t = double;
constexpr auto kMinusInfinity = -std::numeric_limits<scalar_t>::infinity();

CTCLossEngine::CTCLossEngine(int blank_idx_) : blank_idx{blank_idx_} {}

void CTCLossEngine::compute_2d(
    const torch::Tensor& logits_2d,
    const torch::TensorAccessor<int64_t, 1>& targets_1d_a,
    int seq_len,
    int targets_len,
    int batch_i,
    torch::Tensor& losses,
    torch::Tensor& grads) const {
  const auto logits_2d_a = logits_2d.accessor<scalar_t, 2>();

  const auto ext_targets_len = targets_len * 2 + 1;
  auto extended_targets = torch::full(
      ext_targets_len, blank_idx, torch::TensorOptions().dtype(torch::kLong));
  auto extended_targets_a = extended_targets.accessor<int64_t, 1>();

  for (int i = 0; i < targets_len; i++)
    extended_targets_a[i * 2 + 1] = targets_1d_a[i];

  // forward - alpha
  auto log_alpha = torch::full({ext_targets_len, seq_len},
                               kMinusInfinity,
                               torch::TensorOptions().dtype(torch::kDouble));
  auto log_alpha_a = log_alpha.accessor<scalar_t, 2>();

  if (seq_len > 1 || ext_targets_len == 1)
    log_alpha_a[0][0] = logits_2d_a[0][extended_targets_a[0]];
  if (ext_targets_len > 1)
    log_alpha_a[1][0] = logits_2d_a[0][extended_targets_a[1]];
  for (int t = 1; t < seq_len; t++) {  // time step
    auto start = std::max(0, ext_targets_len - 2 * (seq_len - t));
    auto end = std::min(t * 2 + 2, ext_targets_len);

    for (int j = start; j < end; j++) {
      log_alpha_a[j][t] = log_alpha_a[j][t - 1];
      auto current_label = extended_targets_a[j];
      if (j > 0) {
        log_alpha_a[j][t] =
            log_sum_exp(log_alpha_a[j][t], log_alpha_a[j - 1][t - 1]);
        if (current_label != blank_idx && j - 2 >= 0 &&
            extended_targets_a[j - 2] != current_label) {
          log_alpha_a[j][t] =
              log_sum_exp(log_alpha_a[j][t], log_alpha_a[j - 2][t - 1]);
        }
      }
      log_alpha_a[j][t] += logits_2d_a[t][current_label];
    }
  }

  if (ext_targets_len > 1)
    losses[batch_i] =
        -log_sum_exp(log_alpha_a[ext_targets_len - 1][seq_len - 1],
                     log_alpha_a[ext_targets_len - 2][seq_len - 1]);
  else
    losses[batch_i] = -log_alpha[ext_targets_len - 1][seq_len - 1];

  auto loss_forward = -losses[batch_i];

  // backward - beta
  auto log_beta = torch::full_like(log_alpha, kMinusInfinity);
  auto log_beta_a = log_beta.accessor<scalar_t, 2>();

  if (seq_len > 1 || ext_targets_len == 1)
    log_beta_a[ext_targets_len - 1][seq_len - 1] = 0;
  if (ext_targets_len > 1) log_beta_a[ext_targets_len - 2][seq_len - 1] = 0;
  for (int t = seq_len - 2; t >= 0; t--) {  // time steps
    auto start = std::max(0, ext_targets_len - 2 * (seq_len - t));
    auto end = std::min(t * 2 + 2, ext_targets_len);
    for (int j = start; j < end; j++) {
      auto current_label = extended_targets_a[j];
      log_beta_a[j][t] =
          log_beta_a[j][t + 1] + logits_2d_a[t + 1][extended_targets_a[j]];
      if (j < ext_targets_len - 1) {
        log_beta_a[j][t] =
            log_sum_exp(log_beta_a[j][t],
                        log_beta_a[j + 1][t + 1] +
                            logits_2d_a[t + 1][extended_targets_a[j + 1]]);
        if (current_label != blank_idx && j + 2 < ext_targets_len &&
            extended_targets_a[j + 2] != current_label) {
          log_beta_a[j][t] =
              log_sum_exp(log_beta_a[j][t],
                          log_beta_a[j + 2][t + 1] +
                              logits_2d_a[t + 1][extended_targets_a[j + 2]]);
        }
      }
    }
  }

  // compute gradient
  auto alpha_beta = log_alpha + log_beta;
  auto alpha_beta_a = alpha_beta.accessor<scalar_t, 2>();
  auto prob_sum = torch::full(
      {logits_2d_a.size(0), logits_2d_a.size(1)},
      kMinusInfinity,
      torch::TensorOptions().dtype(torch::kDouble));  // seq_len, alphabet_size
  auto prob_sum_a = prob_sum.accessor<scalar_t, 2>();
  for (int i = 0; i < ext_targets_len; i++) {
    auto current_label = extended_targets_a[i];
    for (int j = 0; j < seq_len; j++)
      prob_sum_a[j][current_label] =
          log_sum_exp(prob_sum_a[j][current_label], alpha_beta_a[i][j]);
  }
  const auto negative_term = prob_sum - loss_forward;
  grads[batch_i] = torch::exp(logits_2d) - torch::exp(negative_term);
}
