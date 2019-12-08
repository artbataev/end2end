// Copyright 2019 Vladimir Bataev

#pragma once

#include <cmath>
#include <limits>

template <typename scalar_t>
scalar_t log_sum_exp(const scalar_t log_prob_1, const scalar_t log_prob_2) {
  constexpr auto kMinusInfinity = -std::numeric_limits<scalar_t>::infinity();
  if (log_prob_1 == kMinusInfinity) return log_prob_2;
  if (log_prob_2 == kMinusInfinity) return log_prob_1;
  if (log_prob_1 > log_prob_2)
    return log_prob_1 + std::log(1.0 + std::exp(log_prob_2 - log_prob_1));
  return log_prob_2 + std::log(1.0 + std::exp(log_prob_1 - log_prob_2));
}
