#include <torch/extension.h>

torch::Tensor log_sum_exp(const torch::Tensor& log_prob_1, const torch::Tensor& log_prob_2);
double log_sum_exp(const double log_prob_1, const double log_prob_2);
