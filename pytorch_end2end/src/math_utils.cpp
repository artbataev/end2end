#include "math_utils.h"

double log_sum_exp(const double log_prob_1, const double log_prob_2) {
    if (log_prob_1 == -INFINITY)
        return log_prob_2;
    if (log_prob_2 == -INFINITY)
        return log_prob_1;
    if (log_prob_1 > log_prob_2)
        return log_prob_1 + std::log(1.0 + std::exp(log_prob_2 - log_prob_1));
    return log_prob_2 + std::log(1.0 + std::exp(log_prob_1 - log_prob_2));
}

torch::Tensor log_sum_exp(const torch::Tensor& log_prob_1, const torch::Tensor& log_prob_2) {
    if (log_prob_1.item<double>() == -INFINITY)
        return log_prob_2;
    if (log_prob_2.item<double>() == -INFINITY)
        return log_prob_1;
    if (log_prob_1.item<double>() > log_prob_2.item<double>())
        return log_prob_1 + torch::log(1.0 + torch::exp(log_prob_2 - log_prob_1));
    return log_prob_2 + torch::log(1.0 + torch::exp(log_prob_1 - log_prob_2));
}
