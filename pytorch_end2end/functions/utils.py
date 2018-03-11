import numba

import numpy as np


@numba.vectorize([numba.float64(numba.float64, numba.float64)]) # (nogil=True, nopython=True)
def log_sum_exp(log_prob_1, log_prob_2):
    """
    ln(a + b) = ln(a) + ln(1 + exp(ln(b) - ln(a)))
    :param log_prob_1:
    :param log_prob_2:
    :return:
    """
    # if log_prob_1 == -np.inf and log_prob_2 == -np.inf:
    #     return -np.inf
    # not necessary
    if log_prob_1 == -np.inf:
        return log_prob_2
    if log_prob_2 == -np.inf:
        return log_prob_1
    if log_prob_1 > log_prob_2:
        return log_prob_1 + np.log(1.0 + np.exp(log_prob_2 - log_prob_1))
    else:
        return log_prob_2 + np.log(1.0 + np.exp(log_prob_1 - log_prob_2))
