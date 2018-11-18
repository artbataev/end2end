#include <torch/extension.h>
#include "ctc_loss.h"

std::tuple<
        at::Tensor,
        at::Tensor
> ctc_loss_forward(
        const at::Tensor& logits,
        const at::Tensor& logits_lengths) {

    auto batch_size = logits_lengths.size(0);
    auto losses = at::zeros(batch_size);
    auto grads = at::zeros_like(logits);
    return {losses, grads};
}

