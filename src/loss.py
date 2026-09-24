"""The paper's cost: cross-entropy over the scored timesteps, in bits per sequence.

Shared by the eager training step and the captured CUDA graph, so the two compute the same
thing from the same expression.
"""

import math

import torch.nn.functional as F


def masked_bce(logits, target, mask):
    """Returns the loss to backpropagate and the cost of one sequence in bits.

    The loss is the mean cross-entropy over the bits `mask` marks, which is what the copy task
    minimised before masks existed. The cost is the same quantity summed over a sequence and
    converted to log base 2, the paper's metric: 84 bits is random guessing on a length-20 copy.
    """
    scored = mask.unsqueeze(-1)  # (batch, timesteps, 1), broadcast over the output channels
    per_bit = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * scored
    bits = mask.sum() * target.shape[-1]  # scored bits in the whole batch
    loss = per_bit.sum() / bits.clamp(min=1)  # clamped for the CUDA graph's empty warm-up buffers
    return loss, loss.detach() * bits / (mask.shape[0] * math.log(2))
