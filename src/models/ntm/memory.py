"""Reading, writing and addressing the NTM's memory matrix.

Memory is a tensor of shape (batch, locations, width). It is state, not weights: the model
carries it from one timestep to the next, so it is passed in and handed back.
"""

import torch
import torch.nn.functional as F

EPS = 1e-8


def read(memory, w):
    """Weighted average of the memory locations. (B, N, M) and (B, N) -> (B, M)."""
    return (w.unsqueeze(-1) * memory).sum(dim=1)


def write(memory, w, erase, add):
    """Erase then add, both scaled by the weighting, so untouched locations (w = 0) keep their value."""
    w = w.unsqueeze(-1)  # (B, N, 1), broadcasts across the width of each location
    memory = memory * (1 - w * erase.unsqueeze(1))
    return memory + w * add.unsqueeze(1)


def address(memory, previous_w, key, strength, gate, shift_weights, sharpness):
    """Figure 2 of the paper: the four stages that turn a head's outputs into a weighting.

    Returns (B, N), non-negative and summing to 1: how much the head attends to each location.
    """
    # 1. Content: how much does each location look like the key? (equations 5 and 6)
    similarity = F.cosine_similarity(memory, key.unsqueeze(1), dim=-1, eps=EPS)
    w = F.softmax(strength * similarity, dim=1)

    # 2. Interpolate: gate 1 uses that, gate 0 keeps the head where it was last step (equation 7)
    w = gate * w + (1 - gate) * previous_w

    # 3. Shift: circular convolution, moving attention to neighbouring locations (equation 8)
    pad = shift_weights.shape[1] // 2
    padded = torch.cat([w[:, -pad:], w, w[:, :pad]], dim=1)  # wrap around both ends
    windows = padded.unfold(dimension=1, size=shift_weights.shape[1], step=1)  # (B, N, shifts)
    w = (windows * shift_weights.flip(-1).unsqueeze(1)).sum(dim=-1)

    # 4. Sharpen: a power above 1 makes the weighting peakier again (equation 9)
    w = w.clamp(min=EPS) ** sharpness
    return w / (w.sum(dim=1, keepdim=True) + EPS)
