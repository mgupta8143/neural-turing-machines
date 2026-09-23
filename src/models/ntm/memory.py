"""Reading, writing and addressing the NTM's memory matrix.

Memory is a tensor of shape (batch, locations, width). It is state, not weights: the model
carries it from one timestep to the next, so it is passed in and handed back.

Addressing is the paper's Figure 2, one function per stage. Every stage takes a weighting of
shape (batch, locations) and returns another one, non-negative and summing to 1.
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


def content_weighting(memory, key, strength):
    """Stage 1, equations 5 and 6: attend to locations that look like the key.

    Strength turns the cosine similarities into a weighting that can be flat (strength 0) or
    concentrated on the single best match (large strength).
    """
    similarity = F.cosine_similarity(memory, key.unsqueeze(1), dim=-1, eps=EPS)  # (B, N)
    return F.softmax(strength * similarity, dim=1)


def interpolate(content_w, previous_w, gate):
    """Stage 2, equation 7: gate 1 takes the content weighting, gate 0 stays where the head was."""
    return gate * content_w + (1 - gate) * previous_w


def shift(w, shift_weights):
    """Stage 3, equation 8: a circular convolution that rotates attention to nearby locations.

    shift_weights is a distribution over the allowed moves, e.g. (-1, 0, +1) for three of them,
    and the rotation wraps around the ends of the memory.
    """
    span = shift_weights.shape[1]
    pad = span // 2
    padded = torch.cat([w[:, -pad:], w, w[:, :pad]], dim=1)  # wrap both ends
    windows = padded.unfold(dimension=1, size=span, step=1)  # (B, N, span): each location's neighbourhood
    return (windows * shift_weights.flip(-1).unsqueeze(1)).sum(dim=-1)


def sharpen(w, sharpness):
    """Stage 4, equation 9: a power of at least 1 makes the weighting peakier, undoing shift blur.

    w ** sharpness, renormalised, is the same as softmax(sharpness * log w), and the softmax
    form is what we use: raising small weights to a high power underflows to zero in float32,
    which would leave the head attending to nothing at all.
    """
    return F.softmax(sharpness * w.clamp(min=EPS).log(), dim=1)


def address(memory, previous_w, key, strength, gate, shift_weights, sharpness):
    """All four stages in order: what a head attends to this timestep."""
    w = content_weighting(memory, key, strength)
    w = interpolate(w, previous_w, gate)
    w = shift(w, shift_weights)
    return sharpen(w, sharpness)
