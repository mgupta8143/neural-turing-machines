"""Reading, writing and addressing the NTM's memory matrix.

Memory is a tensor of shape (batch, locations, width). It is state, not weights: the model
carries it from one timestep to the next, so it is passed in and handed back.

Addressing is the paper's Figure 2, one function per stage. Every stage takes a weighting of
shape (batch, locations) and returns another one, non-negative and summing to 1.
"""

import functools

import torch
import torch.nn.functional as F

EPS = 1e-8


def read(memory, w):
    """Weighted average of the memory locations. (B, N, M) and (B, N) -> (B, M).

    Written as a batched matrix multiply rather than multiply-then-sum: one kernel instead of two,
    which matters because this runs once per head per timestep.
    """
    return torch.bmm(w.unsqueeze(1), memory).squeeze(1)


def write(memory, w, erase, add):
    """Erase then add, both scaled by the weighting, so untouched locations (w = 0) keep their value.

    memory * (1 - w erase) + w add, rearranged as memory + w (add - erase memory) so the whole
    update is one fused multiply-add instead of five separate kernels.
    """
    w = w.unsqueeze(-1)  # (B, N, 1), broadcasts across the width of each location
    return torch.addcmul(memory, w, add.unsqueeze(1) - erase.unsqueeze(1) * memory)


def row_norms(memory):
    """The length of every memory location, which `content_weighting` divides by.

    Separate from `content_weighting` because it only changes when the memory does, so all the
    heads looking at one version of the memory can share a single copy.
    """
    return memory.norm(dim=-1).clamp(min=EPS)


def content_weighting(memory, key, strength, norms=None):
    """Stage 1, equations 5 and 6: attend to locations that look like the key.

    Strength turns the cosine similarities into a weighting that can be flat (strength 0) or
    concentrated on the single best match (large strength).

    `norms` is `row_norms(memory)`, passed in when a caller has already computed it.
    """
    # Scaling the key by strength / |key| first means the only work left at the size of the
    # memory is one batched matmul and one divide.
    key = key * (strength / key.norm(dim=-1, keepdim=True).clamp(min=EPS))
    cosine = torch.bmm(memory, key.unsqueeze(-1)).squeeze(-1)  # (B, N), still to be divided by |m|
    return F.softmax(cosine / (row_norms(memory) if norms is None else norms), dim=1)


def interpolate(content_w, previous_w, gate):
    """Stage 2, equation 7: gate 1 takes the content weighting, gate 0 stays where the head was.

    `lerp` is exactly that blend, in one kernel instead of four.
    """
    return torch.lerp(previous_w, content_w, gate)


@functools.lru_cache(maxsize=None)
def sources(locations, span, device):
    """sources[i, k] is the location that sends its weight to location i under the k'th shift.

    Shift k moves attention by k - span // 2, so location i receives from i - (k - span // 2),
    wrapping around the ends of the memory. It only depends on the shape of the memory, so it
    is built once and reused.
    """
    pad = span // 2
    offsets = pad - torch.arange(span, device=device)
    return (torch.arange(locations, device=device).unsqueeze(1) + offsets) % locations


def shift(w, shift_weights):
    """Stage 3, equation 8: a circular convolution that rotates attention to nearby locations.

    shift_weights is a distribution over the allowed moves, e.g. (-1, 0, +1) for three of them,
    and the rotation wraps around the ends of the memory.

    Gathering each location's neighbourhood with a cached index is cheaper than padding and
    unfolding it every time, and the index already has the convolution's flip built in.
    """
    windows = w[:, sources(w.shape[1], shift_weights.shape[1], w.device)]  # (B, N, span)
    return torch.bmm(windows, shift_weights.unsqueeze(-1)).squeeze(-1)


def sharpen(w, sharpness):
    """Stage 4, equation 9: a power of at least 1 makes the weighting peakier, undoing shift blur.

    w ** sharpness, renormalised, is the same as softmax(sharpness * log w), and the softmax
    form is what we use: raising small weights to a high power underflows to zero in float32,
    which would leave the head attending to nothing at all.
    """
    return F.softmax(sharpness * w.clamp(min=EPS).log(), dim=1)


def address(memory, previous_w, key, strength, gate, shift_weights, sharpness, norms=None):
    """All four stages in order: what a head attends to this timestep."""
    w = content_weighting(memory, key, strength, norms)
    w = interpolate(w, previous_w, gate)
    w = shift(w, shift_weights)
    return sharpen(w, sharpness)
