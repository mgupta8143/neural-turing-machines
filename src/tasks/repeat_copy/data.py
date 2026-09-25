"""Repeat copy (paper, Section 4.2): copy with a "for loop" around it.

For L vectors and R repeats an example runs L + 2 + R*L + 1 timesteps:

    0 .. L-1        the 8-bit vectors                 (channels 0-7)
    L               the delimiter                     (channel 8)
    L+1             the repeat count, normalised      (channel 9)
    L+2 ..          zeros; the R copies, then one more step for the end marker

Everything from L+2 on is scored, the end-marker channel included: it has to stay off during
the copies and come on once, which is what forces the network to keep count.
"""

import random

import torch

BITS = 8
INPUT_SIZE = BITS + 2  # bits, delimiter, repeat count
OUTPUT_SIZE = BITS + 1  # bits, end marker
MIN_LEN, MAX_LEN = 1, 10
MIN_REPEATS, MAX_REPEATS = 1, 10

# The paper normalises the count to mean zero and variance one; uniform on 1..10 gives 5.5 and
# 8.25, so the scalar runs from -1.57 to +1.57.
REPEAT_MEAN = (MIN_REPEATS + MAX_REPEATS) / 2
REPEAT_STD = (((MAX_REPEATS - MIN_REPEATS + 1) ** 2 - 1) / 12) ** 0.5


def timesteps(length: int, repeats: int) -> int:
    return length + 2 + repeats * length + 1


# Every length a batch can have, which the CUDA-graph path captures one graph apiece for
TIMESTEPS = tuple(sorted({
    timesteps(length, repeats)
    for length in range(MIN_LEN, MAX_LEN + 1)
    for repeats in range(MIN_REPEATS, MAX_REPEATS + 1)
}))


def make_input(vectors, repeats: int):
    batch_size, length, _ = vectors.shape
    x = torch.zeros(batch_size, timesteps(length, repeats), INPUT_SIZE)
    x[:, :length, :BITS] = vectors
    x[:, length, BITS] = 1.0
    x[:, length + 1, BITS + 1] = (repeats - REPEAT_MEAN) / REPEAT_STD
    return x


def make_target(vectors, repeats: int):
    batch_size, length, _ = vectors.shape
    start = length + 2
    target = torch.zeros(batch_size, timesteps(length, repeats), OUTPUT_SIZE)
    target[:, start:start + repeats * length, :BITS] = vectors.repeat(1, repeats, 1)
    target[:, -1, BITS] = 1.0
    mask = torch.zeros(batch_size, target.shape[1], dtype=torch.bool)
    mask[:, start:] = True
    return target, mask


def batch(batch_size: int, min_len: int = MIN_LEN, max_len: int = MAX_LEN,
          min_repeats: int = MIN_REPEATS, max_repeats: int = MAX_REPEATS):
    """x (batch, T, 10), target (batch, T, 9), mask (batch, T).

    L and R are drawn once per batch, so every sequence in it has the same shape.
    """
    length = random.randint(min_len, max_len)
    repeats = random.randint(min_repeats, max_repeats)
    vectors = torch.randint(0, 2, (batch_size, length, BITS)).float()
    return make_input(vectors, repeats), *make_target(vectors, repeats)


def _case(length, repeats):
    def build():
        vectors = torch.randint(0, 2, (1, length, BITS)).float()
        return make_input(vectors, repeats), *make_target(vectors, repeats)
    return build


# Logged during training by src/probe.py: the trained corner, then each axis doubled.
PROBE_CASES = [
    ("trained", _case(10, 10)),
    ("long", _case(20, 10)),
    ("many", _case(10, 20)),
]
