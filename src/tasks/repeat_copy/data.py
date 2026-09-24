"""Repeat copy task data (paper, Section 4.2).

Copy with a "for loop" around it: the network reads a sequence of random 8-bit vectors and a
scalar saying how many copies to make, then has to output the sequence that many times and
raise an end-of-sequence marker on an extra output channel. As in the copy task, nothing is
fed in during the output phase, so the repeat count has to be both interpreted and counted
down from memory.

Both the sequence length L and the repeat count R are drawn uniformly from 1 to 10.

For L vectors and R repeats the example has L + 2 + R*L + 1 timesteps:

    t = 0 .. L-1                the bit vectors                  (channels 0-7)
    t = L                       the delimiter                    (channel 8)
    t = L+1                     the repeat count, normalised     (channel 9)
    t = L+2 .. L+1+R*L          all zeros: the R copies (scored)
    t = L+2+R*L                 all zeros: the end marker step (scored)

The paper says only that the count "appears on a separate input channel" after the sequence;
giving it a timestep of its own, after the delimiter, keeps each channel's meaning to one step.
"""

import random

import torch

BITS = 8
INPUT_SIZE = BITS + 2  # 8 data bits, a delimiter, and the repeat count
OUTPUT_SIZE = BITS + 1  # 8 data bits and the end-of-sequence marker
MIN_LEN, MAX_LEN = 1, 10
MIN_REPEATS, MAX_REPEATS = 1, 10

# The paper normalises the repeat count to mean zero and variance one. Drawn uniformly from the
# integers 1..10 it has mean (1 + 10) / 2 = 5.5 and variance (10^2 - 1) / 12 = 8.25, so the
# scalar fed in is (R - 5.5) / sqrt(8.25), which runs from -1.57 to +1.57.
REPEAT_MEAN = (MIN_REPEATS + MAX_REPEATS) / 2
REPEAT_STD = ((MAX_REPEATS - MIN_REPEATS + 1) ** 2 - 1) ** 0.5 / 12**0.5


def timesteps(length: int, repeats: int) -> int:
    """The sequence, the delimiter, the repeat count, the copies, and one step for the end marker."""
    return length + 2 + repeats * length + 1


# Every timestep count a batch can have, which is what the CUDA-graph path captures one graph for
TIMESTEPS = tuple(sorted({
    timesteps(length, repeats)
    for length in range(MIN_LEN, MAX_LEN + 1)
    for repeats in range(MIN_REPEATS, MAX_REPEATS + 1)
}))


def batch(batch_size: int, min_len: int = MIN_LEN, max_len: int = MAX_LEN,
          min_repeats: int = MIN_REPEATS, max_repeats: int = MAX_REPEATS):
    """Returns x (batch, T, 10), target (batch, T, 9) and mask (batch, T), with T as above.

    L and R are drawn once per batch, so every sequence in it has the same shape. The mask
    marks the output phase and the end marker step: the only ones scored.
    """
    length = random.randint(min_len, max_len)
    repeats = random.randint(min_repeats, max_repeats)
    vectors = torch.randint(0, 2, (batch_size, length, BITS)).float()
    return make_input(vectors, repeats), *make_target(vectors, repeats)


def make_input(vectors, repeats: int):
    """The vectors, the delimiter, the normalised repeat count, then blanks for the output phase."""
    batch_size, length, _ = vectors.shape
    x = torch.zeros(batch_size, timesteps(length, repeats), INPUT_SIZE)
    x[:, :length, :BITS] = vectors  # the vectors to remember
    x[:, length, BITS] = 1.0  # the delimiter: the sequence is over
    x[:, length + 1, BITS + 1] = (repeats - REPEAT_MEAN) / REPEAT_STD  # "make this many copies"
    return x


def make_target(vectors, repeats: int):
    """The vectors repeated R times, then the end marker, and the mask marking those steps."""
    batch_size, length, _ = vectors.shape
    total = timesteps(length, repeats)
    start = length + 2  # the first output step, right after the repeat count

    target = torch.zeros(batch_size, total, OUTPUT_SIZE)
    target[:, start:start + repeats * length, :BITS] = vectors.repeat(1, repeats, 1)
    target[:, -1, BITS] = 1.0  # the end marker, one step after the last copy

    # The marker channel is scored during the copies too, where it must stay off: that is what
    # makes the network keep count rather than end wherever it likes.
    mask = torch.zeros(batch_size, total, dtype=torch.bool)
    mask[:, start:] = True
    return target, mask
