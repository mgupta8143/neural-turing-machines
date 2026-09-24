"""Copy task data (paper, Section 4.1).

Each example is a sequence of random 8-bit vectors followed by a delimiter flag.
The network must then output the same vectors again while its input is blank.

For a sequence of L vectors the example has 2L + 1 timesteps:

    t = 0 .. L-1     the bit vectors        (channels 0-7)
    t = L            the delimiter          (channel 8)
    t = L+1 .. 2L    all zeros: recall time (the outputs here are scored)
"""

import random

import torch

BITS = 8
INPUT_SIZE = BITS + 1  # 8 data bits + 1 delimiter channel
OUTPUT_SIZE = BITS
MIN_LEN, MAX_LEN = 1, 20  # the paper trains on lengths 1 to 20

# Every timestep count a batch can have, which is what the CUDA-graph path captures one graph for
TIMESTEPS = tuple(2 * length + 1 for length in range(MIN_LEN, MAX_LEN + 1))


def batch(batch_size: int, min_len: int = MIN_LEN, max_len: int = MAX_LEN):
    """Returns x (batch, 2L+1, 9), target (batch, 2L+1, 8) and mask (batch, 2L+1).

    L is drawn from [min_len, max_len]. The mask marks the recall steps: the only ones scored.
    """
    length = random.randint(min_len, max_len)
    vectors = torch.randint(0, 2, (batch_size, length, BITS)).float()
    return make_input(vectors), *make_target(vectors)


def make_input(vectors):
    """Builds the model input for vectors of shape (batch, L, 8): the vectors, a delimiter, then blanks."""
    batch_size, length, _ = vectors.shape
    x = torch.zeros(batch_size, 2 * length + 1, INPUT_SIZE)
    x[:, :length, :BITS] = vectors  # the vectors to remember
    x[:, length, BITS] = 1.0  # the delimiter: "now repeat them"
    return x


def make_target(vectors):
    """The vectors again on the recall steps, and the mask marking those steps."""
    batch_size, length, _ = vectors.shape
    target = torch.zeros(batch_size, 2 * length + 1, OUTPUT_SIZE)
    target[:, length + 1:] = vectors
    mask = torch.zeros(batch_size, 2 * length + 1, dtype=torch.bool)
    mask[:, length + 1:] = True
    return target, mask
