"""Copy task data.

Each example is a sequence of random 8-bit vectors followed by a delimiter flag.
The network must then output the same vectors again while its input is blank.

For a sequence of L vectors the input has 2L + 1 timesteps:

    t = 0 .. L-1     the bit vectors        (channels 0-7)
    t = L            the delimiter          (channel 8)
    t = L+1 .. 2L    all zeros: recall time (the outputs here are scored)
"""

import random

import torch

BITS = 8
INPUT_SIZE = BITS + 1  # 8 data bits + 1 delimiter channel


def copy_batch(batch_size: int, min_len: int = 1, max_len: int = 20):
    """Returns x: (batch, 2L + 1, 9) and target: (batch, L, 8), with L drawn from [min_len, max_len]."""
    length = random.randint(min_len, max_len)
    target = torch.randint(0, 2, (batch_size, length, BITS)).float()

    x = torch.zeros(batch_size, 2 * length + 1, INPUT_SIZE)
    x[:, :length, :BITS] = target  # the vectors to remember
    x[:, length, BITS] = 1.0  # the delimiter: "now repeat them"
    return x, target
