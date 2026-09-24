"""Priority sort task data (paper, Section 4.5).

The network reads 20 random binary vectors, each with a scalar priority drawn uniformly from
[-1, 1] on its own input channel, and must then emit the 16 highest-priority vectors in
priority order while its input is blank. The paper limited the sort to 16 of the 20 "because
we were interested to see if NTM would solve the task using a binary heap sort of depth 4".

For L vectors, of which K = min(16, L) are asked for, the example has L + 1 + K timesteps:

    t = 0 .. L-1        the bit vectors        (channels 0-7)
                        their priorities       (channel 8)
    t = L               the delimiter          (channel 9)
    t = L+1 .. L+K      all zeros: the sorted vectors (the outputs here are scored)

Three things the paper leaves open, and what this module does about them:

  * The width of the binary vectors is never stated for this task, and Tables 1-3 cannot be
    inverted to recover it: they give copy (9 input channels) 17,162 parameters but repeat copy
    (10 input channels, same controller and memory) only 16,712, so the counts are not even
    monotonic in the channel count. Every other task in the paper that states a width uses
    eight-bit vectors (Section 4.1) or six (associative recall, Section 4.3), so this uses 8,
    the same as copy and repeat copy here.
  * "Sorted according to their priorities" does not say which way round. Highest first, so that
    the first output step is the most important vector; nothing in the task is asymmetric, so
    the other order would train just as well.
  * The paper does not mention a delimiter. One is kept anyway, as in the other two tasks: it
    is what marks the end of the input phase when the sequence is shorter than the paper's
    fixed 20 (the demo and the figures pin the length).

Priorities are continuous, so exact ties have probability zero; the sort is stable regardless,
so equal priorities keep their input order and a batch is a deterministic function of its draw.
"""

import random

import torch

BITS = 8
INPUT_SIZE = BITS + 2  # 8 data bits, the priority scalar, and a delimiter
OUTPUT_SIZE = BITS
PRIORITY_RANGE = (-1.0, 1.0)  # "the priority is drawn uniformly from the range [-1, 1]"
MIN_LEN, MAX_LEN = 20, 20  # "each input sequence contained 20 binary vectors"
KEEP = 16  # "each target sequence was the 16 highest-priority vectors in the input"


def kept(length: int) -> int:
    """How many vectors the target asks for: the paper's 16, or all of them if there are fewer."""
    return min(KEEP, length)


def timesteps(length: int) -> int:
    """The sequence, the delimiter, and one step per vector the target asks for."""
    return length + 1 + kept(length)


# Every timestep count a batch can have, which is what the CUDA-graph path captures one graph for
TIMESTEPS = tuple(sorted({timesteps(length) for length in range(MIN_LEN, MAX_LEN + 1)}))


def batch(batch_size: int, min_len: int = MIN_LEN, max_len: int = MAX_LEN):
    """Returns x (batch, T, 10), target (batch, T, 8) and mask (batch, T), with T as above.

    L is drawn once per batch, so every sequence in it has the same shape. The mask marks the
    output steps: the only ones scored.
    """
    length = random.randint(min_len, max_len)
    vectors = torch.randint(0, 2, (batch_size, length, BITS)).float()
    low, high = PRIORITY_RANGE
    priorities = torch.rand(batch_size, length) * (high - low) + low
    return make_input(vectors, priorities), *make_target(vectors, priorities)


def make_input(vectors, priorities):
    """The vectors and their priorities, a delimiter, then blanks for the output phase."""
    batch_size, length, _ = vectors.shape
    x = torch.zeros(batch_size, timesteps(length), INPUT_SIZE)
    x[:, :length, :BITS] = vectors  # the vectors to sort
    x[:, :length, BITS] = priorities  # each one's priority, alongside it
    x[:, length, BITS + 1] = 1.0  # the delimiter: "now give them back in order"
    return x


def make_target(vectors, priorities):
    """The K highest-priority vectors, highest first, and the mask marking those steps."""
    batch_size, length, _ = vectors.shape
    start = length + 1  # the first output step, right after the delimiter

    target = torch.zeros(batch_size, timesteps(length), OUTPUT_SIZE)
    target[:, start:] = sort_by_priority(vectors, priorities)
    mask = torch.zeros(batch_size, timesteps(length), dtype=torch.bool)
    mask[:, start:] = True
    return target, mask


def sort_by_priority(vectors, priorities):
    """The kept(L) vectors with the highest priorities, in descending priority order.

    A stable sort, so vectors that somehow share a priority come out in input order.
    """
    order = priorities.argsort(dim=1, descending=True, stable=True)[:, :kept(vectors.shape[1])]
    return vectors.gather(1, order.unsqueeze(-1).expand(-1, -1, BITS))
