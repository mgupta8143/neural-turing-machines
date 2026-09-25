"""Priority sort task data (paper, Section 4.5).

A sequence of random 8-bit vectors, each with a scalar priority drawn uniformly from [-1, 1]
on its own channel, then a delimiter. The network must emit the 16 highest-priority vectors in
priority order while its input is blank. The paper asks for 16 of the 20 "because we were
interested to see if NTM would solve the task using a binary heap sort of depth 4".

For L vectors, of which K = min(16, L) are asked for, the example has L + 1 + K timesteps:

    t = 0 .. L-1     the bit vectors (channels 0-7), each with its priority (channel 8)
    t = L            the delimiter          (channel 9)
    t = L+1 .. L+K   all zeros: sort time   (the outputs here are scored)

Three things the paper leaves open, and what this module does about them:

  * It never states the vector width for this task, and the parameter counts in Tables 1-3
    cannot be inverted to recover it: they are not even monotonic in the channel count (copy,
    9 input channels, 17,162 parameters; repeat copy, 10 channels on the same controller and
    memory, 16,712). Every width the paper does state is 8 (Section 4.1) or 6 (Section 4.3),
    so this uses 8, as copy and repeat copy do here.
  * "Sorted according to their priorities" does not say which way round. Figure 17 has the read
    head traversing locations in increasing order, but the sign of the priority-to-location fit
    is not given, so that settles nothing either. Highest first, so the first output step is the
    most important vector; nothing in the task is asymmetric, so the other order trains as well.
  * The paper does not mention a delimiter. One is kept anyway, as in the other tasks: it is
    what marks the end of the input phase once the length is not pinned to the paper's 20.

Priorities are continuous, so exact ties have probability zero; the sort is stable regardless.
"""

import random

import torch

BITS = 8
INPUT_SIZE = BITS + 2  # 8 data bits, the priority scalar, and a delimiter channel
OUTPUT_SIZE = BITS
PRIORITY_RANGE = (-1.0, 1.0)  # "the priority is drawn uniformly from the range [-1, 1]"
MIN_LEN, MAX_LEN = 20, 20  # "each input sequence contained 20 binary vectors"
KEEP = 16  # "each target sequence was the 16 highest-priority vectors in the input"


def kept(length):
    """How many vectors the target asks for: the paper's 16, or all of them if there are fewer."""
    return min(KEEP, length)


def timesteps(length):
    return length + 1 + kept(length)


# Every timestep count a batch can have, which is what the CUDA-graph path captures one graph for
TIMESTEPS = tuple(sorted({timesteps(length) for length in range(MIN_LEN, MAX_LEN + 1)}))


def batch(batch_size: int, min_len: int = MIN_LEN, max_len: int = MAX_LEN):
    """Returns x (batch, T, 10), target (batch, T, 8) and mask (batch, T), with T as above.

    L is drawn once per batch. The mask marks the sort steps: the only ones scored.
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
    """The kept vectors on the sort steps, and the mask marking those steps."""
    batch_size, length, _ = vectors.shape
    start = length + 1
    target = torch.zeros(batch_size, timesteps(length), OUTPUT_SIZE)
    target[:, start:] = sort_by_priority(vectors, priorities)
    mask = torch.zeros(batch_size, timesteps(length), dtype=torch.bool)
    mask[:, start:] = True
    return target, mask


def sort_by_priority(vectors, priorities):
    """The kept(L) highest-priority vectors, highest first, ties broken by input order."""
    order = priorities.argsort(dim=1, descending=True, stable=True)[:, :kept(vectors.shape[1])]
    return vectors.gather(1, order.unsqueeze(-1).expand(-1, -1, BITS))


def _case(length):
    def build():
        return batch(1, length, length)
    return build


# Logged during training by src/probe.py. The output is 16 vectors however long the input is, so
# the only axis that can grow is the input: 40 vectors asks the model to rank twice as many
# candidates and still return the same top 16, which is what the paper's write-at-a-location-
# linear-in-priority scheme would have to scale to. Not much further, since memory holds 128.
PROBE_CASES = [("trained", _case(20)), ("long", _case(40))]
