"""Dynamic N-grams task data (paper, Section 4.4).

A prediction task rather than a copy task, so it has a different shape from the others: the
network sees a binary sequence one bit at a time and has to predict the next bit at almost
every step, instead of reading a sequence in and echoing it back in a separate output phase.

Each example is drawn from a fresh 6-gram source. A 6-gram over binary sequences is a table of
2^5 = 32 numbers, one per five-bit history, giving the probability that the next bit is one; the
paper draws all 32 independently from Beta(1/2, 1/2), a U-shaped prior that mostly produces
near-certain contexts. Then 200 successive bits are sampled from that table. A different table
every example is what makes the task "dynamic": there is nothing to learn about any one source,
only about counting the statistics of whichever one is on show.

For a sequence of 200 bits the example has 200 timesteps, one channel in and one channel out:

    t = 0            input 0 (there is no previous bit yet), target bit 0
    t = 1 .. 199     input bit t-1, target bit t

so at every step the network's input is the bit it has just been scored on. The first five bits
are drawn i.i.d. from Bernoulli(0.5), because there is not enough history to index the table
with (the paper's footnote 4), and those five steps are left out of the mask: nothing could
predict them better than chance.
"""

import random

import torch

CONTEXT = 5  # bits of history the source conditions on: a 6-gram
TABLE_SIZE = 2**CONTEXT  # 32 contexts, one probability each
LENGTH = 200  # "drawing 200 successive bits using the current lookup table"

INPUT_SIZE = 1  # the previous bit
OUTPUT_SIZE = 1  # the probability that the current bit is one

# Every timestep count a batch can have, which is what the CUDA-graph path captures one graph for
TIMESTEPS = (LENGTH,)

# A context is read oldest bit first, so the history b[t-5]..b[t-1] is the table index that the
# paper would write as "00010". Any convention works as long as the sampler and the optimal
# estimator below share it.
PLACE_VALUES = 2 ** torch.arange(CONTEXT - 1, -1, -1)


def batch(batch_size: int, min_len: int = LENGTH, max_len: int = LENGTH):
    """Returns x (batch, T, 1), target (batch, T, 1) and mask (batch, T), with T = 200 bits.

    Every sequence in the batch comes from its own table. min_len and max_len let the demo and
    the figures ask for a shorter sequence; below CONTEXT + 1 bits there is nothing to score,
    so that is the floor.
    """
    length = max(random.randint(min_len, max_len), CONTEXT + 1)
    bits = sample_bits(sample_tables(batch_size), length)
    return make_input(bits), *make_target(bits)


def sample_tables(batch_size: int):
    """One 6-gram per example: 32 probabilities drawn independently from Beta(1/2, 1/2)."""
    return torch.distributions.Beta(0.5, 0.5).sample((batch_size, TABLE_SIZE))


def sample_bits(tables, length: int = LENGTH):
    """Draws (batch, length) bits, each from its row of `tables` given the five bits before it."""
    batch_size = tables.shape[0]
    bits = torch.zeros(batch_size, length)
    bits[:, :CONTEXT] = torch.randint(0, 2, (batch_size, CONTEXT)).float()  # no context yet: p = 0.5
    for t in range(CONTEXT, length):
        probability = tables.gather(1, context_index(bits[:, t - CONTEXT:t]).unsqueeze(1))
        bits[:, t] = torch.bernoulli(probability.squeeze(1))
    return bits


def context_index(window):
    """Turns a (batch, 5) window of bits into the table index it stands for, oldest bit first."""
    return (window.long() * PLACE_VALUES).sum(-1)


def make_input(bits):
    """The sequence delayed by one step, so the network has to guess bit t before it arrives."""
    x = torch.zeros(*bits.shape, INPUT_SIZE)
    x[:, 1:, 0] = bits[:, :-1]
    return x


def make_target(bits):
    """The sequence itself, and the mask hiding the first five steps, which are context-free."""
    mask = torch.ones(bits.shape, dtype=torch.bool)
    mask[:, :CONTEXT] = False
    return bits.unsqueeze(-1), mask


def optimal_predictions(bits):
    """The Bayesian estimator the paper compares against, as (batch, length) probabilities.

    Paper Equation 10: having seen the current five-bit context c followed by N1 ones and N0
    zeros so far in this sequence, the probability that the next bit is one is

        (N1 + 1/2) / (N1 + N0 + 1)

    which is the posterior mean under the Beta(1/2, 1/2) prior the tables were drawn from. It
    knows the prior but not the table, so it is the best any predictor can do here, and it gives
    the learning curves their reference line (paper, Figure 13). The first five steps get 0.5;
    they are masked out anyway.
    """
    batch_size, length = bits.shape
    rows = torch.arange(batch_size)
    counts = torch.zeros(batch_size, TABLE_SIZE, 2)  # zeros and ones seen after each context
    predictions = torch.full((batch_size, length), 0.5)
    for t in range(CONTEXT, length):
        context = context_index(bits[:, t - CONTEXT:t])
        seen = counts[rows, context]  # (batch, 2), the N0 and N1 of Equation 10
        predictions[:, t] = (seen[:, 1] + 0.5) / (seen.sum(-1) + 1)
        counts[rows, context, bits[:, t].long()] += 1  # then the bit itself becomes evidence
    return predictions


def cost_in_bits(predictions, bits, mask):
    """The paper's y axis: cross-entropy of those predictions over the scored steps, per sequence."""
    probability = predictions.clamp(1e-6, 1 - 1e-6)
    per_step = -(bits * probability.log2() + (1 - bits) * (1 - probability).log2())
    return (per_step * mask).sum(-1)
