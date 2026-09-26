"""Associative recall task data (paper, Section 4.3).

A test of indirection: one data item points at another. An item is three consecutive six-bit
binary vectors, an episode shows between 2 and 6 of them, and then one of them is shown again
as a query. The network must produce the item that FOLLOWED the queried one, and only those
three answer steps are scored.

Channels, laid out as the rows of "Inputs" in the paper's Figure 12 (which counts from 1):

    0-5   the six data bits of the vector at this step     (rows 1-6)
    6     item delimiter, bounding an item left and right  (row 7)
    7     query delimiter: "what follows is the query"     (row 8)

Delimiters get a step of their own with the data bits blank. For 3 items that is 21 steps:

    t = 0                 item delimiter
    t = 1, 2, 3           item 1
    t = 4                 item delimiter
    t = 5, 6, 7           item 2
    t = 8                 item delimiter
    t = 9, 10, 11         item 3
    t = 12                item delimiter: the list is over
    t = 13                query delimiter
    t = 14, 15, 16        the query item, one of items 1 and 2
    t = 17                item delimiter: the query is over
    t = 18, 19, 20        blank input: the answer, and the only steps scored

The query is never the last item, which would have nothing following it to recall.

The paper never counts the delimiters, so N items could carry N or N + 1 of them. We read
"bounded on the left and right" literally and use N + 1, closing the query item the same way,
because the algorithm the paper reads off the memory traces needs a step *after* an item to
run on: "when each item delimiter is presented, the controller writes a compressed
representation of the previous three time slices of the item ... After the query arrives, the
controller recomputes the same compressed representation of the query item".
"""

import random

import torch

BITS = 6  # paper: "three six-bit binary vectors (giving a total of 18 bits per item)"
ITEM_STEPS = 3
INPUT_SIZE = BITS + 2  # the data bits, the item delimiter, the query delimiter
OUTPUT_SIZE = BITS
ITEM_DELIMITER, QUERY_DELIMITER = BITS, BITS + 1
MIN_ITEMS, MAX_ITEMS = 2, 6  # the paper trains on 2 to 6 items per episode


def timesteps(items: int) -> int:
    query_steps = 1 + ITEM_STEPS + 1  # the query delimiter, the query item, its closing delimiter
    return items * (ITEM_STEPS + 1) + 1 + query_steps + ITEM_STEPS


# Every timestep count a batch can have, which is what the CUDA-graph path captures one graph for
TIMESTEPS = tuple(timesteps(items) for items in range(MIN_ITEMS, MAX_ITEMS + 1))


def batch(batch_size: int, min_len: int = MIN_ITEMS, max_len: int = MAX_ITEMS):
    """Returns x (batch, T, 8), target (batch, T, 6) and mask (batch, T).

    The item count is drawn once so the batch has one shape, but each episode queries its own
    item. The mask marks the three answer steps: the only ones scored.
    """
    items = random.randint(min_len, max_len)
    vectors = torch.randint(0, 2, (batch_size, items, ITEM_STEPS, BITS)).float()
    queried = torch.randint(0, items - 1, (batch_size,))  # never the last: nothing follows it
    return make_input(vectors, queried), *make_target(vectors, queried)


def item(vectors, queried):
    """Picks one item per episode out of vectors (batch, items, 3, 6), by index (batch,)."""
    return vectors[torch.arange(vectors.shape[0]), queried]


def make_input(vectors, queried):
    """The items with delimiters between and around them, the query delimiter, then the query."""
    batch_size, items, _, _ = vectors.shape
    x = torch.zeros(batch_size, timesteps(items), INPUT_SIZE)

    for index in range(items):
        start = index * (ITEM_STEPS + 1)
        x[:, start, ITEM_DELIMITER] = 1.0
        x[:, start + 1:start + 1 + ITEM_STEPS, :BITS] = vectors[:, index]

    closing = items * (ITEM_STEPS + 1)  # bounds the last item on the right
    x[:, closing, ITEM_DELIMITER] = 1.0
    x[:, closing + 1, QUERY_DELIMITER] = 1.0
    x[:, closing + 2:closing + 2 + ITEM_STEPS, :BITS] = item(vectors, queried)
    x[:, closing + 2 + ITEM_STEPS, ITEM_DELIMITER] = 1.0
    return x


def make_target(vectors, queried):
    """The item after the queried one, on the last three steps, and the mask marking them."""
    batch_size, items, _, _ = vectors.shape
    total = timesteps(items)

    target = torch.zeros(batch_size, total, OUTPUT_SIZE)
    target[:, total - ITEM_STEPS:] = item(vectors, queried + 1)
    mask = torch.zeros(batch_size, total, dtype=torch.bool)
    mask[:, total - ITEM_STEPS:] = True
    return target, mask


def _case(items):
    def build():
        return batch(1, items, items)
    return build


# Logged during training by src/probe.py. The axis is the item count, which is what the paper's
# Figure 11 measures: the trained maximum, then twice it, where a feedforward NTM should still be
# "nearly perfect".
PROBE_CASES = [("trained", _case(MAX_ITEMS)), ("long", _case(2 * MAX_ITEMS))]
