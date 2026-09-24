"""Associative recall task data (paper, Section 4.3).

A test of indirection: one data item has to point at another. An item is three consecutive
six-bit binary vectors, and an episode shows between 2 and 6 of them in a row. Afterwards one
of the items is shown again as a query, and the network must produce the item that FOLLOWED
the queried one when the list went by. Only those three answer steps are scored.

The paper: "we define an item as a sequence of binary vectors that is bounded on the left and
right by delimiter symbols. After several items have been propagated to the network, we query
by showing a random item, and we ask the network to produce the next item."

Channels (the paper's Figure 12 shows the same layout, 1-indexed, as rows of "Inputs"):

    0-5   the six data bits of the vector at this step     (paper rows 1-6)
    6     item delimiter: a bound between items            (paper row 7)
    7     query delimiter: "the next item is the query"    (paper row 8)

Both delimiters get a timestep of their own, with the data bits blank, which is how Figure 12
draws them: "the input denotes item delimiters as single bits in row 7 ... a delimiter in row 8
prepares the network to receive a query item". The output has only the six data bits: nothing
else is ever asked for.

An item is bounded on *both* sides, and neighbouring items share the delimiter between them, so
N items need N + 1 item delimiters. For 3 items the episode is 21 steps:

    t = 0                 item delimiter                        (channel 6)
    t = 1, 2, 3           item 1                                (channels 0-5)
    t = 4                 item delimiter
    t = 5, 6, 7           item 2
    t = 8                 item delimiter
    t = 9, 10, 11         item 3
    t = 12                item delimiter: the list is over
    t = 13                query delimiter                       (channel 7)
    t = 14, 15, 16        the query item, one of items 1 and 2
    t = 17                item delimiter: the query is over
    t = 18, 19, 20        all zeros: the answer (the only steps scored)

The query is never the last item, because then there would be no following item to recall.

Ambiguity we had to resolve: the paper says an item is bounded left and right by delimiters but
never counts them, so a run of N items could carry anywhere from N to N + 1 item delimiters
depending on whether the first and last are closed off. We take "bounded on the left and right"
literally, which gives N + 1, and close the query item the same way. That reading also matches
the algorithm the paper reads off the memory traces: "when each item delimiter is presented, the
controller writes a compressed representation of the previous three time slices of the item ...
After the query arrives, the controller recomputes the same compressed representation of the
query item" - both of which need a delimiter step *after* an item to happen on. The looser
reading, one delimiter before each item and none after, would drop the two closing steps and
leave the network to do that work on the query delimiter instead.
"""

import random

import torch

BITS = 6  # a vector is six bits (paper: "three six-bit binary vectors, giving a total of 18 bits")
ITEM_STEPS = 3  # ... and an item is three of those vectors, on three consecutive steps
INPUT_SIZE = BITS + 2  # 6 data bits, the item delimiter, the query delimiter
OUTPUT_SIZE = BITS
ITEM_DELIMITER, QUERY_DELIMITER = BITS, BITS + 1  # input channel indices
MIN_ITEMS, MAX_ITEMS = 2, 6  # the paper trains on 2 to 6 items per episode


def timesteps(items: int) -> int:
    """A delimiter plus three steps per item, a closing delimiter, the query block, the answer."""
    list_steps = items * (ITEM_STEPS + 1) + 1  # N delimiter-and-item blocks, then a closing one
    query_steps = 1 + ITEM_STEPS + 1  # the query delimiter, the query item, its closing delimiter
    return list_steps + query_steps + ITEM_STEPS


# Every timestep count a batch can have, which is what the CUDA-graph path captures one graph for
TIMESTEPS = tuple(timesteps(items) for items in range(MIN_ITEMS, MAX_ITEMS + 1))


def batch(batch_size: int, min_len: int = MIN_ITEMS, max_len: int = MAX_ITEMS):
    """Returns x (batch, T, 8), target (batch, T, 6) and mask (batch, T).

    The item count is drawn once per batch, so every episode in it has the same shape, but each
    episode queries an item of its own. The mask marks the three answer steps: the only ones
    scored.
    """
    items = random.randint(min_len, max_len)
    vectors = torch.randint(0, 2, (batch_size, items, ITEM_STEPS, BITS)).float()
    # Anything but the last item, which has nothing following it to recall
    queried = torch.randint(0, items - 1, (batch_size,))
    return make_input(vectors, queried), *make_target(vectors, queried)


def item(vectors, queried):
    """Picks one item per episode out of vectors (batch, items, 3, 6), by index (batch,)."""
    return vectors[torch.arange(vectors.shape[0]), queried]


def make_input(vectors, queried):
    """The items with delimiters between and around them, the query delimiter, then the query."""
    batch_size, items, _, _ = vectors.shape
    x = torch.zeros(batch_size, timesteps(items), INPUT_SIZE)

    for index in range(items):  # the list, each item preceded by a delimiter
        start = index * (ITEM_STEPS + 1)
        x[:, start, ITEM_DELIMITER] = 1.0
        x[:, start + 1:start + 1 + ITEM_STEPS, :BITS] = vectors[:, index]

    closing = items * (ITEM_STEPS + 1)  # the delimiter that bounds the last item on the right
    x[:, closing, ITEM_DELIMITER] = 1.0
    x[:, closing + 1, QUERY_DELIMITER] = 1.0  # "what follows is the query"
    x[:, closing + 2:closing + 2 + ITEM_STEPS, :BITS] = item(vectors, queried)
    x[:, closing + 2 + ITEM_STEPS, ITEM_DELIMITER] = 1.0  # the query is bounded too
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
