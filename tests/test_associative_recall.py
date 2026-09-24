"""Checks for the associative recall task's data (paper, Section 4.3).

An episode is a list of items, a query, and the answer, so the checks read the episode back out
of x: the item delimiters say where the items are, the query delimiter says where the query is,
and the target has to be whichever item followed the one repeated as the query.
"""

import torch

from src.tasks.associative_recall import data

BITS, STEPS = data.BITS, data.ITEM_STEPS
ITEM_DELIMITER, QUERY_DELIMITER = data.ITEM_DELIMITER, data.QUERY_DELIMITER


def items_of(x):
    """The items shown in the list phase of one episode, as a tensor (items, 3, 6)."""
    query_step = int(x[:, QUERY_DELIMITER].nonzero().item())
    starts = range(1, query_step - 1, STEPS + 1)  # each item starts one step after its delimiter
    return torch.stack([x[start:start + STEPS, :BITS] for start in starts])


def query_of(x):
    """The query item of one episode, as a tensor (3, 6): the three steps after its delimiter."""
    query_step = int(x[:, QUERY_DELIMITER].nonzero().item())
    return x[query_step + 1:query_step + 1 + STEPS, :BITS]


def queried_indices(x):
    """Which items of one episode the query could be. Random items can repeat, so this is a list."""
    query = query_of(x)
    return [index for index, shown in enumerate(items_of(x)) if torch.equal(shown, query)]


def test_shapes_match_the_declared_channel_counts():
    x, target, mask = data.batch(4, 3, 3)
    timesteps = data.timesteps(items=3)  # 3 * 4 + 1 delimited items, + 5 query steps, + 3 answer

    assert x.shape == (4, timesteps, data.INPUT_SIZE) == (4, 21, 8)
    assert target.shape == (4, timesteps, data.OUTPUT_SIZE) == (4, 21, 6)
    assert mask.shape == (4, timesteps)
    assert (data.INPUT_SIZE, data.OUTPUT_SIZE) == (8, 6)  # 6 bits + 2 delimiters in, 6 bits out


def test_the_input_is_delimited_items_then_a_delimited_query():
    x, _, _ = data.batch(2, 4, 4)

    # Item delimiters bound every item on both sides: four items need five, plus one closing the
    # query item. The query delimiter sits between the list's closing delimiter and the query.
    assert x[0, :, ITEM_DELIMITER].nonzero().flatten().tolist() == [0, 4, 8, 12, 16, 21]
    assert x[0, :, QUERY_DELIMITER].nonzero().flatten().tolist() == [17]

    delimiters = x[:, :, BITS:].sum(dim=2)  # a delimiter step never carries data bits
    assert (x[:, :, :BITS].sum(dim=2)[delimiters > 0] == 0).all()
    assert (delimiters <= 1).all() and int(delimiters[0].sum()) == 7  # never two flags at once
    assert set(x[:, :, :BITS].flatten().tolist()) <= {0.0, 1.0}  # the data is random bits
    assert x[:, -STEPS:, :].sum() == 0  # nothing at all is fed in during the answer


def test_the_query_is_one_of_the_items_shown_and_never_the_last():
    for _ in range(50):
        x, _, _ = data.batch(1, 2, 6)
        matches = queried_indices(x[0])

        assert matches, "the query is not one of the items that were presented"
        assert min(matches) < len(items_of(x[0])) - 1  # something must follow it to be recalled


def test_the_target_is_the_item_that_followed_the_query():
    for _ in range(50):
        x, target, _ = data.batch(1, 2, 6)
        items = items_of(x[0])

        followers = [items[index + 1] for index in queried_indices(x[0]) if index + 1 < len(items)]
        assert any(torch.equal(target[0, -STEPS:], follower) for follower in followers)
        assert target[0, :-STEPS].sum() == 0  # nothing is asked for before the answer


def test_every_episode_in_a_batch_queries_independently():
    x, _, _ = data.batch(256, 6, 6)
    queried = {min(queried_indices(episode)) for episode in x}

    assert queried == {0, 1, 2, 3, 4}  # all five queryable items turn up, the sixth never


def test_the_mask_scores_the_three_answer_steps_and_nothing_else():
    for items in range(data.MIN_ITEMS, data.MAX_ITEMS + 1):
        _, _, mask = data.batch(2, items, items)

        assert mask[:, -STEPS:].all()
        assert not mask[:, :-STEPS].any()
        assert int(mask[0].sum()) == STEPS == 3


def test_TIMESTEPS_covers_every_length_batch_can_produce():
    assert data.TIMESTEPS == tuple(sorted(set(data.TIMESTEPS)))  # one CUDA graph per distinct count

    lengths = set()
    for _ in range(200):
        x, target, mask = data.batch(1)
        assert x.shape[1] == target.shape[1] == mask.shape[1]
        lengths.add(x.shape[1])

    assert lengths == set(data.TIMESTEPS)  # every declared count shows up, and no other
    assert len(data.TIMESTEPS) == data.MAX_ITEMS - data.MIN_ITEMS + 1 == 5


def test_random_draws_stay_inside_the_paper_s_range():
    for _ in range(50):
        x, _, _ = data.batch(1)
        items = len(items_of(x[0]))

        assert data.MIN_ITEMS <= items <= data.MAX_ITEMS == 6
        assert x.shape[1] == data.timesteps(items)
