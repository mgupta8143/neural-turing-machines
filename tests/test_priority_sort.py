"""Checks for the priority sort task's data (paper, Section 4.5).

The interesting claim is the target: the 16 highest-priority of the 20 input vectors, highest
priority first. Every check here reads the priorities back out of x and redoes the sort in
plain Python, so the module is compared against the definition rather than against itself.
"""

import torch

from src.tasks.priority_sort import data

PRIORITY, DELIMITER = data.BITS, data.BITS + 1  # input channel indices


def expected_target(x, length):
    """The top-K vectors sorted by descending priority, worked out row by row in Python."""
    rows = []
    for sequence in x:
        ranked = sorted(
            range(length),
            key=lambda t: (-sequence[t, PRIORITY].item(), t),  # ties broken by input order
        )
        rows.append(torch.stack([sequence[t, :data.BITS] for t in ranked[:data.kept(length)]]))
    return torch.stack(rows)


def test_shapes_match_the_declared_channel_counts():
    x, target, mask = data.batch(4)
    length = data.MAX_LEN  # the paper's fixed 20 vectors in, 16 out

    assert x.shape == (4, data.timesteps(length), data.INPUT_SIZE) == (4, 37, 10)
    assert target.shape == (4, data.timesteps(length), data.OUTPUT_SIZE) == (4, 37, 8)
    assert mask.shape == (4, 37)


def test_timesteps_lists_what_batch_actually_produces():
    assert data.TIMESTEPS == tuple(sorted(set(data.TIMESTEPS)))  # one CUDA graph per distinct count
    for _ in range(10):
        x, target, mask = data.batch(2)
        assert x.shape[1] in data.TIMESTEPS
        assert target.shape[1] == mask.shape[1] == x.shape[1]


def test_the_input_is_the_vectors_with_their_priorities_then_a_delimiter_then_nothing():
    x, _, _ = data.batch(3)
    length = data.MAX_LEN

    assert set(x[:, :length, :data.BITS].flatten().tolist()) <= {0.0, 1.0}  # random bits
    assert not x[:, :length, DELIMITER].any()  # the flag stays off while the sequence comes in
    assert x[:, length, DELIMITER].tolist() == [1.0, 1.0, 1.0]
    assert x[:, length, :DELIMITER].sum() == 0  # the delimiter step carries nothing else
    assert x[:, length + 1:, :].sum() == 0  # no input at all during the output phase


def test_the_priorities_are_uniform_over_the_paper_s_range():
    low, high = data.PRIORITY_RANGE
    priorities = data.batch(256)[0][:, :data.MAX_LEN, PRIORITY]

    assert priorities.min() >= low and priorities.max() <= high
    assert priorities.min() < low / 2 and priorities.max() > high / 2  # both ends are reached
    assert abs(priorities.mean().item() - (low + high) / 2) < 0.05


def test_the_target_is_the_top_16_vectors_in_descending_priority_order():
    x, target, _ = data.batch(8)
    length = data.MAX_LEN

    outputs = target[:, length + 1:]
    assert outputs.shape[1] == data.kept(length) == 16  # 16 of the 20 go out
    assert torch.equal(outputs, expected_target(x, length))
    assert not target[:, :length + 1].any()  # nothing is asked for before the output phase


def test_the_target_holds_only_the_four_lowest_priorities_back():
    x, target, _ = data.batch(4)
    length = data.MAX_LEN

    for sequence, outputs in zip(x, target[:, length + 1:]):
        priorities = sequence[:length, PRIORITY]
        vectors = sequence[:length, :data.BITS]
        chosen = sorted(range(length), key=lambda t: -priorities[t].item())
        kept, dropped = chosen[:16], chosen[16:]

        assert priorities[kept].min() > priorities[dropped].max()  # the cut is by priority
        for step, t in enumerate(kept):
            assert torch.equal(outputs[step], vectors[t])  # and the order is the sorted one


def test_ties_keep_their_input_order():
    vectors = torch.tensor([[[1.0] * 8, [0.0] * 8, [1.0, 0.0] * 4]])
    priorities = torch.tensor([[0.5, 0.5, 0.9]])

    ordered = data.sort_by_priority(vectors, priorities)

    # 0.9 first, then the two tied at 0.5 in the order they came in: a stable sort, so this is
    # the same on every call and on every platform.
    assert torch.equal(ordered, vectors[:, [2, 0, 1]])
    assert torch.equal(data.sort_by_priority(vectors, priorities), ordered)


def test_the_mask_scores_the_output_phase_and_nothing_before_it():
    _, _, mask = data.batch(3)
    length = data.MAX_LEN

    assert not mask[:, :length + 1].any()  # the sequence and the delimiter are not scored
    assert mask[:, length + 1:].all()  # every output step is
    assert int(mask[0].sum()) == data.kept(length) == 16


def test_a_shorter_sequence_than_the_paper_s_sorts_all_of_it():
    """main.py's demo pins the length; with fewer than 16 vectors the target is all of them."""
    x, target, mask = data.batch(2, 3, 3)

    assert x.shape == (2, data.timesteps(3), data.INPUT_SIZE) == (2, 7, 10)
    assert int(mask[0].sum()) == data.kept(3) == 3
    assert torch.equal(target[:, 4:], expected_target(x, 3))
