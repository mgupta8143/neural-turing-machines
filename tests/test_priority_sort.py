"""Checks for the priority sort task's data (paper, Section 4.5).

The encoding puts each piece in one place: the bits, the priority scalar beside them, the
delimiter, and the top-16 sorted output. Every check fixes L and looks at one of those.
"""

import torch

from src.tasks.priority_sort import data

PRIORITY, DELIMITER = data.BITS, data.BITS + 1  # input channel indices


def fixed_batch(batch_size=4, length=20):
    return data.batch(batch_size, length, length)


def test_shapes_match_the_declared_channel_counts():
    x, target, mask = fixed_batch()

    assert x.shape == (4, 37, data.INPUT_SIZE) == (4, 37, 10)  # 20 + 1 + 16
    assert target.shape == (4, 37, data.OUTPUT_SIZE) == (4, 37, 8)
    assert mask.shape == (4, 37)


def test_timesteps_covers_every_length_batch_can_draw():
    assert data.TIMESTEPS == tuple(sorted(set(data.TIMESTEPS)))  # one CUDA graph per distinct count
    for _ in range(10):
        x, _, _ = data.batch(1)
        assert x.shape[1] in data.TIMESTEPS
    for length in range(data.MIN_LEN, data.MAX_LEN + 1):
        assert data.timesteps(length) in data.TIMESTEPS


def test_the_input_is_the_vectors_and_priorities_then_a_delimiter_then_nothing():
    x, _, _ = fixed_batch(length=20)

    assert set(x[:, :20, :data.BITS].flatten().tolist()) <= {0.0, 1.0}
    assert x[:, :20, DELIMITER].sum() == 0  # the delimiter stays off while the sequence comes in
    assert x[:, 20, DELIMITER].tolist() == [1.0] * 4
    assert x[:, 20, :PRIORITY].sum() == 0  # nothing else on the delimiter step
    assert x[:, 21:, :].sum() == 0  # no input at all during the output phase


def test_priorities_are_drawn_uniformly_from_minus_one_to_one():
    x, _, _ = data.batch(512)
    priorities = x[:, :20, PRIORITY]

    assert priorities.min() >= -1.0 and priorities.max() <= 1.0
    assert priorities.min() < -0.95 and priorities.max() > 0.95  # the range is really used
    assert abs(priorities.mean().item()) < 0.02


def test_the_target_is_the_sixteen_highest_priority_vectors_highest_first():
    x, target, _ = fixed_batch(length=20)
    vectors, priorities = x[:, :20, :data.BITS], x[:, :20, PRIORITY]
    outputs = target[:, 21:]

    assert outputs.shape[1] == data.KEEP == 16
    for b in range(4):
        order = sorted(range(20), key=lambda i: -priorities[b, i])
        assert torch.equal(outputs[b], vectors[b][order[:16]])
        # the ones left out are exactly the four lowest priorities
        assert priorities[b, order[15]] >= priorities[b, order[16]]


def test_a_short_sequence_asks_for_all_of_its_vectors():
    x, target, mask = data.batch(1, 3, 3)  # what `demo` runs

    assert x.shape[1] == 7 == data.timesteps(3)
    assert int(mask[0].sum()) == 3
    assert sorted(map(tuple, target[0, 4:].tolist())) == sorted(map(tuple, x[0, :3, :data.BITS].tolist()))


def test_the_mask_scores_the_output_phase_and_nothing_before_it():
    _, _, mask = fixed_batch(length=20)

    assert not mask[:, :21].any()  # the vectors and the delimiter step are not scored
    assert mask[:, 21:].all()
    assert int(mask[0].sum()) == 16


def test_probe_cases_build_examples_the_task_can_score():
    labels = [label for label, _ in data.PROBE_CASES]
    assert labels[0] == "trained"

    for _, build in data.PROBE_CASES:
        x, target, mask = build()
        assert x.shape[0] == 1 and x.shape[2] == data.INPUT_SIZE
        assert target.shape == (1, x.shape[1], data.OUTPUT_SIZE)
        assert int(mask[0].sum()) == data.KEEP  # the output stays 16 long however long the input
