"""Checks for the repeat copy task's data (paper, Section 4.2).

An example is built from a length L and a repeat count R, so every check fixes both and then
looks at the one place the encoding puts each piece: the delimiter, the repeat scalar, the R
copies of the sequence and the end marker.
"""

import torch

from src.tasks.repeat_copy import data

BITS, DELIMITER, REPEATS = data.BITS, data.BITS, data.BITS + 1  # input channel indices
END = data.BITS  # output channel index of the end-of-sequence marker


def fixed_batch(batch_size=2, length=3, repeats=4):
    return data.batch(batch_size, length, length, repeats, repeats)


def test_shapes_match_the_declared_channel_counts():
    x, target, mask = fixed_batch()
    timesteps = data.timesteps(length=3, repeats=4)  # 3 + 2 + 12 + 1

    assert x.shape == (2, timesteps, data.INPUT_SIZE) == (2, 18, 10)
    assert target.shape == (2, timesteps, data.OUTPUT_SIZE) == (2, 18, 9)
    assert mask.shape == (2, timesteps)
    assert data.TIMESTEPS == tuple(sorted(set(data.TIMESTEPS)))  # one CUDA graph per distinct count
    assert timesteps in data.TIMESTEPS


def test_the_input_is_the_sequence_then_a_delimiter_then_nothing():
    x, _, _ = fixed_batch(length=3, repeats=4)

    assert set(x[:, :3, :BITS].flatten().tolist()) <= {0.0, 1.0}  # random bits
    assert x[:, :3, DELIMITER:].sum() == 0  # the flags stay off while the sequence comes in
    assert x[:, 3, DELIMITER].tolist() == [1.0, 1.0]
    assert x[:, 4:, :].abs().sum() > 0  # only the repeat scalar, checked below
    assert x[:, 5:, :].sum() == 0  # no input at all during the output phase


def test_the_repeat_scalar_appears_once_and_is_normalised():
    x, _, _ = fixed_batch(length=3, repeats=4)

    appearances = x[0, :, REPEATS].nonzero().flatten()
    assert appearances.tolist() == [4]  # its own step, right after the delimiter
    assert torch.isclose(x[0, 4, REPEATS], torch.tensor((4 - 5.5) / 8.25**0.5))

    # Mean zero and variance one over the 1..10 the paper draws from
    scalars = torch.tensor([data.batch(1, 2, 2, r, r)[0][0, 3, REPEATS] for r in range(1, 11)])
    assert torch.isclose(scalars.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(scalars.var(unbiased=False), torch.tensor(1.0), atol=1e-6)


def test_the_target_is_the_sequence_repeated_that_many_times():
    length, repeats = 3, 4
    x, target, _ = fixed_batch(length=length, repeats=repeats)
    vectors = x[:, :length, :BITS]

    outputs = target[:, length + 2:, :BITS]  # the output phase, including the end marker step
    for copy in range(repeats):
        assert torch.equal(outputs[:, copy * length:(copy + 1) * length], vectors)
    assert torch.equal(outputs[:, repeats * length:], torch.zeros(2, 1, BITS))


def test_the_end_marker_is_on_the_step_after_the_last_copy():
    length, repeats = 3, 4
    _, target, _ = fixed_batch(length=length, repeats=repeats)

    marker = target[0, :, END].nonzero().flatten()
    assert marker.tolist() == [length + 2 + repeats * length]  # exactly one step, the last one
    assert marker.item() == target.shape[1] - 1


def test_the_mask_scores_the_output_phase_and_nothing_before_it():
    length, repeats = 3, 4
    _, _, mask = fixed_batch(length=length, repeats=repeats)

    assert not mask[:, :length + 2].any()  # the input phase is not scored
    assert mask[:, length + 2:].all()  # every copy and the end marker step are
    assert int(mask[0].sum()) == repeats * length + 1


def test_random_draws_stay_inside_the_paper_s_range():
    for _ in range(20):
        x, target, mask = data.batch(1)
        length = int(x[0, :, DELIMITER].nonzero().item())
        repeats = (int(mask[0].sum()) - 1) // length

        assert data.MIN_LEN <= length <= data.MAX_LEN
        assert data.MIN_REPEATS <= repeats <= data.MAX_REPEATS
        assert x.shape[1] == data.timesteps(length, repeats)
