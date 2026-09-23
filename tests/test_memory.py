"""Checks for the memory matrix and each addressing stage.

The last test is the important one: it compares `address` against a direct, slow transcription
of equations 5 to 9 from the paper, so the fast tensor version can't quietly disagree with them.
"""

import math

import torch

from src.models.ntm.memory import address, content_weighting, interpolate, read, sharpen, shift, write

N, M = 8, 4  # a small memory, so expected values can be written out by hand


def one_hot(location, locations=N):
    w = torch.zeros(1, locations)
    w[0, location] = 1.0
    return w


# --- memory ------------------------------------------------------------------------------


def test_read_returns_the_addressed_location():
    memory = torch.rand(1, N, M)
    assert torch.allclose(read(memory, one_hot(5)), memory[:, 5])


def test_read_of_a_flat_weighting_is_the_average_location():
    memory = torch.rand(1, N, M)
    assert torch.allclose(read(memory, torch.full((1, N), 1 / N)), memory.mean(dim=1), atol=1e-6)


def test_write_replaces_one_location_and_leaves_the_others_alone():
    memory = torch.rand(1, N, M)
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    updated = write(memory, one_hot(5), erase=torch.ones(1, M), add=value)

    assert torch.allclose(updated[0, 5], value[0])
    assert torch.allclose(updated[0, :5], memory[0, :5])
    assert torch.allclose(updated[0, 6:], memory[0, 6:])


def test_writing_with_a_zero_weighting_changes_nothing():
    memory = torch.rand(1, N, M)
    updated = write(memory, torch.zeros(1, N), erase=torch.ones(1, M), add=torch.rand(1, M))
    assert torch.allclose(updated, memory)


def test_erase_and_add_act_per_element_of_a_location():
    memory = torch.ones(1, N, M)
    erase = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # clear the first two numbers only
    add = torch.tensor([[0.0, 0.0, 0.0, 5.0]])
    updated = write(memory, one_hot(2), erase, add)
    assert torch.allclose(updated[0, 2], torch.tensor([0.0, 0.0, 1.0, 6.0]))


# --- addressing stages -------------------------------------------------------------------


def test_content_weighting_peaks_at_the_matching_location():
    memory = torch.rand(1, N, M)
    w = content_weighting(memory, key=memory[:, 3].clone(), strength=torch.tensor([[50.0]]))

    assert w.argmax(dim=1).item() == 3
    assert torch.allclose(w.sum(dim=1), torch.ones(1))


def test_zero_strength_ignores_the_key_entirely():
    memory = torch.rand(1, N, M)
    w = content_weighting(memory, key=torch.rand(1, M), strength=torch.zeros(1, 1))
    assert torch.allclose(w, torch.full((1, N), 1 / N))


def test_content_weighting_sees_direction_not_length():
    # cosine similarity ignores magnitude, so a scaled copy of a location still matches it
    memory = torch.rand(1, N, M) + 0.1
    w = content_weighting(memory, key=memory[:, 6] * 10, strength=torch.tensor([[50.0]]))
    assert w.argmax(dim=1).item() == 6


def test_interpolate_blends_the_two_weightings():
    content, previous = one_hot(1), one_hot(6)
    assert torch.allclose(interpolate(content, previous, torch.ones(1, 1)), content)
    assert torch.allclose(interpolate(content, previous, torch.zeros(1, 1)), previous)
    assert torch.allclose(interpolate(content, previous, torch.tensor([[0.5]])), 0.5 * content + 0.5 * previous)


def test_shift_moves_attention_by_one_location():
    right = torch.tensor([[0.0, 0.0, 1.0]])  # weights for shifts -1, 0, +1
    left = torch.tensor([[1.0, 0.0, 0.0]])
    stay = torch.tensor([[0.0, 1.0, 0.0]])

    assert torch.allclose(shift(one_hot(3), right), one_hot(4))
    assert torch.allclose(shift(one_hot(3), left), one_hot(2))
    assert torch.allclose(shift(one_hot(3), stay), one_hot(3))


def test_shift_wraps_around_both_ends():
    right = torch.tensor([[0.0, 0.0, 1.0]])
    left = torch.tensor([[1.0, 0.0, 0.0]])

    assert torch.allclose(shift(one_hot(N - 1), right), one_hot(0))
    assert torch.allclose(shift(one_hot(0), left), one_hot(N - 1))


def test_an_unsure_shift_spreads_attention_over_the_neighbours():
    blurred = shift(one_hot(3), torch.tensor([[0.1, 0.8, 0.1]]))
    expected = 0.1 * one_hot(2) + 0.8 * one_hot(3) + 0.1 * one_hot(4)

    assert torch.allclose(blurred, expected)
    assert torch.allclose(blurred.sum(dim=1), torch.ones(1), atol=1e-6)


def test_sharpening_concentrates_a_blurry_weighting():
    blurred = torch.tensor([[0.1, 0.2, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0]])

    assert torch.allclose(sharpen(blurred, torch.ones(1, 1)), blurred)  # a power of 1 changes nothing
    sharp = sharpen(blurred, torch.tensor([[8.0]]))
    assert sharp.max() > blurred.max()
    assert sharp.argmax() == blurred.argmax()  # the order of locations is unchanged
    assert torch.allclose(sharp.sum(dim=1), torch.ones(1), atol=1e-5)


def test_sharpening_survives_tiny_weights():
    # raising these to the 10th power underflows float32, so the naive formula would return zeros
    tiny = torch.full((1, N), 1e-7)
    tiny[0, 2] = 1e-6
    sharp = sharpen(tiny, torch.tensor([[10.0]]))

    assert torch.isfinite(sharp).all()
    assert torch.allclose(sharp.sum(dim=1), torch.ones(1), atol=1e-6)
    assert sharp.argmax() == 2  # still points at the largest weight


# --- the whole pipeline against the paper's equations --------------------------------------


def paper_address(memory, previous_w, key, strength, gate, shift_weights, sharpness):
    """Equations 5 to 9 written out with loops, exactly as they appear in the paper."""
    batch, locations, _ = memory.shape
    span = shift_weights.shape[1]
    offsets = range(-(span // 2), span // 2 + 1)  # e.g. -1, 0, +1
    result = torch.zeros(batch, locations, dtype=memory.dtype)

    for b in range(batch):
        # Equation 5 and 6: softmax over strength * cosine similarity
        similarity = [
            torch.dot(key[b], memory[b, i]) / (key[b].norm() * memory[b, i].norm()) for i in range(locations)
        ]
        scaled = [math.exp(strength[b, 0] * s) for s in similarity]
        content = [value / sum(scaled) for value in scaled]

        # Equation 7: interpolate with the previous weighting
        gated = [gate[b, 0] * content[i] + (1 - gate[b, 0]) * previous_w[b, i] for i in range(locations)]

        # Equation 8: circular convolution. Location j sends its weight to j + offset, modulo N.
        shifted = [0.0] * locations
        for j in range(locations):
            for k, offset in enumerate(offsets):
                shifted[(j + offset) % locations] += gated[j] * shift_weights[b, k]

        # Equation 9: sharpen and renormalise
        powered = [value ** sharpness[b, 0] for value in shifted]
        for i in range(locations):
            result[b, i] = powered[i] / sum(powered)

    return result


def random_addressing_arguments(batch, shift_positions):
    """Random but valid head outputs, in double precision so the comparison can be strict."""
    return dict(
        key=torch.rand(batch, M, dtype=torch.float64),
        strength=torch.rand(batch, 1, dtype=torch.float64) * 10,
        gate=torch.rand(batch, 1, dtype=torch.float64),
        shift_weights=torch.softmax(torch.rand(batch, shift_positions, dtype=torch.float64), dim=1),
        sharpness=1 + torch.rand(batch, 1, dtype=torch.float64) * 5,
    )


def test_address_matches_the_paper_equations():
    torch.manual_seed(0)
    batch = 3
    for _ in range(5):
        # away from zero, where cosine similarity is undefined
        memory = torch.rand(batch, N, M, dtype=torch.float64) + 0.05
        previous_w = torch.softmax(torch.rand(batch, N, dtype=torch.float64), dim=1)
        arguments = random_addressing_arguments(batch, shift_positions=3)

        ours = address(memory, previous_w, **arguments)
        theirs = paper_address(memory, previous_w, **arguments)

        assert torch.allclose(ours, theirs, atol=1e-8), (ours - theirs).abs().max()
        assert torch.allclose(ours.sum(dim=1), torch.ones(batch, dtype=torch.float64))
        assert (ours >= 0).all()


def test_address_matches_the_paper_with_five_shift_positions():
    torch.manual_seed(1)
    memory = torch.rand(2, N, M, dtype=torch.float64) + 0.05
    previous_w = torch.softmax(torch.rand(2, N, dtype=torch.float64), dim=1)
    arguments = random_addressing_arguments(2, shift_positions=5)  # shifts -2 to +2

    ours = address(memory, previous_w, **arguments)
    theirs = paper_address(memory, previous_w, **arguments)
    assert torch.allclose(ours, theirs, atol=1e-8), (ours - theirs).abs().max()
