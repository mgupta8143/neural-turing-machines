"""Checks for the NTM: memory, addressing, and the whole model."""

import torch

from src.models.ntm.memory import address, read, write
from src.models.ntm.ntm import NTM

N, M = 8, 4  # a small memory, so expected results can be written out by hand


def one_hot(location, locations=N):
    w = torch.zeros(1, locations)
    w[0, location] = 1.0
    return w


def address_with(memory, previous_w, key, strength=50.0, gate=1.0, shifts=(0, 1, 0), sharpness=1.0):
    return address(
        memory,
        previous_w,
        key=key,
        strength=torch.tensor([[strength]]),
        gate=torch.tensor([[gate]]),
        shift_weights=torch.tensor([list(shifts)]).float(),
        sharpness=torch.tensor([[sharpness]]),
    )


def test_write_replaces_one_location_and_read_returns_it():
    memory = torch.rand(1, N, M)
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    updated = write(memory, one_hot(5), erase=torch.ones(1, M), add=value)

    assert torch.allclose(updated[0, 5], value[0])  # the targeted location was replaced
    assert torch.allclose(updated[0, :5], memory[0, :5])  # everything else is untouched
    assert torch.allclose(updated[0, 6:], memory[0, 6:])
    assert torch.allclose(read(updated, one_hot(5)), value)


def test_content_lookup_finds_the_matching_location():
    memory = torch.rand(1, N, M)
    w = address_with(memory, torch.full((1, N), 1 / N), key=memory[:, 3].clone())

    assert w.argmax(dim=1).item() == 3  # asked for what is stored at location 3
    assert torch.allclose(w.sum(dim=1), torch.ones(1))


def test_shifting_moves_the_weighting_and_wraps_around():
    memory = torch.rand(1, N, M)
    key = torch.rand(1, M)

    # gate 0 ignores the content lookup entirely and shifts the previous weighting
    assert address_with(memory, one_hot(3), key, gate=0.0, shifts=(0, 0, 1)).argmax().item() == 4
    assert address_with(memory, one_hot(3), key, gate=0.0, shifts=(1, 0, 0)).argmax().item() == 2
    assert address_with(memory, one_hot(N - 1), key, gate=0.0, shifts=(0, 0, 1)).argmax().item() == 0


def test_sharpening_concentrates_a_blurry_weighting():
    memory = torch.rand(1, N, M)
    key = torch.rand(1, M)
    blurred = address_with(memory, one_hot(3), key, gate=0.0, shifts=(0.1, 0.8, 0.1))
    sharpened = address_with(memory, one_hot(3), key, gate=0.0, shifts=(0.1, 0.8, 0.1), sharpness=8.0)

    assert sharpened.max() > blurred.max()
    assert torch.allclose(sharpened.sum(dim=1), torch.ones(1))


def test_ntm_runs_and_every_parameter_gets_a_gradient():
    for controller in ["feedforward", "lstm"]:
        model = NTM(controller=controller, memory_locations=N, memory_width=M, controller_size=16)
        logits = model(torch.rand(2, 7, 9))
        assert logits.shape == (2, 7, 8)

        logits.sum().backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{controller}: {name} got no gradient"
            assert torch.isfinite(parameter.grad).all(), f"{controller}: {name} has a non-finite gradient"


def test_parameter_counts_are_close_to_the_paper():
    assert 15_000 < NTM(controller="feedforward").num_parameters() < 19_000  # paper: 17,162
    assert 60_000 < NTM(controller="lstm").num_parameters() < 72_000  # paper: 67,561
