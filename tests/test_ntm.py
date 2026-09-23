"""Checks for the whole NTM model. The memory and addressing pieces are in test_memory.py."""

import torch

from src.models.ntm.ntm import NTM

N, M = 8, 4  # a small memory, to keep these fast


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
