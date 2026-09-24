"""Read and write heads.

A head is one linear layer on the controller's hidden vector. Its output is split into the
addressing parameters, each with its own activation, and turned into a weighting over memory.
What the head does with that weighting is the only difference between reading and writing.

Every head reads the same hidden vector, so the NTM runs all their linear layers as one
matmul (`stacked_projection`) and hands each head its own slice of the result.
"""

import torch
import torch.nn.functional as F
from torch import nn

from src.models.ntm.memory import address, read, write


class Head(nn.Module):
    """Emits key, strength, gate, shift and sharpness, plus `extra` more numbers for subclasses."""

    def __init__(self, controller_size, memory_width, shift_range=3, extra=0):
        super().__init__()
        self.sizes = [memory_width, 1, 1, shift_range, 1] + ([memory_width] * extra)
        self.fc = nn.Linear(controller_size, sum(self.sizes))

    def weighting(self, parameters, previous_w, memory, norms):
        """Returns the new weighting, and any extra vectors this head asked for.

        `parameters` is this head's slice of the stacked projection: self.fc(h).
        """
        key, strength, gate, shift_weights, sharpness, *extra = torch.split(parameters, self.sizes, dim=1)
        w = address(
            memory,
            previous_w,
            key=key,
            strength=F.softplus(strength),  # positive
            gate=torch.sigmoid(gate),  # between 0 and 1
            shift_weights=F.softmax(shift_weights, dim=1),  # a distribution over shifts
            sharpness=1 + F.softplus(sharpness),  # at least 1, so this never blurs
            norms=norms,
        )
        return w, extra


def stacked_projection(heads):
    """Every head's linear layer stacked into one weight and bias, plus the sizes to split by.

    One matmul per timestep instead of one per head. At batch 1 this model is bound by how many
    PyTorch calls it makes rather than by arithmetic, so that is worth doing; the stacking itself
    happens once per sequence. Each head keeps its own parameters, so checkpoints are unaffected.
    """
    weight = torch.cat([head.fc.weight for head in heads])
    bias = torch.cat([head.fc.bias for head in heads])
    return weight, bias, [sum(head.sizes) for head in heads]


class ReadHead(Head):
    def forward(self, parameters, previous_w, memory, norms):
        """Returns what it read, (B, M), and the weighting it used."""
        w, _ = self.weighting(parameters, previous_w, memory, norms)
        return read(memory, w), w


class WriteHead(Head):
    def __init__(self, controller_size, memory_width, shift_range=3):
        super().__init__(controller_size, memory_width, shift_range, extra=2)  # erase and add

    def forward(self, parameters, previous_w, memory, norms):
        """Returns the updated memory, the weighting it used, and the vector it added."""
        w, (erase, add) = self.weighting(parameters, previous_w, memory, norms)
        return write(memory, w, torch.sigmoid(erase), add), w, add
