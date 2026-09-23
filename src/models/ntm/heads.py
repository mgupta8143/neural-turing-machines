"""Read and write heads.

A head is one linear layer on the controller's hidden vector. Its output is split into the
addressing parameters, each with its own activation, and turned into a weighting over memory.
What the head does with that weighting is the only difference between reading and writing.
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

    def weighting(self, h, previous_w, memory):
        """Returns the new weighting, and any extra vectors this head asked for."""
        key, strength, gate, shift_weights, sharpness, *extra = torch.split(self.fc(h), self.sizes, dim=1)
        w = address(
            memory,
            previous_w,
            key=key,
            strength=F.softplus(strength),  # positive
            gate=torch.sigmoid(gate),  # between 0 and 1
            shift_weights=F.softmax(shift_weights, dim=1),  # a distribution over shifts
            sharpness=1 + F.softplus(sharpness),  # at least 1, so this never blurs
        )
        return w, extra


class ReadHead(Head):
    def forward(self, h, previous_w, memory):
        """Returns what it read, (B, M), and the weighting it used."""
        w, _ = self.weighting(h, previous_w, memory)
        return read(memory, w), w


class WriteHead(Head):
    def __init__(self, controller_size, memory_width, shift_range=3):
        super().__init__(controller_size, memory_width, shift_range, extra=2)  # erase and add

    def forward(self, h, previous_w, memory):
        """Returns the updated memory and the weighting it used."""
        w, (erase, add) = self.weighting(h, previous_w, memory)
        return write(memory, w, torch.sigmoid(erase), add), w
