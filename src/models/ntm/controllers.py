"""Controllers for the NTM.

Both take one timestep at a time and expose the same two methods, so the NTM never has to
know which one it is driving. That is what makes them interchangeable.
"""

import torch
from torch import nn


class FeedForwardController(nn.Module):
    """One layer, no memory of its own: everything it recalls has to come from the memory matrix."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)

    def initial_state(self, batch_size):
        return None

    def forward(self, x, state):
        return torch.tanh(self.fc(x)), None


class LSTMController(nn.Module):
    """An LSTM cell, so the controller also has a small internal memory alongside the memory matrix."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        # The paper resets to a learned bias at the start of each sequence
        self.h0 = nn.Parameter(torch.zeros(1, hidden_size))
        self.c0 = nn.Parameter(torch.zeros(1, hidden_size))

    def initial_state(self, batch_size):
        return self.h0.expand(batch_size, -1), self.c0.expand(batch_size, -1)

    def forward(self, x, state):
        h, c = self.cell(x, state)
        return h, (h, c)
