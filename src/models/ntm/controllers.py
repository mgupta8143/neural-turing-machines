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
    """LSTM cells, so the controller also has a small internal memory alongside the memory matrix.

    The paper's Table 2 gives priority sort a controller of "2 x 100", two layers of 100 units,
    so the number of layers is configurable; every other task uses one.
    """

    def __init__(self, input_size, hidden_size, layers=1):
        super().__init__()
        sizes = [input_size] + [hidden_size] * layers
        self.cells = nn.ModuleList(nn.LSTMCell(sizes[i], sizes[i + 1]) for i in range(layers))
        # The paper resets to a learned bias at the start of each sequence
        self.h0 = nn.Parameter(torch.zeros(layers, 1, hidden_size))
        self.c0 = nn.Parameter(torch.zeros(layers, 1, hidden_size))

    def initial_state(self, batch_size):
        return ([h.expand(batch_size, -1) for h in self.h0],
                [c.expand(batch_size, -1) for c in self.c0])

    def forward(self, x, state):
        hs, cs = state
        new_hs, new_cs = [], []
        for cell, h, c in zip(self.cells, hs, cs):
            h, c = cell(x, (h, c))
            new_hs.append(h)
            new_cs.append(c)
            x = h  # each layer feeds the one above
        return x, (new_hs, new_cs)
