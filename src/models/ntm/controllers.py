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
        # Every layer sees the network input, not just the layer below it: that is the deep LSTM
        # of Graves (2013), which Section 4.6 cites for the optimiser and whose architecture the
        # paper follows. It is also what reconciles the parameter count - without the skip our
        # two-layer priority sort controller is 19% under Table 2, and with it 2.6%, in line with
        # every single-layer model. For one layer it changes nothing.
        self.cells = nn.ModuleList(
            nn.LSTMCell(input_size if i == 0 else hidden_size + input_size, hidden_size)
            for i in range(layers)
        )
        # The paper resets to a learned bias at the start of each sequence
        self.h0 = nn.Parameter(torch.zeros(layers, 1, hidden_size))
        self.c0 = nn.Parameter(torch.zeros(layers, 1, hidden_size))

    def initial_state(self, batch_size):
        return ([h.expand(batch_size, -1) for h in self.h0],
                [c.expand(batch_size, -1) for c in self.c0])

    def forward(self, x, state):
        hs, cs = state
        new_hs, new_cs = [], []
        below = x
        for i, (cell, h, c) in enumerate(zip(self.cells, hs, cs)):
            h, c = cell(below if i == 0 else torch.cat([below, x], dim=1), (h, c))
            new_hs.append(h)
            new_cs.append(c)
            below = h
        return below, (new_hs, new_cs)
