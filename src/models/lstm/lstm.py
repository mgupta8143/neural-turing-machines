import torch


class LSTM(torch.nn.Module):
    """LSTM baseline from the NTM paper: 3 layers of 256 units plus a readout to the output bits."""

    def __init__(self, input_size: int = 9, hidden_size: int = 256, num_layers: int = 3, output_size: int = 8):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.readout = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size) -> logits: (batch, seq_len, output_size)
        out, _ = self.lstm(x)
        return self.readout(out)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
