"""The Neural Turing Machine: a controller that reads and writes an external memory matrix.

Same call signature as the LSTM baseline, (batch, timesteps, input_size) -> logits, so the
copy task, the training loop and the plots work with either model.
"""

import torch
from torch import nn

from src.models.ntm.controllers import FeedForwardController, LSTMController
from src.models.ntm.heads import ReadHead, WriteHead


class NTM(nn.Module):
    def __init__(
        self,
        input_size: int = 9,
        output_size: int = 8,
        controller: str = "feedforward",
        controller_size: int = 100,
        memory_locations: int = 128,
        memory_width: int = 20,
        num_read_heads: int = 1,
        num_write_heads: int = 1,
    ):
        super().__init__()
        self.memory_locations = memory_locations
        self.memory_width = memory_width

        # The controller sees the external input plus what the read heads found last timestep
        controller_input = input_size + num_read_heads * memory_width
        controller_class = {"feedforward": FeedForwardController, "lstm": LSTMController}[controller]
        self.controller = controller_class(controller_input, controller_size)

        self.read_heads = nn.ModuleList(ReadHead(controller_size, memory_width) for _ in range(num_read_heads))
        self.write_heads = nn.ModuleList(WriteHead(controller_size, memory_width) for _ in range(num_write_heads))
        self.output = nn.Linear(controller_size + num_read_heads * memory_width, output_size)

        # Learned starting state, reset at the start of every sequence ("bias values" in the paper).
        # These start random rather than constant on purpose: identical memory rows and uniform
        # weightings leave every location symmetric, so the gradients are identical too and the
        # 128 locations never differentiate. In practice the model then plateaus near chance.
        self.initial_memory = nn.Parameter(torch.randn(memory_locations, memory_width) * 0.05)
        self.initial_w = nn.Parameter(torch.randn(len(self.read_heads) + len(self.write_heads), memory_locations))
        self.initial_reads = nn.Parameter(torch.zeros(num_read_heads, memory_width))

    def initial_state(self, batch_size: int):
        """The state reset at the start of every sequence, from learned bias values."""
        memory = self.initial_memory.unsqueeze(0).expand(batch_size, -1, -1)
        weightings = [w.softmax(dim=0).unsqueeze(0).expand(batch_size, -1) for w in self.initial_w]
        reads = [r.unsqueeze(0).expand(batch_size, -1) for r in self.initial_reads]
        return memory, weightings, reads, self.controller.initial_state(batch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, _ = x.shape
        memory, weightings, reads, controller_state = self.initial_state(batch_size)
        heads = list(self.write_heads) + list(self.read_heads)  # weightings are stored in this order

        outputs = []
        for t in range(timesteps):
            h, controller_state = self.controller(torch.cat([x[:, t], *reads], dim=1), controller_state)

            # Write first, so the read heads see the memory as it was just written
            for i, head in enumerate(self.write_heads):
                memory, weightings[i] = head(h, weightings[i], memory)

            reads = []
            for j, head in enumerate(self.read_heads, start=len(self.write_heads)):
                r, weightings[j] = head(h, weightings[j], memory)
                reads.append(r)

            outputs.append(self.output(torch.cat([h, *reads], dim=1)))

        return torch.stack(outputs, dim=1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
