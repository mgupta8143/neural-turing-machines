"""Running a training step as a replayable CUDA graph.

The NTM launches a few hundred tiny GPU kernels per sequence, so on a GPU it spends its time
launching work rather than doing it. A CUDA graph records the whole step once - forward,
backward, gradient clipping and the optimiser update - and replays it as a single launch.

Capture needs static shapes, and the copy task's sequence length varies from 1 to 20. Padding
every sequence to the longest would double the timesteps, so instead each length gets its own
graph, which keeps the arithmetic identical to the eager version.
"""

import math

import torch
import torch.nn.functional as F


class CapturedStep:
    """One captured training step per sequence length, plus the running cost in bits."""

    def __init__(self, model, optimizer, clip_norm, max_length=20, input_size=9, bits=8):
        self.model = model
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.parameters = list(model.parameters())
        device = self.parameters[0].device

        # Filled in by every graph, read at log time, so nothing has to wait on the GPU per step
        self.cost_total = torch.zeros((), device=device)
        self.sequences = torch.zeros((), device=device)

        self.graphs, self.buffers = {}, {}
        for length in range(1, max_length + 1):
            self._capture(length, device, input_size, bits)

    def _capture(self, length, device, input_size, bits):
        x = torch.zeros(1, 2 * length + 1, input_size, device=device)
        target = torch.zeros(1, length, bits, device=device)
        # cost per sequence in bits, the paper's metric, accumulated inside the graph
        to_bits = length * bits / math.log(2)

        def step():
            logits = self.model(x)[:, length + 1:]
            loss = F.binary_cross_entropy_with_logits(logits, target)
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.grad.zero_()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, self.clip_norm)
            self.optimizer.step()
            self.cost_total += loss.detach() * to_bits
            self.sequences += 1

        # Warm up on a side stream, which CUDA requires before capturing
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for parameter in self.parameters:
            if parameter.grad is None:  # capture needs every gradient to already exist
                parameter.grad = torch.zeros_like(parameter)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            step()
        self.graphs[length], self.buffers[length] = graph, (x, target)

    @property
    def pool(self):
        """Share one memory pool across the graphs so 20 of them do not each reserve their own."""
        existing = next(iter(self.graphs.values()), None)
        return existing.pool() if existing else None

    def reset(self, model_state, optimizer_state):
        """Restore weights in place, since the captured graphs point at these exact tensors."""
        with torch.no_grad():
            for parameter, saved in zip(self.parameters, model_state):
                parameter.copy_(saved)
            for state in self.optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        value.zero_()

    def run(self, x, target):
        length = target.shape[1]
        buffered_x, buffered_target = self.buffers[length]
        buffered_x.copy_(x, non_blocking=True)
        buffered_target.copy_(target, non_blocking=True)
        self.graphs[length].replay()

    def average_cost(self) -> float:
        """Mean cost in bits since the last call. Reads from the GPU, so call it at log time."""
        average = (self.cost_total / self.sequences.clamp(min=1)).item()
        self.cost_total.zero_()
        self.sequences.zero_()
        return average
