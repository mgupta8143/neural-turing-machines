"""Running a training step as a replayable CUDA graph.

The NTM launches a few hundred tiny GPU kernels per sequence, so on a GPU it spends its time
launching work rather than doing it. A CUDA graph records the whole step once - forward,
backward, gradient clipping and the optimiser update - and replays it as a single launch.

Capture needs static shapes, and every task here draws a random sequence length, so the number
of timesteps varies from batch to batch. Padding every example to the longest would add work to
every step, so instead each timestep count the task can produce gets its own graph, which keeps
the arithmetic identical to the eager version.
"""

import torch

from src.loss import masked_bce


class CapturedStep:
    """One captured training step per timestep count, plus the running cost in bits."""

    def __init__(self, model, optimizer, clip_norm, task, batch_size=1):
        self.model = model
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.parameters = list(model.parameters())
        device = self.parameters[0].device

        # Filled in by every graph, read at log time, so nothing has to wait on the GPU per step
        self.cost_total = torch.zeros((), device=device)
        self.cost_squared = torch.zeros((), device=device)  # for the spread; see average_cost
        self.sequences = torch.zeros((), device=device)

        self.graphs, self.buffers = {}, {}
        for timesteps in task.TIMESTEPS:
            self._capture(timesteps, task, batch_size, device)

    def _capture(self, timesteps, task, batch_size, device):
        x = torch.zeros(batch_size, timesteps, task.INPUT_SIZE, device=device)
        target = torch.zeros(batch_size, timesteps, task.OUTPUT_SIZE, device=device)
        # The mask is held as floats rather than bools so the loss's multiply stays in float32
        mask = torch.zeros(batch_size, timesteps, device=device)

        def step():
            loss, cost = masked_bce(self.model(x), target, mask)
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.grad.zero_()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, self.clip_norm)
            self.optimizer.step()
            self.cost_total += cost
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
        self.graphs[timesteps], self.buffers[timesteps] = graph, (x, target, mask)

    @property
    def pool(self):
        """Share one memory pool across the graphs so they do not each reserve their own."""
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

    def run(self, x, target, mask):
        timesteps = x.shape[1]
        for buffer, value in zip(self.buffers[timesteps], (x, target, mask)):
            buffer.copy_(value, non_blocking=True)
        self.graphs[timesteps].replay()

    def average_cost(self):
        """Mean and standard deviation in bits since the last call. Reads from the GPU, so call it
        at log time.

        The spread is worth having: with one sequence per update the mean is dominated by a rare
        heavy tail, which can make a converged run look as though it has diverged.
        """
        count = self.sequences.clamp(min=1)
        average = (self.cost_total / count).item()
        spread = max(0.0, (self.cost_squared / count).item() - average**2) ** 0.5
        self.cost_total.zero_()
        self.cost_squared.zero_()
        self.sequences.zero_()
        return average, spread
