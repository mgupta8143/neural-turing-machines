"""Training a model on a task, with the settings from the paper."""

import json
import os
import random
import subprocess
import time
from dataclasses import dataclass

import torch

from src.loss import masked_bce
from src.models.build import build_model, learning_rate as paper_learning_rate
from src.probe import probe
from src.tasks.registry import get_task


@dataclass
class TrainConfig:
    model: str = "lstm"  # one of src.models.build.MODELS
    task: str = "copy"  # one of src.tasks.registry.TASKS

    # From the paper: Section 4.6 and Tables 1-3
    total_sequences: int = 1_000_000  # Figure 3's x-axis runs to 1000 thousand sequences
    learning_rate: float = 0.0  # 0 means take the paper's rate for this task and model
    momentum: float = 0.9
    clip_value: float = 10.0
    clip_norm: float = 10.0
    # Section 4.6 cites RMSProp "in the form described in (Graves, 2013)": centered, decay 0.95,
    # and a damping term of 1e-4 inside the square root. PyTorch's defaults are 0.99 and 1e-8,
    # and that 1e-8 inflates the step for parameters whose gradient variance is small, which is
    # most of an LSTM controller's recurrent matrix.
    alpha: float = 0.95
    eps: float = 1e-4
    centered: bool = True  # NTM gradients spike; clipping the whole gradient's norm is steadier

    # Not stated in the paper
    batch_size: int = 1
    seed: int = 0  # everything random is drawn from this, so a run can be repeated exactly
    device: str = ""  # "cuda", "cpu", or empty to choose automatically
    # torch.compile fuses the NTM's many small operations. It recompiles for each sequence length
    # it sees, so the warmup is slow, but over a full run it is worth about 1.2x.
    compile: bool = False
    # On a GPU, replay the whole step as one CUDA graph instead of launching every kernel: about
    # 8x faster for the NTM, and bit-identical to the eager version.
    cuda_graphs: bool = True
    # These models are small enough that thread synchronisation costs more than it saves; four
    # threads measured fastest, and `all` divides the cores between its processes.
    threads: int = 4

    log_every: int = 1_000  # sequences between log lines
    # Sequences between diagnostics rows. These cost three forward passes, so they are rare, but
    # they are the only record of whether what the model learned will survive a longer sequence:
    # see src/probe.py. Set to 0 to skip them.
    probe_every: int = 10_000

    @property
    def results_path(self):
        return f"results/{self.task}/{self.model}"

    @property
    def run_path(self):
        return f"{self.results_path}/run.json"

    @property
    def log_path(self):
        return f"{self.results_path}/log.csv"

    @property
    def diagnostics_path(self):
        return f"{self.results_path}/diagnostics.csv"

    @property
    def best_checkpoint_path(self):
        return f"results/{self.task}/{self.model}/best.pt"

    @property
    def checkpoint_path(self):
        return f"{self.results_path}/model.pt"


def clip(parameters, config: TrainConfig):
    """The paper clips each gradient component to (-10, 10). That bounds each value but not the
    step: with a million components all at the limit the update is still huge, which is how the
    NTM diverges after it has started learning. For the NTM we clip the gradient norm instead."""
    if config.model.startswith("ntm"):
        torch.nn.utils.clip_grad_norm_(parameters, config.clip_norm)
    else:
        torch.nn.utils.clip_grad_value_(parameters, config.clip_value)


def choose_device(config: TrainConfig) -> str:
    """The NTM runs a Python loop of small operations per timestep, so a GPU spends its time
    launching kernels rather than computing: on the copy task it is several times slower than a
    CPU. The LSTM is the opposite, since cuDNN runs a whole sequence in one kernel."""
    if config.device:
        return config.device
    if config.model.startswith("ntm"):
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def record_run(config: TrainConfig, device: str, parameters: int):
    """Write down everything needed to repeat this run, next to its results."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        commit = "unknown"
    details = {
        "model": config.model,
        "task": config.task,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "momentum": config.momentum,
        "clip_norm": config.clip_norm,
        "alpha": config.alpha,
        "eps": config.eps,
        "centered": config.centered,
        "clip_value": config.clip_value,
        "total_sequences": config.total_sequences,
        "parameters": parameters,
        "device": device,
        "torch": torch.__version__,
        "commit": commit,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(config.run_path, "w") as f:
        json.dump(details, f, indent=2)
    return details


# --------------------------------------------------------------------------------------------
# Replaying a training step as a CUDA graph
#
# The NTM launches a few hundred tiny GPU kernels per sequence, so on a GPU it spends its time
# launching work rather than doing it. A CUDA graph records the whole step once - forward,
# backward, gradient clipping and the optimiser update - and replays it as a single launch.
#
# Capture needs static shapes, and every task here draws a random sequence length, so the number
# of timesteps varies from batch to batch. Padding every example to the longest would add work to
# every step, so instead each timestep count the task can produce gets its own graph, which keeps
# the arithmetic identical to the eager version.
# --------------------------------------------------------------------------------------------


class CapturedStep:
    """One captured training step per timestep count, plus the running cost in bits."""

    def __init__(self, model, optimizer, clip, task, batch_size=1):
        self.model = model
        self.optimizer = optimizer
        self.clip = clip  # the same callable the eager path uses, so both clip identically
        self.parameters = list(model.parameters())
        device = self.parameters[0].device

        # Filled in by every graph, read at log time, so nothing has to wait on the GPU per step
        self.cost_total = torch.zeros((), device=device)
        self.cost_squared = torch.zeros((), device=device)  # for the spread; see average_cost
        self.sequences = torch.zeros((), device=device)

        self.graphs, self.buffers = {}, {}
        for timesteps in task.TIMESTEPS:
            self._capture(timesteps, task, batch_size, device)
        # Capture warms up by running the step three times per timestep count. Those runs see
        # all-zero buffers, so they add nothing to the cost, but they do count sequences - which
        # would divide the first log line's mean by a denominator inflated by 3 x len(TIMESTEPS).
        self.cost_total.zero_()
        self.cost_squared.zero_()
        self.sequences.zero_()

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
            self.clip(self.parameters)
            self.optimizer.step()
            self.cost_total += cost
            self.cost_squared += cost * cost
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
        # population variance, E[x^2] - E[x]^2; the max() absorbs float cancellation near zero
        spread = max(0.0, (self.cost_squared / count).item() - average**2) ** 0.5
        self.cost_total.zero_()
        self.cost_squared.zero_()
        self.sequences.zero_()
        return average, spread


def train(config: TrainConfig):
    if config.threads:
        torch.set_num_threads(config.threads)

    # Seeding both generators makes a run repeatable: a task draws its lengths with `random`
    # and its bits with torch, and the model's initial weights come from torch as well.
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    task = get_task(config.task)
    if not config.learning_rate:
        config.learning_rate = paper_learning_rate(config.task, config.model)

    device = choose_device(config)
    model = build_model(config.model, task.INPUT_SIZE, task.OUTPUT_SIZE, config.task).to(device)
    graphed = device == "cuda" and config.cuda_graphs
    # Compile a copy for speed, but keep `model` for checkpoints: a compiled module renames its
    # parameters, which would make the saved file unloadable by `plot` and `try`.
    step_model = torch.compile(model, dynamic=True) if config.compile else model
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, momentum=config.momentum,
                                    alpha=config.alpha, eps=config.eps, centered=config.centered,
                                    capturable=graphed, foreach=graphed)

    # Capture the step as CUDA graphs once the optimiser exists: the graphs hold references to
    # these exact parameter, gradient and optimiser-state tensors.
    captured = None
    if graphed:
        saved = [p.detach().clone() for p in model.parameters()]
        captured = CapturedStep(model, optimizer, lambda p: clip(p, config), task, config.batch_size)
        captured.reset(saved, optimizer.state)  # undo what capture's warm-up did to the weights

    steps = config.total_sequences // config.batch_size
    log_every_steps = max(1, config.log_every // config.batch_size)
    print(f"training {config.model} on {config.task}, {device}: {model.num_parameters():,} parameters, "
          f"{steps:,} steps, batch {config.batch_size}, lr {config.learning_rate}, seed {config.seed}"
          f"{', cuda graphs' if graphed else ''}")

    os.makedirs(os.path.dirname(config.log_path), exist_ok=True)
    record_run(config, device, model.num_parameters())
    log_lines = ["sequences,cost_bits,median_bits,spread_bits"]
    diagnostics = []
    best_cost = float("inf")
    costs = []
    start = time.time()

    for step in range(1, steps + 1):
        x, target, mask = task.batch(config.batch_size)
        x, target, mask = x.to(device), target.to(device), mask.to(device)

        if captured:
            captured.run(x, target, mask)  # one launch: forward, backward, clipping and the update
        else:
            loss, cost = masked_bce(step_model(x), target, mask)

            optimizer.zero_grad()
            loss.backward()
            clip(model.parameters(), config)
            optimizer.step()

            # The paper's metric, the whole sequence's cost in bits, kept on the device and only
            # read at log time: .item() every step forces the GPU to finish and wait, which
            # dominates when steps are this small.
            costs.append(cost)

        if step % log_every_steps == 0:
            sequences = step * config.batch_size
            if captured:
                # a captured graph cannot sort, so it reports a standard deviation, not a median
                cost, spread = captured.average_cost()
                middle = ""
            else:
                window = torch.stack(costs)
                cost, spread = window.mean().item(), window.std().item()
                middle = f"{window.median().item():.4f}"
            costs = []
            log_lines.append(f"{sequences},{cost:.4f},{middle},{spread:.4f}")
            save_log(log_lines, config.log_path)
            save_checkpoint(model, config.checkpoint_path)
            if cost < best_cost:  # training can drift away from a good solution; keep the best one
                best_cost = cost
                save_checkpoint(model, config.best_checkpoint_path)
            minutes = (time.time() - start) / 60
            print(f"sequences {sequences:>9,}  cost (bits) {cost:6.2f} +/- {spread:6.2f}  {minutes:5.1f} min")

        if config.probe_every and step * config.batch_size % config.probe_every == 0:
            row = probe(model, task, device)
            if row:
                if not diagnostics:
                    diagnostics.append("sequences," + ",".join(row))
                diagnostics.append(f"{step * config.batch_size}," +
                                   ",".join(f"{v:.4f}" for v in row.values()))
                save_log(diagnostics, config.diagnostics_path)
                print("  probe  " + "  ".join(f"{k} {v:.3f}" for k, v in row.items()))

    return model


# Both saves write to a temporary file and then swap it in, so anything reading these files
# (like `plot` during training, or Google Drive syncing them) never sees a half-written file.


def save_log(lines, path):
    with open(path + ".tmp", "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(path + ".tmp", path)


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path + ".tmp")
    os.replace(path + ".tmp", path)
