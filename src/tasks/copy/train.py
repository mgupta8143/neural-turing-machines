"""Training the LSTM baseline on the copy task, with the settings from the paper."""

import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.models.build import build_model
from src.tasks.copy.data import copy_batch


@dataclass
class TrainConfig:
    model: str = "lstm"  # one of src.models.build.MODELS

    # From the paper: Section 4.6 and Tables 1-3
    total_sequences: int = 1_000_000  # Figure 3's x-axis runs to 1000 thousand sequences
    learning_rate: float = 3e-5  # per model; see LEARNING_RATES
    momentum: float = 0.9
    clip_value: float = 10.0
    clip_norm: float = 10.0  # NTM gradients spike; clipping the whole gradient's norm is steadier

    # Not stated in the paper
    batch_size: int = 1
    seed: int = 0  # everything random is drawn from this, so a run can be repeated exactly
    device: str = ""  # "cuda", "cpu", or empty to choose automatically
    # torch.compile fuses the NTM's many small operations. It recompiles for each sequence length
    # it sees, so the warmup is slow, but over a full run it is worth about 1.2x.
    compile: bool = False
    # These models are small enough that thread synchronisation costs more than it saves; four
    # threads measured fastest, and `all` divides the cores between its processes.
    threads: int = 4

    log_every: int = 1_000  # sequences between log lines

    @property
    def run_path(self):
        return f"results/copy/{self.model}/run.json"

    @property
    def log_path(self):
        return f"results/copy/{self.model}/log.csv"

    @property
    def checkpoint_path(self):
        return f"results/copy/{self.model}/model.pt"


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
        "seed": config.seed,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "momentum": config.momentum,
        "clip_norm": config.clip_norm,
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


def train(config: TrainConfig):
    if config.threads:
        torch.set_num_threads(config.threads)

    # Seeding both generators makes a run repeatable: copy_batch draws its length with `random`
    # and its bits with torch, and the model's initial weights come from torch as well.
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = choose_device(config)
    model = build_model(config.model).to(device)
    # Compile a copy for speed, but keep `model` for checkpoints: a compiled module renames its
    # parameters, which would make the saved file unloadable by `plot` and `try`.
    step_model = torch.compile(model, dynamic=True) if config.compile else model
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

    steps = config.total_sequences // config.batch_size
    log_every_steps = max(1, config.log_every // config.batch_size)
    print(f"training {config.model} on {device}: {model.num_parameters():,} parameters, "
          f"{steps:,} steps, batch {config.batch_size}, lr {config.learning_rate}, seed {config.seed}")

    os.makedirs(os.path.dirname(config.log_path), exist_ok=True)
    record_run(config, device, model.num_parameters())
    log_lines = ["sequences,cost_bits"]
    costs = []
    start = time.time()

    for step in range(1, steps + 1):
        x, target = copy_batch(config.batch_size)
        x, target = x.to(device), target.to(device)
        length = target.shape[1]

        logits = step_model(x)[:, length + 1:]  # only the recall steps are scored
        loss = F.binary_cross_entropy_with_logits(logits, target)

        optimizer.zero_grad()
        loss.backward()
        clip(model.parameters(), config)
        optimizer.step()

        # The paper's metric: the whole sequence's cost in bits (mean loss per bit × number of bits,
        # in log base 2). Kept on the device and only read at log time: .item() every step forces
        # the GPU to finish and wait, which is most of the cost when the steps are this small.
        costs.append(loss.detach() * (target[0].numel() / math.log(2)))

        if step % log_every_steps == 0:
            sequences = step * config.batch_size
            cost = (torch.stack(costs).mean()).item()
            costs = []
            log_lines.append(f"{sequences},{cost:.4f}")
            save_log(log_lines, config.log_path)
            save_checkpoint(model, config.checkpoint_path)
            minutes = (time.time() - start) / 60
            print(f"sequences {sequences:>9,}  cost (bits) {cost:6.2f}  {minutes:5.1f} min")

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
