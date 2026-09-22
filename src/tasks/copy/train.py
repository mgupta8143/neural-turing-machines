"""Training the LSTM baseline on the copy task, with the settings from the paper."""

import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.models.lstm import LSTM
from src.tasks.copy.data import copy_batch


@dataclass
class TrainConfig:
    # From the paper: Section 4.6 and Table 3
    total_sequences: int = 1_000_000  # Figure 3's x-axis runs to 1000 thousand sequences
    learning_rate: float = 3e-5
    momentum: float = 0.9
    clip_value: float = 10.0

    # Not stated in the paper
    batch_size: int = 1

    # Where results go
    log_every: int = 1_000  # sequences between log lines
    log_path: str = "results/copy/log.csv"
    checkpoint_path: str = "results/copy/lstm.pt"


def train(config: TrainConfig) -> LSTM:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTM().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

    steps = config.total_sequences // config.batch_size
    log_every_steps = max(1, config.log_every // config.batch_size)
    print(f"training on {device}: {model.num_parameters():,} parameters, {steps:,} steps")

    os.makedirs(os.path.dirname(config.log_path), exist_ok=True)
    log = open(config.log_path, "w")
    log.write("sequences,cost_bits\n")
    costs = []
    start = time.time()

    for step in range(1, steps + 1):
        x, target = copy_batch(config.batch_size)
        x, target = x.to(device), target.to(device)
        length = target.shape[1]

        logits = model(x)[:, length + 1:]  # only the recall steps are scored
        loss = F.binary_cross_entropy_with_logits(logits, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_value)
        optimizer.step()

        # The paper's metric: the whole sequence's cost in bits (mean loss per bit × number of bits, in log base 2)
        costs.append(loss.item() * target[0].numel() / math.log(2))

        if step % log_every_steps == 0:
            sequences = step * config.batch_size
            cost = sum(costs) / len(costs)
            costs = []
            log.write(f"{sequences},{cost:.4f}\n")
            log.flush()
            torch.save(model.state_dict(), config.checkpoint_path)
            minutes = (time.time() - start) / 60
            print(f"sequences {sequences:>9,}  cost (bits) {cost:6.2f}  {minutes:5.1f} min")

    log.close()
    return model
