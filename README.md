# neural-turing-machines

A PyTorch reproduction of the LSTM baseline on the **copy task** from
[Neural Turing Machines](https://arxiv.org/abs/1410.5401) (Graves, Wayne & Danihelka, 2014).

## The copy task

The network reads a sequence of random 8-bit vectors, then a delimiter flag, and must then
output the same sequence from memory while receiving no further input.

- Sequence length `L` is random between 1 and 20
- Each timestep's input has 9 channels: 8 data bits + 1 delimiter channel
- The input is `L` data steps, 1 delimiter step, then `L` blank steps, for `2L + 1` in total
- Only the last `L` outputs are scored against the target

## Model

| | Paper (Table 3) | This repo |
|---|---|---|
| LSTM | 3 layers × 256 | 3 layers × 256 (`nn.LSTM`) |
| Output layer | sigmoid | `Linear(256, 8)`, sigmoid applied in the loss |
| Loss | cross-entropy, reported in bits per sequence | same |
| Optimizer | RMSProp, momentum 0.9 | same |
| Learning rate | 3 × 10⁻⁵ | same |
| Gradient clipping | elementwise to (−10, 10) | same |
| Parameters | 1,352,969 | 1,328,136 |

The parameter count differs slightly because Graves' LSTM variant has details `nn.LSTM` doesn't
include.

## Usage

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync

# Run one batch through the untrained model and print the shapes
uv run python -m src.tasks.copy

# Train on the copy task
uv run main.py
```

## Layout

```
src/tasks/copy.py    copy task data: copy_batch() -> x (B, 2L+1, 9), target (B, L, 8)
src/models/lstm.py   LSTM baseline: nn.LSTM + linear readout
main.py              training loop
```

## Roadmap

- [x] Copy task data
- [x] LSTM baseline model
- [x] Training loop matching the paper's settings
- [ ] Log training cost and save checkpoints
- [ ] Learning curve (paper Figure 3)
- [ ] Generalisation to lengths 10, 20, 30, 50, 120 (paper Figure 5)
