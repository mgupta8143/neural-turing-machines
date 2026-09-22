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
uv run main.py demo

# Train on the copy task (1M sequences as in the paper; log and model saved to results/copy/)
uv run main.py train
uv run main.py train --sequences 50000 --batch-size 16 --learning-rate 1e-4   # quicker run

# Draw the learning curve (Figure 3) and generalisation plot (Figure 5) into figures/
uv run main.py plot
```

### On Google Colab (GPU)

[Open colab.ipynb in Colab](https://colab.research.google.com/github/mgupta8143/neural-turing-machines/blob/main/colab.ipynb),
switch the runtime to a T4 GPU, and run the cells. Training runs in the background and you can
plot at any point.

## Results

The full 1M-sequence run hasn't been done yet. When it has, `uv run main.py plot` writes both
figures to `figures/`; commit them and they'll show up here.

**Learning curve** (paper Figure 3, LSTM only): the paper's LSTM goes below 10 bits at around
60–70k sequences, is at about 2 bits by 200k, and settles near 0.5 bits.

<!-- ![Copy learning curve](figures/copy_learning_curve.png) -->

**Generalisation** (paper Figure 5): trained on lengths 1–20, tested on 10, 20, 30, 50 and 120.
The paper's LSTM copies up to length 20 almost perfectly and fails on longer sequences.

<!-- ![Copy generalisation](figures/copy_generalisation.png) -->

## Layout

```
main.py                    command line: demo / train / plot
colab.ipynb                run training and plots on a Colab GPU
src/models/lstm.py         LSTM baseline: nn.LSTM (3 × 256) + linear readout to 8 bits
src/tasks/copy/data.py     copy_batch() -> x (B, 2L+1, 9), target (B, L, 8)
src/tasks/copy/train.py    training loop; the paper's settings live in TrainConfig
src/tasks/copy/plots.py    Figure 3 (learning curve) and Figure 5 (generalisation)
figures/                   saved plots shown in this README
results/                   training logs and model checkpoints (not committed)
```

## Roadmap

- [x] Copy task data
- [x] LSTM baseline model
- [x] Training loop matching the paper's settings
- [x] Log training cost and save checkpoints
- [x] Learning curve plot (paper Figure 3)
- [x] Generalisation plot for lengths 10, 20, 30, 50, 120 (paper Figure 5)
- [ ] Full 1M-sequence training run and results
