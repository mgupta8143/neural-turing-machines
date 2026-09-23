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

## Models

| | Paper | This repo |
|---|---|---|
| `lstm` | 3 layers x 256, 1,352,969 parameters | 1,328,136 |
| `ntm-ff` | NTM, feed-forward controller, 17,162 | 16,096 |
| `ntm-lstm` | NTM, LSTM controller, 67,561 | 62,660 |

All three share the copy task, the training loop and the plots, and take
`(batch, 2L+1, 9)` to `(batch, 2L+1, 8)`.

### Speed and stability notes for the NTM

The NTM runs a Python loop over timesteps with many small operations per step, so a GPU spends
its time launching kernels rather than computing: on the copy task it is several times slower
than a CPU. Training therefore defaults to the CPU for `ntm-ff` and `ntm-lstm`, and to the GPU
for the LSTM baseline, whose whole sequence runs in one cuDNN kernel. Override with `--device`.
`--compile` gives about 1.7x per step, after a warmup that recompiles for each sequence length.

The NTM also clips gradients by norm rather than by value. The paper clips each component to
(-10, 10), which bounds each number but not the size of the update: NTM gradient norms spike to
several hundred times their median, and with value clipping those spikes produce an enormous step
that destroys what the model has learned. The LSTM baseline keeps the paper's value clipping.

Common settings from the paper (Section 4.6 and Tables 1-3): RMSProp with momentum 0.9,
gradients clipped elementwise to (-10, 10), cross-entropy reported in bits per sequence, and a
learning rate of 3e-5 for the LSTM baseline or 1e-4 for the NTMs. The NTMs use a 128 x 20
memory, a controller of 100 units, and one read and one write head.

## Usage

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync

# Run one batch through an untrained model and print the shapes
uv run main.py demo --model ntm-ff

# Train on the copy task (1M sequences as in the paper; results go to results/copy/<model>/)
uv run main.py train --model lstm       # the 3 x 256 LSTM baseline
uv run main.py train --model ntm-ff     # NTM, feed-forward controller
uv run main.py train --model ntm-lstm   # NTM, LSTM controller
uv run main.py train --model ntm-ff --sequences 50000 --batch-size 16   # quicker run
uv run main.py train --model ntm-ff --device cpu --compile                # NTM speed options

# Draw the learning curve (Figure 3) and generalisation plot (Figure 5) into figures/
uv run main.py plot --model ntm-ff

# The paper's Figure 6: the write and read weightings over time (NTM only)
uv run main.py memory --model ntm-ff --length 20   # figures/ntm-ff_memory_length20.png

# Give a trained model your own 8-bit vectors, or a random sequence, and see what it copies back
uv run main.py try --model ntm-ff 10110010 01100101 11110000
uv run main.py try --model ntm-ff --random 30

# Train both models and keep every figure refreshed; safe to leave running overnight
bash overnight.sh

# Checks for the NTM's memory, addressing and gradients
uv run pytest
```

### On Google Colab (GPU)

[Open colab.ipynb in Colab](https://colab.research.google.com/github/mgupta8143/neural-turing-machines/blob/main/colab.ipynb),
switch the runtime to a T4 GPU, and run the cells. Training runs in the background and you can
plot at any point.

## Results

Full write-up with the figures: [results.md](results.md).

The LSTM baseline has been trained for 1M sequences (batch size 16, learning rate 1e-4). It ends
at 0.9 bits per sequence against the paper's ~0.5, copies lengths 10 and 20 almost perfectly, and
fails past length 20 the way the paper's does. NTM runs are in progress.

## Layout

```
main.py                    command line: demo / train / plot / memory / try
colab.ipynb                run training and plots on a Colab GPU
src/models/lstm.py         LSTM baseline: nn.LSTM (3 × 256) + linear readout to 8 bits
src/models/build.py        model names -> models, and the paper's learning rate for each
src/models/ntm/memory.py   read, write and the four addressing stages (paper Figure 2)
src/models/ntm/heads.py    ReadHead and WriteHead: one Linear -> addressing -> read or write
src/models/ntm/controllers.py  FeedForwardController and LSTMController, same interface
src/models/ntm/ntm.py      controller + heads + memory + output layer, one timestep at a time
tests/test_ntm.py          checks for the memory, addressing and gradients
src/tasks/copy/data.py     copy_batch() -> x (B, 2L+1, 9), target (B, L, 8)
src/tasks/copy/train.py    training loop; the paper's settings live in TrainConfig
src/tasks/copy/plots.py    Figure 3 (learning curve) and Figure 5 (generalisation)
src/tasks/copy/try_it.py   run the trained model on a sequence you type in
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
- [x] Full 1M-sequence run (batch size 16) and results
- [ ] Paper-faithful run at batch size 1
- [x] NTM: memory, addressing, heads, both controllers
- [ ] NTM copy-task runs and figures
- [ ] Memory-use plot (paper Figure 6)
