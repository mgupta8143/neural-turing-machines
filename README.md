# neural-turing-machines

The copy task from [Neural Turing Machines](https://arxiv.org/abs/1410.5401) (Graves, Wayne &
Danihelka, 2014), reproduced from scratch: the LSTM baseline and the NTM with a feed-forward and
an LSTM controller.

## The copy task

The network reads `L` random 8-bit vectors, then a delimiter, then `L` blank steps during which it
must output the same vectors again. `L` is random between 1 and 20, so an example is `2L + 1`
timesteps and only the last `L` outputs are scored. Cost is cross-entropy in bits per sequence:
84 is random guessing, 0 is a perfect copy.

## Models

| | Parameters here | Paper |
|---|---|---|
| `lstm` — 3 layers x 256 | 1,328,136 | 1,352,969 |
| `ntm-ff` — feed-forward controller | 16,096 | 17,162 |
| `ntm-lstm` — LSTM controller | 62,660 | 67,561 |

Both NTMs use a 128 x 20 memory, a controller of 100 units, and one read and one write head. All
three share the task, the training loop and the plots, mapping `(batch, 2L+1, 9)` to
`(batch, 2L+1, 8)`.

From the paper (Section 4.6, Tables 1–3): RMSProp with momentum 0.9, one sequence per update,
gradients clipped to (−10, 10), learning rate 3e-5 for the LSTM and 1e-4 for the NTMs.

## Usage

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest                  # checks for the memory, addressing and gradients

uv run main.py all             # train all three, redrawing every figure every 15 minutes
uv run main.py all --sequences 100000 --refresh 300 --seed 1

uv run main.py demo --model ntm-ff                 # one batch through an untrained model
uv run main.py train --model ntm-ff                # one model on its own
uv run main.py plot --model ntm-ff                 # learning curve and generalisation
uv run main.py compare                             # all three curves on one plot
uv run main.py memory --model ntm-ff --length 40   # write and read weightings over time
```

Every command takes `--task`, which defaults to copy: `--task associative-recall` runs the other one.

On a GPU each training step replays as a single CUDA graph, one captured per sequence length,
which is what makes a GPU worth using here: eager, the NTM launches a few hundred tiny kernels per
sequence and runs several times slower on a GPU than on a laptop CPU. Measured at batch 1:

| | Mac CPU | A10G eager | A10G + CUDA graphs |
|---|---|---|---|
| `ntm-ff` | 11.3 ms/seq | 81 ms/seq | 7.8 ms/seq |
| `lstm` | 18.2 ms/seq | 3.3 ms/seq | 1.0 ms/seq |

`modal_run.py` trains on a rented A10G and downloads the results:

```sh
uv run --with modal modal run modal_run.py --model lstm
```

Training writes to `results/copy/<model>/`, figures to `figures/`. The NTMs train on the CPU by
default: their per-timestep loop of small operations is faster there than on a GPU, which spends
its time launching kernels. `--device` overrides, and `--compile` is about 1.7x per step after a
slow warmup.

## Reproducing a run

The repository pins everything that decides the numbers, and each run records the rest.

```sh
uv sync --frozen    # exact package versions from uv.lock, Python 3.12 from .python-version
uv run main.py all  # seed 0, batch 1, the paper's learning rates
```

Every run writes `results/copy/<model>/run.json` with its seed, batch size, learning rate,
clipping, parameter count, device, PyTorch version and the git commit, so a figure can always be
traced back to the settings that produced it. Pass `--seed` to `train` or `all` to vary it;
`copy_batch` draws its lengths and bits from that seed, as do the initial weights.

Two caveats worth knowing. Runs repeat exactly on the same machine and PyTorch build, but floating
point accumulates differently across CPU architectures, thread counts and CUDA versions, so
another machine reproduces the curve, not the digits. And a single run is a single sample: the NTM
in particular varies between seeds in how quickly it finds the addressing solution, so compare
runs at a few seeds before concluding a change helped.

## Results

[results.md](results.md) has the figures and the comparison with the paper. In short: both NTMs
reach 0.00 bits within 10k sequences and copy perfectly at six times their training length, while
the LSTM baseline needs 500k sequences to reach 1.03 bits and is wrong on half the bits at
length 120.

## Layout

```
main.py                        command line
src/models/lstm.py             LSTM baseline
src/models/ntm/memory.py       read, write, and the four addressing stages of Figure 2
src/models/ntm/heads.py        ReadHead and WriteHead: one Linear -> weighting -> read or write
src/models/ntm/controllers.py  the two controllers, same interface
src/models/ntm/ntm.py          controller + heads + memory, one timestep at a time
src/tasks/copy/                the task, the training loop, the plots
tests/                         18 checks, including addressing against the paper's equations
```

## Notes

Things the paper leaves out that decide whether this trains:

- **Clip by norm, not by value.** The paper clips each gradient component to (−10, 10). NTM
  gradient norms spike to hundreds of times their median, and value clipping turns such a spike
  into an enormous update that destroys the learned addressing. Clipping the norm instead is the
  difference between diverging mid-run and reaching zero.
- **The starting memory has to break symmetry.** With identical memory rows and uniform weightings
  every location is interchangeable, so their gradients are identical and the 128 locations never
  differentiate. Random values fix it: on a fixed-length-5 copy, 1,200 updates reach 1.1 bits with
  random initialisation against 34 bits with constant.
- **Sharpening underflows if written literally.** Equation 9 raises the weighting to a power;
  small weights at a high power underflow float32 to zero and the head attends to nothing.
  `softmax(gamma * log w)` is the same expression and is stable.
- **The learning rates assume one sequence per update.** At batch 16 there are 16x fewer updates,
  and the LSTM-controller NTM then sits near chance at 1e-4 and needs about 1e-3.
