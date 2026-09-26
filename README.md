# neural-turing-machines

[Neural Turing Machines](https://arxiv.org/abs/1410.5401) (Graves, Wayne & Danihelka, 2014)
reproduced from scratch in PyTorch: the NTM with a feed-forward and an LSTM controller, against
the paper's LSTM baseline.

**[results.md](results.md) has the write-up**; the figures are in `figures/<task>/`.

In short: on both tasks the NTMs reach zero cost while the LSTM baseline never does. On copy they
then copy sequences six times longer than anything they trained on, at 0% bit error. On
associative recall they beat the paper's own generalisation numbers at every point.

## Tasks

Two of the paper's five are implemented, chosen because they make the sharpest claims:

- **copy** (Section 4.1) — read `L` 8-bit vectors, then emit them again. Trained on `L` ∈ 1..20.
- **associative recall** (Section 4.3) — read 2 to 6 three-vector items, then one of them again
  as a query; emit the item that followed it.

Adding another is a module with five names — `INPUT_SIZE`, `OUTPUT_SIZE`, `TIMESTEPS`, `batch()`
and `PROBE_CASES` — registered in `src/tasks/registry.py`, plus a settings entry in
`src/models/build.py`. Nothing else changes: the models, training loop, CUDA-graph path and
learning-curve plots are all task-agnostic. `src/tasks/registry.py` documents the interface.

## Models

Parameter counts, ours against the paper's Tables 1–3:

| | copy | paper | associative recall | paper |
|---|---|---|---|---|
| `lstm` | 1,328,136 | 1,352,969 | 1,326,598 | 1,344,518 |
| `ntm-ff` | 16,096 | 17,162 | 123,046 | 146,845 |
| `ntm-lstm` | 65,696 | 67,561 | 65,054 | 70,330 |

We come out consistently under. The published counts are not reconstructible from the published
architectures — `results.md` has the details, including where Table 2 contradicts itself.

Training follows Section 4.6: RMSProp in the Graves (2013) form, momentum 0.9, one sequence per
update, and the per-task learning rates from the tables.

## Usage

Requires [uv](https://docs.astral.sh/uv/). Every command takes `--task`, defaulting to copy.

```sh
uv sync
uv run pytest                                       # 29 checks, incl. addressing vs the paper

uv run main.py all                                  # train all three, redrawing figures as it goes
uv run main.py train --model ntm-ff                 # one model
uv run main.py plot --model ntm-ff                  # learning curve and generalisation
uv run main.py compare                              # all three curves on one plot
uv run main.py memory --model ntm-ff --length 40    # what the heads read and wrote
uv run main.py demo --model ntm-ff                  # one batch through an untrained model
```

On a rented A10G, which is worth it for the LSTM and for long NTM runs:

```sh
uv run --with modal modal run --detach src/remote.py::train --model ntm-ff --sequences 500000
uv run --with modal modal run src/remote.py::fetch    # pull results mid-run
```

Results land in `results/<task>/<model>/`: `log.csv`, `diagnostics.csv`, the checkpoint, and a
`run.json` recording seed, learning rate, device, PyTorch version and git commit, so any figure
traces back to the settings that made it. Runs repeat exactly on the same machine and build;
across architectures you reproduce the curve, not the digits.

## Codebase

```
main.py                  the command line
src/train.py             training loop, checkpoints, logging
src/graphs.py            the whole step captured as one CUDA graph, per sequence length
src/probe.py             diagnostics logged beside the cost: head behaviour, error out of range
src/plots.py             learning curves, shared by every task
src/remote.py            train on a Modal GPU, fetch the results
src/models/ntm/          memory and addressing, heads, controllers, and the model itself
src/models/lstm.py       the baseline: an nn.LSTM and a readout
src/models/build.py      per-task architectures and learning rates, from Tables 1-3
src/tasks/<task>/        data.py generates the task, plots.py draws its paper figures
```

`src/models/ntm/memory.py` is the interesting file: read, write, and the four addressing stages
of the paper's Figure 2, each as its own function, tested against the equations.

## What the paper leaves out

Four details decide whether this trains at all:

- **RMSProp as Graves (2013) defines it** — centered, decay 0.95, damping 1e-4. PyTorch's
  defaults instead make the LSTM-controller NTM converge and then drift back off zero.
- **Clip the gradient norm, not each component.** NTM gradient norms spike to hundreds of times
  their median, and the paper's elementwise clipping turns such a spike into an update that
  destroys the learned addressing.
- **The starting memory must break symmetry.** With identical rows every location has identical
  gradients and the 128 locations never differentiate.
- **Sharpening underflows if written literally.** Equation 9 raises the weighting to a power;
  `softmax(gamma * log w)` is the same expression and survives float32.

And one that decides whether it trains *quickly*: the range of training lengths. Copy converges in
7,500 sequences on lengths 1–20 and does not converge at all on 1–10, across three seeds.
