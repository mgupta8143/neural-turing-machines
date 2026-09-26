# neural-turing-machines

A from-scratch PyTorch reproduction of [Neural Turing Machines](https://arxiv.org/abs/1410.5401)
(Graves, Wayne & Danihelka, 2014). Three models: the NTM with a feed-forward controller, the NTM
with an LSTM controller, and the plain LSTM the paper uses as a baseline.

The write-up is in [results.md](results.md), and the figures are in `figures/copy/` and
`figures/associative-recall/`.

## Getting started

You need [uv](https://docs.astral.sh/uv/) and nothing else.

```sh
git clone https://github.com/mgupta8143/neural-turing-machines
cd neural-turing-machines
uv sync
```

Train the feed-forward NTM on the copy task and watch it learn:

```sh
uv run main.py train --model ntm-ff --sequences 20000
```

It starts around 84 bits, which is chance, and should be near zero inside ten thousand sequences.
That takes a few minutes on a laptop. Then draw its figures:

```sh
uv run main.py plot --model ntm-ff
uv run main.py memory --model ntm-ff --length 40
```

Everything else:

```sh
uv run main.py all                                  # train all three and keep the figures current
uv run main.py compare                              # all three learning curves on one plot
uv run main.py demo --model ntm-ff                  # one batch through an untrained model
uv run pytest                                       # 29 checks
```

Every command takes `--task`. It defaults to copy; `--task associative-recall` runs the other one.

## Running on a GPU

Most of the results here were trained on rented A10Gs through [Modal](https://modal.com), which
is a good fit for this: you pay by the second, the runs are short, and nothing has to be set up
on your own machine beyond an account.

```sh
uv run --with modal modal run --detach src/remote.py::train --model ntm-ff --sequences 500000
uv run --with modal modal run src/remote.py::fetch
```

Use `--detach` for anything long, or the job dies with the terminal that launched it. Training
writes into a Modal volume as it goes, so `fetch` will pull down a run that is still in progress.

Worth knowing before you rent anything: the NTM is slower on a GPU than on a CPU if you run it
naively, because it launches a few hundred tiny kernels per sequence and spends all its time on
launch overhead rather than arithmetic. We capture the whole training step as a CUDA graph, one
per distinct sequence length, which is what makes the GPU worth paying for. The LSTM baseline has
no such problem and is much faster on a GPU either way.

## The tasks

Two of the paper's five are implemented, the ones that make the sharpest claims.

**Copy** (Section 4.1) reads `L` random 8-bit vectors and a delimiter, then has to emit the same
vectors again. Training draws `L` from 1 to 20.

**Associative recall** (Section 4.3) reads two to six items, each three six-bit vectors, then one
of those items again as a query. It has to emit whichever item followed the query in the list.

The rest of the code does not know which task it is running, so a third one is mostly a matter of
writing its data module. `src/tasks/registry.py` says what that module has to provide.

## Results in brief

Parameter counts, ours against the paper's Tables 1 to 3:

| | copy | paper | associative recall | paper |
|---|---|---|---|---|
| `lstm` | 1,328,136 | 1,352,969 | 1,326,598 | 1,344,518 |
| `ntm-ff` | 16,096 | 17,162 | 123,046 | 146,845 |
| `ntm-lstm` | 65,696 | 67,561 | 65,054 | 70,330 |

Ours come out consistently smaller, and we could not find a reading of the tables that closes the
gap. The published numbers do not appear to be reconstructible from the published architectures,
and in one place Table 2 contradicts itself. There is more on this in results.md.

Training follows Section 4.6: RMSProp in the form Graves (2013) describes, momentum 0.9, one
sequence per update, and the per-task learning rates from the tables.

Each run writes to `results/<task>/<model>/`. Alongside the checkpoint and the cost log there is a
`run.json` with the seed, learning rate, device, PyTorch version and git commit, so you can always
work out which settings produced a given figure. Runs repeat exactly on the same machine and
PyTorch build; on different hardware you get the same curve but not the same digits.

## Code

```
main.py                  the command line
src/train.py             training loop, CUDA-graph capture, checkpoints, logging
src/probe.py             diagnostics logged beside the cost
src/plots.py             learning curves, shared by every task
src/remote.py            training on a Modal GPU and fetching the results
src/models/ntm/          memory and addressing, heads, controllers, the model
src/models/lstm/         the baseline, which is an nn.LSTM and a readout
src/models/build.py      per-task architectures and learning rates
src/tasks/<task>/        data.py makes the task, plots.py draws its figures
```

If you only read one file, read `src/models/ntm/memory.py`. Reading, writing and the four
addressing stages of the paper's Figure 2 are each a separate function there, and the tests check
them against the equations.

## Things the paper does not tell you

These four decide whether it trains at all, and none of them is in the paper.

Section 4.6 says RMSProp "in the form described in (Graves, 2013)", which means centered, decay
0.95, and a damping term of 1e-4. If you use PyTorch's defaults instead, the LSTM-controller NTM
reaches zero cost and then drifts back off it later in the run.

The paper clips each gradient component to (-10, 10). NTM gradient norms spike to hundreds of
times their median, and clipping componentwise turns one of those spikes into an update large
enough to destroy the addressing the model has learned. Clip the norm instead.

The starting memory has to break symmetry. If every location holds the same values, every location
gets the same gradient, and the 128 locations never differentiate.

Equation 9 sharpens the weighting by raising it to a power. Written literally, small weights
underflow float32 to zero and the head ends up attending to nothing. `softmax(gamma * log w)` is
the same expression and does not underflow.

There is also one that decides how fast it trains rather than whether it trains, and it surprised
us: the range of sequence lengths you train on. Copy converges in about 7,500 sequences when
lengths are drawn from 1 to 20, and does not converge at all when they are drawn from 1 to 10.
That held across three seeds.
