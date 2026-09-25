"""Learning curves, which every task draws the same way, and the per-task figures.

Figure 3 of the paper (Figure 7 for repeat copy): cost per sequence in bits against sequences
seen. Anything that depends on what the task looks like - the copy task's generalisation grid,
its memory-use trace - lives in the task's own plots module and is reached through TASK_PLOTS.
"""

import csv
import json
import os

import matplotlib.pyplot as plt

from src.tasks.copy import plots as copy_plots
from src.tasks.priority_sort import plots as priority_sort_plots

# Task name -> the module drawing that task's own figures. A module may define
# plot_generalisation(model_name, checkpoint_path, out_path) and
# plot_memory_use(model_name, checkpoint_path, out_path, length); a task with neither still
# trains and still gets its learning curves.
TASK_PLOTS = {"copy": copy_plots, "priority-sort": priority_sort_plots}


def task_plot(task_name: str, figure: str):
    """The named figure function for a task, or None if the task does not draw that one."""
    return getattr(TASK_PLOTS.get(task_name), figure, None)


def read_curve(log_path: str, chunk: int = 0):
    """A training log averaged into chunks of `chunk` sequences: (thousands of sequences, cost).

    Single log lines are noisy, since every batch draws a random sequence length and longer
    sequences cost more bits. The paper's dots are about 10k sequences apart, but a chunk that
    size hides everything in a short run, so chunk 0 aims for ~50 points and caps it at 10k.
    """
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], []
    if not chunk:
        chunk = min(10_000, max(1_000, int(rows[-1]["sequences"]) // 50))
    chunks = {}
    for row in rows:
        end = -(-int(row["sequences"]) // chunk) * chunk  # round up to the chunk boundary
        chunks.setdefault(end, []).append(float(row["cost_bits"]))
    thousands = [end / 1000 for end in chunks]
    return thousands, [sum(values) / len(values) for values in chunks.values()]


def plot_learning_curve(log_path: str, out_path: str, label: str = "LSTM", chunk: int = 0):
    thousands, cost = read_curve(log_path, chunk)

    fig, (full, paper) = plt.subplots(1, 2, figsize=(12, 4))
    for ax in (full, paper):
        ax.plot(thousands, cost, "o-", color="#1f3f99", markersize=3, linewidth=1, label=label)
        ax.set_xlim(0, max(1000, thousands[-1]))
        ax.set_xlabel("sequence number (thousands)")
        ax.set_ylabel("cost per sequence (bits)")
        ax.legend(frameon=False)
    full.set_ylim(0, max(cost) * 1.05)
    full.set_title("Whole run")
    paper.set_ylim(0, 10)
    paper.set_title("Same scale as the paper's Figure 3")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


PAPER_STYLE = {  # colours and markers of the paper's Figure 3
    "lstm": ("LSTM", "#1f3f99", "o"),
    "ntm-lstm": ("NTM with LSTM Controller", "#1f8a3f", "s"),
    "ntm-ff": ("NTM with Feedforward Controller", "#c81e1e", "^"),
}


def planned_sequences(log_path: str) -> int:
    """How many sequences the run that wrote this log was asked for, from its run.json."""
    run = os.path.join(os.path.dirname(log_path), "run.json")
    if os.path.exists(run):
        with open(run) as f:
            return json.load(f).get("total_sequences", 0)
    return 0


def plot_all_learning_curves(log_paths: dict, out_path: str, chunk: int = 10_000):
    """The paper's Figure 3: every model on one pair of axes.

    The x-axis runs to the length the runs were asked for, not to where they have got to, so a
    plot drawn while training is still going keeps the same axes as the finished one.
    """
    fig, (full, paper) = plt.subplots(1, 2, figsize=(12, 4.5))
    longest = max((planned_sequences(p) for p in log_paths.values()), default=0) / 1000 or 1

    for model, log_path in log_paths.items():
        if not os.path.exists(log_path):
            continue
        thousands, cost = read_curve(log_path, chunk)
        if not thousands:
            continue
        longest = max(longest, thousands[-1])

        label, colour, marker = PAPER_STYLE[model]
        for ax in (full, paper):
            ax.plot(thousands, cost, marker=marker, color=colour, markersize=3.5, linewidth=1, label=label)

    for ax in (full, paper):
        # The paper's Figure 3 runs to 1000k because that is how long it trained; ours ends where
        # the run ends, so the curve fills the axis instead of hugging the left edge.
        ax.set_xlim(0, longest)
        ax.set_xlabel("sequence number (thousands)")
        ax.set_ylabel("cost per sequence (bits)")
        ax.legend(frameon=False, fontsize=9)
    full.set_ylim(bottom=0)
    full.set_title("Whole run")
    paper.set_ylim(0, 10)
    paper.set_title("Same scale as the paper's Figure 3")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_memory_use(task_name: str, model_name: str, checkpoint_path: str, out_path: str):
    """Figure 6 of the paper, for any task: what the heads did, timestep by timestep.

    Left column: the input, the vectors written to memory, and the write weightings. Right column:
    the output, the vectors read back, and the read weightings. A model that has learned to use
    memory shows sharp weightings that move over time; one that has not shows a smear.

    The copy task overrides this with its own version, which pins the sequence length so the
    diagonal is easy to compare against the paper's figure.
    """
    import torch

    from src.models.build import load_model
    from src.tasks.registry import get_task

    task = get_task(task_name)
    model = load_model(model_name, checkpoint_path, task, task_name)
    x, _, _ = task.batch(1)
    trace = {}
    with torch.no_grad():
        outputs = torch.sigmoid(model(x, trace=trace)[0])

    write_w = torch.stack(trace["write_weightings"], dim=1)
    read_w = torch.stack(trace["read_weightings"], dim=1)
    adds = torch.stack(trace["adds"], dim=1)
    reads = torch.stack(trace["reads"], dim=1)

    used = (write_w.max(dim=1).values + read_w.max(dim=1).values) > 0.01
    rows = used.nonzero().flatten()
    window = slice(max(0, int(rows.min()) - 2), int(rows.max()) + 3) if len(rows) else slice(0, 40)

    fig, axes = plt.subplots(3, 2, figsize=(11, 7), gridspec_kw={"height_ratios": [1, 1.3, 2.6]})
    panels = [
        ("Inputs", x[0].T, "gray", axes[0, 0]),
        ("Outputs", outputs.T, "gray", axes[0, 1]),
        ("Adds", adds, "jet", axes[1, 0]),
        ("Reads", reads, "jet", axes[1, 1]),
        ("Write weightings", write_w[window], "gray", axes[2, 0]),
        ("Read weightings", read_w[window], "gray", axes[2, 1]),
    ]
    for title, image, cmap, ax in panels:
        ax.imshow(image, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[2, 0].set_ylabel("Location")
    for ax in (axes[2, 0], axes[2, 1]):
        ax.set_xlabel("Time \u2192")
    fig.suptitle(f"{task_name}: {model_name} memory use", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
