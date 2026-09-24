"""Learning curves, which every task draws the same way, and the per-task figures.

Figure 3 of the paper (Figure 7 for repeat copy): cost per sequence in bits against sequences
seen. Anything that depends on what the task looks like - the copy task's generalisation grid,
its memory-use trace - lives in the task's own plots module and is reached through TASK_PLOTS.
"""

import csv
import os

import matplotlib.pyplot as plt

from src.tasks.copy import plots as copy_plots

# Task name -> the module drawing that task's own figures. A module may define
# plot_generalisation(model_name, checkpoint_path, out_path) and
# plot_memory_use(model_name, checkpoint_path, out_path, length); a task with neither still
# trains and still gets its learning curves.
TASK_PLOTS = {"copy": copy_plots}


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


def plot_all_learning_curves(log_paths: dict, out_path: str, chunk: int = 10_000):
    """The paper's Figure 3: every model on one pair of axes."""
    fig, (full, paper) = plt.subplots(1, 2, figsize=(12, 4.5))
    longest = 1

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
        ax.set_xlim(0, max(1000, longest))
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
