"""Learning curves, which every task draws the same way, and the per-task figures.

Figure 3 of the paper (Figure 7 for repeat copy): cost per sequence in bits against sequences
seen. Anything that depends on what the task looks like - the copy task's generalisation grid,
its memory-use trace - lives in the task's own plots module and is reached through TASK_PLOTS.
"""

import csv
import json
import os

import matplotlib.pyplot as plt

from src.tasks.associative_recall import plots as associative_recall_plots
from src.tasks.copy import plots as copy_plots

# Task name -> the module drawing that task's own figures. A module may define
# plot_generalisation(model_name, checkpoint_path, out_path) and
# plot_memory_use(model_name, checkpoint_path, out_path, length); a task with neither still
# trains and still gets its learning curves, which every task shares.
TASK_PLOTS = {"copy": copy_plots, "associative-recall": associative_recall_plots}


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
    stopped = []

    for model, log_path in log_paths.items():
        if not os.path.exists(log_path):
            continue
        thousands, cost = read_curve(log_path, chunk)
        if not thousands:
            continue
        longest = max(longest, thousands[-1])

        label, colour, marker = PAPER_STYLE[model]
        planned = planned_sequences(log_path) / 1000
        for ax in (full, paper):
            ax.plot(thousands, cost, marker=marker, color=colour, markersize=3.5, linewidth=1, label=label)
            # A run stopped once it had clearly converged is continued to the length it was asked
            # for, dashed, so it can be read against the runs that finished. Dashed because it is
            # drawn rather than measured: the last value held flat, it was not observed to.
            if planned - thousands[-1] > 0.05 * planned:
                ax.plot([thousands[-1], planned], [cost[-1], cost[-1]],
                        color=colour, linewidth=1, linestyle=(0, (4, 3)))
                stopped.append(f"{label} stopped at {thousands[-1]:,.0f}k")

    for ax in (full, paper):
        # The paper's Figure 3 runs to 1000k because that is how long it trained; ours ends where
        # the run ends, so the curve fills the axis instead of hugging the left edge.
        ax.set_xlim(0, longest)
        ax.set_xlabel("sequence number (thousands)")
        ax.set_ylabel("cost per sequence (bits)")
        ax.legend(frameon=False, fontsize=9)
    if stopped:
        fig.text(0.5, 0.005, "dashed: " + "; ".join(sorted(set(stopped))) + ", held flat thereafter",
                 ha="center", fontsize=8, color="0.4")
    full.set_ylim(bottom=0)
    full.set_title("Whole run")
    paper.set_ylim(0, 10)
    paper.set_title("Same scale as the paper's Figure 3")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
