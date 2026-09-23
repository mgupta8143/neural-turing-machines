"""The paper's copy-task figures, LSTM only.

Figure 3: learning curve, cost per sequence (bits) against sequences seen.
Figure 5: targets and outputs for test lengths longer than the training range of 1-20.
"""

import csv

import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec

from src.models.build import build_model
from src.tasks.copy.data import copy_batch


def load_model(model_name: str, checkpoint_path: str):
    model = build_model(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def plot_learning_curve(log_path: str, out_path: str, label: str = "LSTM", chunk: int = 10_000):
    with open(log_path) as f:
        rows = list(csv.DictReader(f))

    # Average the log into chunks of 10k sequences, like the spacing of the paper's dots.
    # Single log lines are noisy because every batch has random sequence lengths.
    chunks = {}
    for row in rows:
        end = -(-int(row["sequences"]) // chunk) * chunk  # round up to the chunk boundary
        chunks.setdefault(end, []).append(float(row["cost_bits"]))
    thousands = [end / 1000 for end in chunks]
    cost = [sum(values) / len(values) for values in chunks.values()]

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


def plot_generalisation(model_name: str, checkpoint_path: str, out_path: str):
    model = load_model(model_name, checkpoint_path)

    # The page is a grid with one column per timestep, so each panel's width matches its length.
    # Top: lengths 10, 20, 30, 50 side by side. Bottom: length 120 across the whole width.
    gap = 3
    fig = plt.figure(figsize=(12, 4.5))
    grid = GridSpec(5, 126, figure=fig, height_ratios=[1, 1, 0.5, 1, 1], hspace=0.15, wspace=0)

    panels = []  # (length, first grid row, grid columns)
    column = 0
    for length in [10, 20, 30, 50]:
        panels.append((length, 0, slice(column, column + length)))
        column += length + gap
    panels.append((120, 3, slice(0, 120)))

    for length, row, columns in panels:
        x, target = copy_batch(1, length, length)
        with torch.no_grad():
            outputs = torch.sigmoid(model(x)[0, length + 1:])

        # Transposed so time runs left to right and the 8 bits run top to bottom
        for offset, (label, image) in enumerate([("Targets", target[0].T), ("Outputs", outputs.T)]):
            ax = fig.add_subplot(grid[row + offset, columns])
            im = ax.imshow(image, cmap="jet", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if columns.start == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=12)

    fig.colorbar(im, cax=fig.add_subplot(grid[:, 124:]))
    fig.text(0.13, 0.03, "Time  ⟶", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_memory_use(model_name: str, checkpoint_path: str, out_path: str, length: int = 20):
    """Figure 6 of the paper: what the heads did, timestep by timestep.

    Left column: the input, the vectors written to memory, and the write weightings.
    Right column: the output, the vectors read back, and the read weightings.
    A network that has learned to copy shows a diagonal stripe in both weightings: the write head
    steps along memory while reading the input, and the read head retraces the same locations.
    """
    model = load_model(model_name, checkpoint_path)
    x, _ = copy_batch(1, length, length)
    trace = {}
    with torch.no_grad():
        outputs = torch.sigmoid(model(x, trace=trace)[0])

    write_w = torch.stack(trace["write_weightings"], dim=1)  # (locations, timesteps)
    read_w = torch.stack(trace["read_weightings"], dim=1)
    adds = torch.stack(trace["adds"], dim=1)  # (memory width, timesteps)
    reads = torch.stack(trace["reads"], dim=1)

    # Show only the locations the heads actually used, as the paper does
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
    axes[2, 0].set_xlabel("Time →")
    axes[2, 1].set_xlabel("Time →")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
