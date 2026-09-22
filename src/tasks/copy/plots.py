"""The paper's copy-task figures, LSTM only.

Figure 3: learning curve, cost per sequence (bits) against sequences seen.
Figure 5: targets and outputs for test lengths longer than the training range of 1-20.
"""

import csv

import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec

from src.models.lstm import LSTM
from src.tasks.copy.data import copy_batch


def plot_learning_curve(log_path: str, out_path: str):
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    thousands = [int(row["sequences"]) / 1000 for row in rows]
    cost = [float(row["cost_bits"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thousands, cost, "o-", color="#1f3f99", markersize=3, linewidth=1, label="LSTM")
    ax.set_xlim(0, max(1000, thousands[-1]))
    ax.set_ylim(0, 10)  # same scale as the paper
    ax.set_xlabel("sequence number (thousands)")
    ax.set_ylabel("cost per sequence (bits)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_generalisation(checkpoint_path: str, out_path: str):
    model = LSTM()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

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
