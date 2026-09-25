"""The priority sort task's own figures.

Figure 16: an example input sequence and the target it asks for.
Figure 17: what the heads did - write locations fitted from the priorities, the write locations
actually observed, and the read locations.

The learning curves, which look the same for every task, are in src/plots.py.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from src.models.build import load_model
from src.tasks.priority_sort import data as sort_data


def plot_generalisation(model_name=None, checkpoint_path=None, out_path="figures/priority-sort/example.png"):
    """Figure 16 of the paper: random vectors with random priorities, and the sorted subset.

    Nothing here depends on the model, so it draws with no checkpoint: it documents the task
    rather than a run. The priority is a continuous scalar sharing a sequence with binary
    channels, so it gets its own axis above each block of bits; as a ninth row of the heatmap
    its grey level would be read as a bit.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    vectors = torch.randint(0, 2, (1, sort_data.MAX_LEN, sort_data.BITS)).float()
    low, high = sort_data.PRIORITY_RANGE
    priorities = torch.rand(1, sort_data.MAX_LEN) * (high - low) + low
    target, mask = sort_data.make_target(vectors, priorities)
    kept = int(mask[0].sum())

    fig = plt.figure(figsize=(11, 5))
    grid = GridSpec(4, sort_data.MAX_LEN, figure=fig, height_ratios=[0.7, 1, 0.7, 1], hspace=0.4)
    panels = [
        (0, sort_data.MAX_LEN, priorities[0], vectors[0], "Input"),
        (2, kept, priorities[0].sort(descending=True).values[:kept], target[0][mask[0]], "Target"),
    ]
    for row, width, priority, bits, label in panels:
        strip = fig.add_subplot(grid[row, :width])
        strip.bar(range(width), priority, width=0.7, color="#1f3f99")
        strip.axhline(0, color="black", linewidth=0.5)
        strip.set_xlim(-0.5, width - 0.5)
        strip.set_ylim(low * 1.1, high * 1.1)
        strip.set_xticks([])
        strip.set_yticks([low, 0, high])
        strip.tick_params(labelsize=7)
        strip.set_ylabel("Priority", fontsize=8)

        ax = fig.add_subplot(grid[row + 1, :width])
        ax.imshow(bits.T, cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=12)
    fig.text(0.13, 0.03, "Time  ⟶", fontsize=12)
    fig.suptitle(f"{sort_data.MAX_LEN} random vectors and priorities; the {kept} highest, in order", fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_memory_use(model_name, checkpoint_path, out_path, length: int = sort_data.MAX_LEN):
    """Figure 17 of the paper: fitted write locations, observed write locations, read locations.

    The claim is that the network turns a priority into a location, so a and b and the R^2 of
    the fit are in the title: "closely match" is the whole result and should be checkable.

    Table 1 gives priority sort 8 heads, so there are 8 write weightings per timestep where the
    paper shows one. Head 0 would be arbitrary and summing them blurs eight locations into one
    smear, so this draws the head the priorities explain best - the head the hypothesis is
    about - and prints every head's R^2 underneath, where a fit that is only the best of eight
    noisy ones is visible as one. The read head is picked the same way, against the paper's
    other claim: that reads run in increasing order of location.

    Panels are scaled to their own maximum, with a colourbar, because an untrained model
    spreads its weightings over all 128 locations and on a fixed 0-1 scale is a black square.
    """
    model = load_model(model_name, checkpoint_path, sort_data, "priority-sort")
    x, _, mask = sort_data.batch(1, length, length)
    trace = {}
    with torch.no_grad():
        model(x, trace=trace)

    steps = x.shape[1]
    priorities = x[0, :length, sort_data.BITS].numpy()
    # The trace appends one entry per head per timestep, in head order
    write_w = torch.stack(trace["write_weightings"]).reshape(steps, -1, model.memory_locations).numpy()
    read_w = torch.stack(trace["read_weightings"]).reshape(steps, -1, model.memory_locations).numpy()
    output_steps = mask[0].nonzero().flatten().numpy()

    write_w, read_w = write_w[:length], read_w[output_steps]  # the input phase, then the output phase
    write_fits = [_fit(priorities, w.argmax(axis=1)) for w in write_w.transpose(1, 0, 2)]
    read_fits = [_fit(np.arange(len(output_steps)), r.argmax(axis=1)) for r in read_w.transpose(1, 0, 2)]
    head = int(np.argmax([fit[2] for fit in write_fits]))
    read_head = int(np.argmax([fit[2] for fit in read_fits]))
    a, b, r2, fitted = write_fits[head]

    locations = np.arange(model.memory_locations)
    ideal = np.exp(-0.5 * (locations[:, None] - fitted[None, :]) ** 2)  # one narrow blob per timestep
    observed, reads = write_w[:, head].T, read_w[:, read_head].T

    used = np.maximum.reduce([ideal.max(1), observed.max(1), reads.max(1)]) > 0.01
    rows = used.nonzero()[0]
    window = slice(max(0, rows.min() - 2), rows.max() + 3) if len(rows) else slice(None)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    panels = [
        (f"Fitted: location = {a:.1f} × priority + {b:.1f}   (R² = {r2:.2f})", ideal),
        (f"Observed write locations (head {head} of {write_w.shape[1]})", observed),
        (f"Read locations (head {read_head}: location = {read_fits[read_head][0]:.1f} × step + "
         f"{read_fits[read_head][1]:.1f}, R² = {read_fits[read_head][2]:.2f})", reads),
    ]
    for ax, (title, image) in zip(axes, panels):
        window_image = image[window]
        im = ax.imshow(window_image, cmap="gray", aspect="auto", interpolation="nearest",
                       extent=(-0.5, image.shape[1] - 0.5, window.stop - 0.5, window.start - 0.5))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time →")
        fig.colorbar(im, ax=ax, fraction=0.05)
    axes[0].set_ylabel("Location")
    fig.text(0.5, 0.005, "write-head R²: " + "  ".join(f"{fit[2]:.2f}" for fit in write_fits) +
             "     read-head R²: " + "  ".join(f"{fit[2]:.2f}" for fit in read_fits),
             ha="center", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _fit(inputs, locations):
    """Least squares locations ≈ a * inputs + b, as (a, b, R², fitted locations)."""
    a, b = np.polyfit(inputs, locations.astype(float), 1)
    fitted = a * inputs + b
    spread = ((locations - locations.mean()) ** 2).sum()
    return a, b, 1 - ((locations - fitted) ** 2).sum() / spread if spread else 0.0, fitted
