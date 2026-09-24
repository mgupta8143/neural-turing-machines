"""Repeat copy's own figure: generalisation past the training range (paper, Figure 8).

The paper tests two directions separately - a sequence twice as long as anything seen in
training, and twice as many repeats - because a network can manage one without the other.
"""

import matplotlib.pyplot as plt
import torch

from src.models.build import load_model
from src.tasks.repeat_copy import data as repeat_copy_data

# Training draws both from 1 to 10, so each of these doubles one of them
CASES = [
    ("trained range: 10 vectors, 10 repeats", 10, 10),
    ("twice the length: 20 vectors, 10 repeats", 20, 10),
    ("twice the repeats: 10 vectors, 20 repeats", 10, 20),
]


def plot_generalisation(model_name: str, checkpoint_path: str, out_path: str):
    model = load_model(model_name, checkpoint_path, repeat_copy_data, "repeat-copy")

    fig, axes = plt.subplots(len(CASES) * 2, 1, figsize=(12, 7),
                             gridspec_kw={"height_ratios": [1, 1] * len(CASES)})
    for i, (title, length, repeats) in enumerate(CASES):
        vectors = torch.randint(0, 2, (1, length, repeat_copy_data.BITS)).float()
        x = repeat_copy_data.make_input(vectors, repeats)
        target, mask = repeat_copy_data.make_target(vectors, repeats)
        with torch.no_grad():
            outputs = torch.sigmoid(model(x))

        scored = mask[0]
        for offset, (label, image) in enumerate((("Targets", target[0][scored].T),
                                                 ("Outputs", outputs[0][scored].T))):
            ax = axes[i * 2 + offset]
            ax.imshow(image, cmap="jet", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
            ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if offset == 0:
                ax.set_title(title, fontsize=10)
    axes[-1].set_xlabel("Time →")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
