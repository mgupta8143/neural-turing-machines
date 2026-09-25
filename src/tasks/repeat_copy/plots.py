"""Repeat copy's figure: generalisation past the training range (paper, Figure 8).

Length and repeats are doubled separately, because a network can manage one without the other.
"""

import matplotlib.pyplot as plt
import torch

from src.models.build import load_model
from src.tasks.repeat_copy import data

CASES = [  # training draws both from 1 to 10, so each of these doubles one of them
    ("trained range: 10 vectors, 10 repeats", 10, 10),
    ("twice the length: 20 vectors, 10 repeats", 20, 10),
    ("twice the repeats: 10 vectors, 20 repeats", 10, 20),
]


def plot_generalisation(model_name: str, checkpoint_path: str, out_path: str):
    model = load_model(model_name, checkpoint_path, data, "repeat-copy")
    fig, axes = plt.subplots(2 * len(CASES), 1, figsize=(12, 7))

    for i, (title, length, repeats) in enumerate(CASES):
        vectors = torch.randint(0, 2, (1, length, data.BITS)).float()
        target, mask = data.make_target(vectors, repeats)
        with torch.no_grad():
            outputs = torch.sigmoid(model(data.make_input(vectors, repeats)))

        scored = mask[0]
        for offset, label in enumerate(("Targets", "Outputs")):
            image = (target if label == "Targets" else outputs)[0][scored].T
            ax = axes[2 * i + offset]
            ax.imshow(image, cmap="jet", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
            ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if not offset:
                ax.set_title(title, fontsize=10)

    axes[-1].set_xlabel("Time →")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
