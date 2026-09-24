"""Give the trained model your own sequence and see what it copies back."""

import matplotlib.pyplot as plt
import torch

from src.models.build import load_model
from src.tasks.copy import data as copy_data


def parse_vectors(vectors: list[str]) -> torch.Tensor:
    """Turns ["10110010", "01100101"] into a target of shape (1, L, 8)."""
    for v in vectors:
        if len(v) != copy_data.BITS or set(v) - {"0", "1"}:
            raise SystemExit(f"'{v}' isn't an 8-bit vector: use exactly 8 characters of 0 and 1, e.g. 10110010")
    return torch.tensor([[[int(bit) for bit in v] for v in vectors]]).float()


def try_sequence(model_name: str, checkpoint_path: str, target: torch.Tensor, out_path: str):
    model = load_model(model_name, checkpoint_path, copy_data, "copy")
    length = target.shape[1]
    x = copy_data.make_input(target)
    with torch.no_grad():
        outputs = torch.sigmoid(model(x)[0, length + 1:])  # probability each bit is 1, recall steps only
    predicted = (outputs > 0.5).int()

    print(f"{'step':>4}  {'you gave':<8}  {'model says':<10}  wrong bits")
    for t in range(length):
        given = "".join(str(int(b)) for b in target[0, t])
        said = "".join(str(int(b)) for b in predicted[t])
        wrong = sum(a != b for a, b in zip(given, said))
        print(f"{t:>4}  {given:<8}  {said:<10}  {wrong or ''}")
    total_wrong = int((predicted != target[0].int()).sum())
    print(f"{total_wrong} of {target.numel()} bits wrong")

    # Picture: what went in, what should come out, and what came out
    fig, axes = plt.subplots(3, 1, figsize=(max(4, x.shape[1] * 0.25), 5), gridspec_kw={"height_ratios": [9, 8, 8]})
    style = dict(cmap="jet", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    axes[0].imshow(x[0].T, **style)
    axes[0].set_ylabel("Input\n(bits + delimiter)", rotation=0, ha="right", va="center")
    axes[1].imshow(target[0].T, **style)
    axes[1].set_ylabel("Target", rotation=0, ha="right", va="center")
    axes[2].imshow(outputs.T, **style)
    axes[2].set_ylabel("Output", rotation=0, ha="right", va="center")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
