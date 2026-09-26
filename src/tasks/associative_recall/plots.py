"""The associative recall task's own figures.

Figure 11: cost per sequence against the number of items per episode, one line per model.
Figure 12: what the heads did on a single episode, with the paper's green, red and black boxes.

The learning curves, which look the same for every task, are in src/plots.py.
"""

import os
import random

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from src.loss import masked_bce
from src.models.build import MODELS, load_model
from src.tasks.associative_recall import data as recall_data
from src.train import TrainConfig

TASK = "associative-recall"
ITEM_COUNTS = [6, 10, 15, 20]  # the four points the paper's Figure 11 plots
EPISODES = 200  # per point; enough that the mean moves by well under a bit between redraws


def _cost(model, items, episodes, chunk=50):
    """Mean cost of one episode in bits, over `episodes` episodes of `items` items each."""
    total = 0.0
    with torch.no_grad():
        for start in range(0, episodes, chunk):
            size = min(chunk, episodes - start)
            x, target, mask = recall_data.batch(size, items, items)
            total += masked_bce(model(x), target, mask)[1].item() * size
    return total / episodes


def plot_generalisation(model_name: str, checkpoint_path: str, out_path: str, episodes: int = EPISODES):
    """Figure 11 of the paper: every model on one pair of axes.

    main.py calls this once per trained model, but the paper's figure compares all three, so we
    draw whichever have a checkpoint and let the caller's own path win for its model. A model
    still training simply has no line yet, and the axes stay the paper's either way.
    """
    from src.plots import PAPER_STYLE  # local: src.plots imports this module

    checkpoints = {model: TrainConfig(model=model, task=TASK).checkpoint_path for model in MODELS}
    checkpoints[model_name] = checkpoint_path

    fig, ax = plt.subplots(figsize=(7, 5))
    for model in ("lstm", "ntm-lstm", "ntm-ff"):  # the paper's legend order
        if not os.path.exists(checkpoints[model]):
            continue
        # Same seed per model, so every line is scored on the same episodes and the gaps between
        # the lines are the models rather than the draw.
        random.seed(0)
        torch.manual_seed(0)
        net = load_model(model, checkpoints[model], recall_data, TASK)
        label, colour, marker = PAPER_STYLE[model]
        ax.plot(ITEM_COUNTS, [_cost(net, items, episodes) for items in ITEM_COUNTS],
                marker=marker, color=colour, markersize=5, linewidth=1, label=label)

    # Training uses 2-6 items, so the leftmost point is already the trained maximum and
    # everything to the right of it is extrapolation.
    ax.axvline(recall_data.MAX_ITEMS, color="0.7", linestyle=":", linewidth=1, zorder=0)
    ax.text(recall_data.MAX_ITEMS + 0.2, 39, "trained maximum", color="0.45", fontsize=8, va="top")

    ax.set_xlim(5, 21)
    ax.set_ylim(0, 40)
    ax.set_xticks(range(6, 21, 2))
    ax.set_yticks(range(0, 41, 5))
    ax.set_xlabel("number of items per sequence")
    ax.set_ylabel("cost per sequence (bits)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title(f"Generalisation on associative recall ({episodes} episodes per point)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _per_head(trace, key, timesteps):
    """A traced quantity as (heads, rows, timesteps): the trace lists it head by head per step."""
    stacked = torch.stack(trace[key])
    return stacked.view(timesteps, -1, stacked.shape[-1]).permute(1, 2, 0)


def _box(ax, first_column, first_row, columns, rows, colour):
    ax.add_patch(Rectangle((first_column - 0.5, first_row - 0.5), columns, rows,
                           fill=False, edgecolor=colour, linewidth=1.6))


def plot_memory_use(model_name: str, checkpoint_path: str, out_path: str, items: int = 3):
    """Figure 12 of the paper: what the heads did on one episode, with the paper's boxes.

    The episode is built here rather than drawn from data.batch, so the query, the target and the
    delimiter that closes the queried item are known timesteps and can be boxed:

        green   the query item in "Inputs"
        red     the target item where it appears in the list, and the answer in "Outputs"
        black   the add at the delimiter closing the queried item, which the paper's analysis
                says is the compressed representation the later content lookup keys on
        red     on the weightings, the locations the read head used for the answer

    A network that has solved the task reads three contiguous locations for the answer and wrote
    them when the delimiters went by; an untrained one smears over memory and the red location
    box grows to fill the panel, which is the control.
    """
    model = load_model(model_name, checkpoint_path, recall_data, TASK)
    torch.manual_seed(0)
    vectors = torch.randint(0, 2, (1, items, recall_data.ITEM_STEPS, recall_data.BITS)).float()
    queried = torch.tensor([min(1, items - 2)])  # the paper queries the second item
    x = recall_data.make_input(vectors, queried)

    trace = {}
    with torch.no_grad():
        outputs = torch.sigmoid(model(x, trace=trace)[0])

    steps, block = recall_data.ITEM_STEPS, recall_data.ITEM_STEPS + 1
    timesteps = x.shape[1]
    index = int(queried)
    query_start = items * block + 2
    target_start = (index + 1) * block + 1
    key_step = (index + 1) * block  # the delimiter that closes the queried item
    answer_start = timesteps - steps

    write_w = _per_head(trace, "write_weightings", timesteps)
    read_w = _per_head(trace, "read_weightings", timesteps)
    adds = _per_head(trace, "adds", timesteps)
    reads = _per_head(trace, "reads", timesteps)

    # The feedforward config has four read and four write heads; the paper shows one of each. We
    # show the sharpest head, measured as the mean peak of its weighting, because a diffuse head
    # is a smear whatever it is doing and the sharp one is the one whose addressing can be read
    # off the picture. The read head is judged on the answer steps, which is what the figure is
    # about. The chosen index goes in the panel title so the figure never hides the other three.
    write_head = int(write_w.max(dim=1).values.mean(dim=1).argmax())
    read_head = int(read_w[:, :, answer_start:].max(dim=1).values.mean(dim=1).argmax())
    write_w, read_w = write_w[write_head], read_w[read_head]

    # The locations the read head took the answer from: the paper's red boxes on the weightings
    peaks = read_w[:, answer_start:].argmax(dim=0)
    used = (write_w.max(dim=1).values + read_w.max(dim=1).values) > 0.01
    used[peaks] = True  # so the box is always in view, however diffuse the weightings are
    rows = used.nonzero().flatten()
    window = slice(max(0, int(rows.min()) - 2), int(rows.max()) + 3)
    # The paper's box is three locations tall, one per time slice of the target item; a network
    # that reads them all from one location still gets a box rather than a hairline.
    height = max(recall_data.ITEM_STEPS, int(peaks.max()) - int(peaks.min()) + 1)
    first = int(peaks.min()) - window.start

    heads = len(model.read_heads)
    fig, axes = plt.subplots(3, 2, figsize=(11, 7), gridspec_kw={"height_ratios": [1, 1.3, 2.6]})
    panels = [
        ("Inputs", x[0].T, "gray", axes[0, 0]),
        ("Outputs", outputs.T, "gray", axes[0, 1]),
        (f"Adds (write head {write_head + 1} of {heads})", adds[write_head], "jet", axes[1, 0]),
        (f"Reads (read head {read_head + 1} of {heads})", reads[read_head], "jet", axes[1, 1]),
        (f"Write weightings (head {write_head + 1} of {heads})", write_w[window], "gray", axes[2, 0]),
        (f"Read weightings (head {read_head + 1} of {heads})", read_w[window], "gray", axes[2, 1]),
    ]
    for title, image, cmap, ax in panels:
        ax.imshow(image, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    _box(axes[0, 0], query_start, 0, steps, recall_data.BITS, "#12b012")
    _box(axes[0, 0], target_start, 0, steps, recall_data.BITS, "#e01010")
    _box(axes[0, 1], answer_start, 0, steps, recall_data.OUTPUT_SIZE, "#e01010")
    _box(axes[1, 0], key_step, 0, 1, model.memory_width, "black")
    # Those locations were written at some point while the list went by: the exact step is what
    # the figure is there to show, so the box spans the list rather than guessing at one column.
    _box(axes[2, 0], 0, first, items * block + 1, height, "#e01010")
    _box(axes[2, 1], answer_start, first, steps, height, "#e01010")

    axes[2, 0].set_ylabel("Location")
    for ax in (axes[2, 0], axes[2, 1]):
        ax.set_xlabel("Time →")
    fig.suptitle(f"{model_name} memory use on associative recall, {items} items", fontsize=11)
    fig.text(0.5, 0.005, "green: the query item    red: the target item, the answer, and the "
             "locations it was read from    black: the add the query is later looked up by",
             ha="center", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
