"""The models you can train, and the paper's learning rate for each.

Each takes the task's channel counts, so the same three models fit any task in the registry.
"""

import torch

from src.models.lstm import LSTM
from src.models.ntm.ntm import NTM

MODELS = ["lstm", "ntm-ff", "ntm-lstm"]

# The paper gives every task its own architecture and learning rate, in Tables 1 (NTM with a
# feed-forward controller), 2 (NTM with an LSTM controller) and 3 (the LSTM baseline). Running
# every task with the copy task's model, as an earlier version of this code did, is not a fair
# test: priority sort is meant to have 8 heads and a 512-unit controller, not 1 and 100.
#
# "#Heads" is read here as that many read heads and that many write heads.
SETTINGS = {
    "copy": {
        "ntm-ff": dict(heads=1, controller_size=100, learning_rate=1e-4),  # Table 1
        "ntm-lstm": dict(heads=1, controller_size=100, learning_rate=1e-4),  # Table 2
        "lstm": dict(hidden_size=256, num_layers=3, learning_rate=3e-5),  # Table 3
    },
    "priority-sort": {
        "ntm-ff": dict(heads=8, controller_size=512, learning_rate=3e-5),  # Table 1
        # Table 2's "2 x 100" is two stacked layers of 100 units
        "ntm-lstm": dict(heads=5, controller_size=100, controller_layers=2, learning_rate=3e-5),
        "lstm": dict(hidden_size=128, num_layers=3, learning_rate=3e-5),  # Table 3
    },
}


def settings(task_name, model_name):
    if model_name not in MODELS:
        raise SystemExit(f"unknown model '{model_name}': choose from {', '.join(MODELS)}")
    return dict(SETTINGS[task_name][model_name])


def learning_rate(task_name, model_name):
    return settings(task_name, model_name)["learning_rate"]


def build_model(name, input_size, output_size, task_name="copy"):
    options = settings(task_name, name)
    options.pop("learning_rate")
    if name == "lstm":
        return LSTM(input_size=input_size, output_size=output_size, **options)
    heads = options.pop("heads")
    return NTM(input_size=input_size, output_size=output_size,
               controller="feedforward" if name == "ntm-ff" else "lstm",
               num_read_heads=heads, num_write_heads=heads, **options)


def load_model(name, checkpoint_path, task, task_name="copy"):
    """The trained model, ready for the plots: same shape as the run that saved it."""
    model = build_model(name, task.INPUT_SIZE, task.OUTPUT_SIZE, task_name)
    state = torch.load(checkpoint_path, map_location="cpu")

    # Checkpoints saved before the controller could stack layers hold (1, hidden) starting states
    # where it now expects (layers, 1, hidden). Reshape them rather than lose the run.
    for key in ("controller.h0", "controller.c0"):
        if key in state and state[key].dim() == 2:
            state[key] = state[key].unsqueeze(0)
    for key in list(state):
        if key.startswith("controller.cell."):
            state[key.replace("controller.cell.", "controller.cells.0.")] = state.pop(key)

    model.load_state_dict(state)
    model.eval()
    return model
