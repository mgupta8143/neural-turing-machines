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
    #                     ntm-ff: heads, controller   ntm-lstm: heads, controller, layers   lstm: units, layers
    "copy": {
        "ntm-ff": dict(heads=1, controller_size=100, learning_rate=1e-4),
        "ntm-lstm": dict(heads=1, controller_size=100, learning_rate=1e-4),
        "lstm": dict(hidden_size=256, num_layers=3, learning_rate=3e-5),
    },
    "repeat-copy": {
        "ntm-ff": dict(heads=1, controller_size=100, learning_rate=1e-4),
        "ntm-lstm": dict(heads=1, controller_size=100, learning_rate=1e-4),
        "lstm": dict(hidden_size=512, num_layers=3, learning_rate=3e-5),
    },
    "associative-recall": {
        "ntm-ff": dict(heads=4, controller_size=256, learning_rate=1e-4),
        "ntm-lstm": dict(heads=1, controller_size=100, learning_rate=1e-4),
        "lstm": dict(hidden_size=256, num_layers=3, learning_rate=1e-4),
    },
    "dynamic-ngrams": {
        "ntm-ff": dict(heads=1, controller_size=100, learning_rate=3e-5),
        "ntm-lstm": dict(heads=1, controller_size=100, learning_rate=3e-5),
        "lstm": dict(hidden_size=128, num_layers=3, learning_rate=1e-4),
    },
    "priority-sort": {
        "ntm-ff": dict(heads=8, controller_size=512, learning_rate=3e-5),
        "ntm-lstm": dict(heads=5, controller_size=100, controller_layers=2, learning_rate=3e-5),
        "lstm": dict(hidden_size=128, num_layers=3, learning_rate=3e-5),
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
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model
