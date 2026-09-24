"""The models you can train, and the paper's learning rate for each.

Each takes the task's channel counts, so the same three models fit any task in the registry.
"""

import torch

from src.models.lstm import LSTM
from src.models.ntm.ntm import NTM

MODELS = {
    # the baseline: 3 x 256 LSTM, Table 3
    "lstm": lambda i, o: LSTM(input_size=i, output_size=o),
    "ntm-ff": lambda i, o: NTM(input_size=i, output_size=o, controller="feedforward"),  # Table 1
    "ntm-lstm": lambda i, o: NTM(input_size=i, output_size=o, controller="lstm"),  # Table 2
}

# The paper's rates, Tables 1-3. They assume one sequence per update, which is the default here.
# With a larger batch there are proportionally fewer updates and these are too low: at batch 16
# the LSTM-controller NTM sits near chance at 1e-4 and needs about 1e-3.
LEARNING_RATES = {"lstm": 3e-5, "ntm-ff": 1e-4, "ntm-lstm": 1e-4}


def build_model(name, input_size, output_size):
    if name not in MODELS:
        raise SystemExit(f"unknown model '{name}': choose from {', '.join(MODELS)}")
    return MODELS[name](input_size, output_size)


def load_model(name, checkpoint_path, task):
    """The trained model, ready for the plots: same shape as the run that saved it."""
    model = build_model(name, task.INPUT_SIZE, task.OUTPUT_SIZE)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model
