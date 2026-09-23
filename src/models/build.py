"""The models you can train on the copy task, and the paper's learning rate for each."""

from src.models.lstm import LSTM
from src.models.ntm.ntm import NTM

MODELS = {
    "lstm": lambda: LSTM(),  # the baseline: 3 x 256 LSTM, Table 3
    "ntm-ff": lambda: NTM(controller="feedforward"),  # Table 1
    "ntm-lstm": lambda: NTM(controller="lstm"),  # Table 2
}

LEARNING_RATES = {"lstm": 3e-5, "ntm-ff": 1e-4, "ntm-lstm": 1e-4}


def build_model(name):
    if name not in MODELS:
        raise SystemExit(f"unknown model '{name}': choose from {', '.join(MODELS)}")
    return MODELS[name]()
