"""The models you can train on the copy task, and the paper's learning rate for each."""

from src.models.lstm import LSTM
from src.models.ntm.ntm import NTM

MODELS = {
    "lstm": lambda: LSTM(),  # the baseline: 3 x 256 LSTM, Table 3
    "ntm-ff": lambda: NTM(controller="feedforward"),  # Table 1
    "ntm-lstm": lambda: NTM(controller="lstm"),  # Table 2
}

# The paper's rates (Tables 1-3) assume one sequence per update. We default to batch 16, which is
# 16x fewer updates, and the LSTM-controller NTM is the one model that will not learn at the
# paper's rate under that: at 1e-4 it sits near chance, at 1e-3 it reaches 0.02 bits on a
# fixed-length-5 copy in 1,000 updates, matching the feed-forward controller.
LEARNING_RATES = {"lstm": 3e-5, "ntm-ff": 1e-4, "ntm-lstm": 1e-3}


def build_model(name):
    if name not in MODELS:
        raise SystemExit(f"unknown model '{name}': choose from {', '.join(MODELS)}")
    return MODELS[name]()
