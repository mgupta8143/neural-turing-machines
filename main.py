import math

import torch
import torch.nn.functional as F

from src.models.lstm import LSTM
from src.tasks.copy import copy_batch

# Copy task settings for the LSTM baseline in Graves et al. (2014)
STEPS = 100_000
BATCH_SIZE = 1
LEARNING_RATE = 3e-5
MOMENTUM = 0.9
LOG_EVERY = 500


def train():
    model = LSTM()
    print(f"parameters: {model.num_parameters():,}")
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    running = 0.0
    for step in range(1, STEPS + 1):
        x, target = copy_batch(BATCH_SIZE)
        length = target.shape[1]

        logits = model(x)[:, length + 1:]  # only the recall steps are scored
        loss = F.binary_cross_entropy_with_logits(logits, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()

        # The paper reports cost per sequence in bits: summed BCE, in log base 2
        running += loss.item() * target[0].numel() / math.log(2)
        if step % LOG_EVERY == 0:
            print(f"step {step:>7}  cost per sequence (bits): {running / LOG_EVERY:.2f}")
            running = 0.0

    return model


def show_example(model, length: int = 5):
    x, target = copy_batch(1, length, length)
    with torch.no_grad():
        prediction = (torch.sigmoid(model(x)[:, length + 1:]) > 0.5).int()
    print("target:\n", target[0].int())
    print("prediction:\n", prediction[0])


if __name__ == "__main__":
    model = train()
    show_example(model)
