import random

import torch


def copy_batch(batch_size: int, min_len: int = 1, max_len: int = 20, bits: int = 8):
    """Copy task: show `length` random bit vectors, then a delimiter, then expect them back.

    Returns x: (batch, 2*length + 1, bits + 1) and target: (batch, length, bits).
    """
    length = random.randint(min_len, max_len)
    target = torch.randint(0, 2, (batch_size, length, bits)).float()

    x = torch.zeros(batch_size, 2 * length + 1, bits + 1)
    x[:, :length, :bits] = target
    x[:, length, bits] = 1.0  # delimiter on its own channel
    return x, target


if __name__ == "__main__":
    from src.models.lstm import LSTM

    torch.manual_seed(0)
    random.seed(0)

    x, target = copy_batch(batch_size=2, min_len=3, max_len=3)
    print("input x:", tuple(x.shape), "(batch, seq_len, 8 bits + delimiter)")
    print(x[0].int())
    print("target:", tuple(target.shape))

    model = LSTM()
    logits = model(x)
    print("output logits:", tuple(logits.shape), "(batch, seq_len, 8 bits)")

    length = target.shape[1]
    prediction = (torch.sigmoid(logits[:, length + 1:]) > 0.5).int()
    print("prediction on recall steps (untrained, so random):")
    print(prediction[0])
    print("target:")
    print(target[0].int())
