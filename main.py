"""Command line for the copy task.

    uv run main.py demo     run one batch through the untrained model and print the shapes
    uv run main.py train    train the LSTM (options: --sequences, --batch-size, --learning-rate)
    uv run main.py plot     draw Figures 3 and 5 from the saved log and model into figures/
"""

import argparse
import os

import torch

from src.models.lstm import LSTM
from src.tasks.copy.data import copy_batch
from src.tasks.copy.plots import plot_generalisation, plot_learning_curve
from src.tasks.copy.train import TrainConfig, train


def demo():
    x, target = copy_batch(batch_size=2, min_len=3, max_len=3)
    print("input x:", tuple(x.shape), "= (batch, 2L + 1 timesteps, 8 bits + delimiter)")
    print(x[0].int())

    logits = LSTM()(x)
    print("output logits:", tuple(logits.shape), "= (batch, timesteps, 8 bits)")

    length = target.shape[1]
    print("prediction on the recall steps (untrained, so random):")
    print((logits[0, length + 1:] > 0).int())
    print("target:")
    print(target[0].int())


def main():
    parser = argparse.ArgumentParser(description="LSTM baseline on the NTM copy task")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo")
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--sequences", type=int, default=TrainConfig.total_sequences)
    train_parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    train_parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    commands.add_parser("plot")
    args = parser.parse_args()

    config = TrainConfig()
    if args.command == "demo":
        torch.manual_seed(0)
        demo()
    elif args.command == "train":
        config.total_sequences = args.sequences
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        train(config)
    elif args.command == "plot":
        if not os.path.exists(config.checkpoint_path):
            print(f"Nothing to plot yet: {config.checkpoint_path} is written after the first {config.log_every:,} sequences.")
            return
        # Figures go in figures/ (committed, shown in the README); logs and models stay in results/
        os.makedirs("figures", exist_ok=True)
        plot_learning_curve(config.log_path, "figures/copy_learning_curve.png")
        plot_generalisation(config.checkpoint_path, "figures/copy_generalisation.png")
        print("saved figures/copy_learning_curve.png and figures/copy_generalisation.png")


if __name__ == "__main__":
    main()
