"""Command line for the copy task.

    uv run main.py demo                        run one batch through an untrained model
    uv run main.py train --model ntm-ff        train (also: lstm, ntm-lstm)
    uv run main.py plot --model ntm-ff         draw Figures 3 and 5 into figures/
    uv run main.py try 10110010 01100101 ...   copy your own 8-bit vectors
    uv run main.py try --random 30             or a random sequence of that length
    uv run main.py memory --model ntm-ff       the paper's Figure 6: what the heads read and wrote
"""

import argparse
import os

import torch

from src.models.build import LEARNING_RATES, MODELS, build_model
from src.tasks.copy.data import copy_batch
from src.tasks.copy.plots import plot_generalisation, plot_learning_curve, plot_memory_use
from src.tasks.copy.train import TrainConfig, train
from src.tasks.copy.try_it import parse_vectors, try_sequence


def demo(model_name):
    x, target = copy_batch(batch_size=2, min_len=3, max_len=3)
    print("input x:", tuple(x.shape), "= (batch, 2L + 1 timesteps, 8 bits + delimiter)")
    print(x[0].int())

    model = build_model(model_name)
    print(f"{model_name}: {model.num_parameters():,} parameters")
    logits = model(x)
    print("output logits:", tuple(logits.shape), "= (batch, timesteps, 8 bits)")

    length = target.shape[1]
    print("prediction on the recall steps (untrained, so random):")
    print((logits[0, length + 1:] > 0).int())
    print("target:")
    print(target[0].int())


def main():
    parser = argparse.ArgumentParser(description="NTM and LSTM on the copy task")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["demo", "train", "plot", "try", "memory"]:
        command = commands.add_parser(name)
        command.add_argument("--model", choices=list(MODELS), default="lstm")
    commands.choices["train"].add_argument("--sequences", type=int, default=TrainConfig.total_sequences)
    commands.choices["train"].add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    commands.choices["train"].add_argument("--learning-rate", type=float)
    commands.choices["train"].add_argument("--device", choices=["cpu", "cuda"], default="")
    commands.choices["train"].add_argument("--compile", action="store_true", help="torch.compile: ~1.7x per step after a slow warmup")
    commands.choices["try"].add_argument("vectors", nargs="*", help="8-bit vectors like 10110010")
    commands.choices["try"].add_argument("--random", type=int, help="use a random sequence of this length")
    commands.choices["memory"].add_argument("--length", type=int, default=20, help="sequence length to trace")
    args = parser.parse_args()

    config = TrainConfig(model=args.model)

    if args.command == "demo":
        torch.manual_seed(0)
        demo(args.model)
        return

    if args.command == "train":
        config.total_sequences = args.sequences
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate or LEARNING_RATES[args.model]
        config.device = args.device
        config.compile = args.compile
        train(config)
        return

    if not os.path.exists(config.checkpoint_path):
        print(f"Nothing saved yet: train {args.model} first, or wait for its first {config.log_every:,} sequences.")
        return

    os.makedirs("figures", exist_ok=True)
    if args.command == "plot":
        plot_learning_curve(config.log_path, f"figures/{args.model}_learning_curve.png", label=args.model)
        plot_generalisation(args.model, config.checkpoint_path, f"figures/{args.model}_generalisation.png")
        print(f"saved figures/{args.model}_learning_curve.png and figures/{args.model}_generalisation.png")
    elif args.command == "memory":
        if not args.model.startswith("ntm"):
            raise SystemExit("memory plots need an NTM: --model ntm-ff or ntm-lstm")
        out = f"figures/{args.model}_memory_length{args.length}.png"
        plot_memory_use(args.model, config.checkpoint_path, out, args.length)
        print(f"saved {out}")
    elif args.command == "try":
        if args.random:
            target = torch.randint(0, 2, (1, args.random, 8)).float()
        elif args.vectors:
            target = parse_vectors(args.vectors)
        else:
            raise SystemExit("Give some 8-bit vectors (e.g. 10110010 01100101) or --random LENGTH")
        try_sequence(args.model, config.checkpoint_path, target, f"figures/{args.model}_try.png")
        print(f"saved figures/{args.model}_try.png")


if __name__ == "__main__":
    main()
