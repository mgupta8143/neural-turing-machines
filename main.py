"""Command line for the copy task.

    uv run main.py demo                        run one batch through an untrained model
    uv run main.py train --model ntm-ff        train (also: lstm, ntm-lstm)
    uv run main.py plot --model ntm-ff         draw Figures 3 and 5 into figures/
    uv run main.py try 10110010 01100101 ...   copy your own 8-bit vectors
    uv run main.py try --random 30             or a random sequence of that length
    uv run main.py memory --model ntm-ff       the paper's Figure 6: what the heads read and wrote
    uv run main.py compare                     all three learning curves on one plot (Figure 3)
    uv run main.py all                         train all three and keep every figure up to date
"""

import argparse
import os
import subprocess
import sys
import time

import torch

from src.models.build import LEARNING_RATES, MODELS, build_model
from src.tasks.copy.data import copy_batch
from src.tasks.copy.plots import plot_all_learning_curves, plot_generalisation, plot_learning_curve, plot_memory_use
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


def draw_every_figure():
    """Redraw whatever the saved checkpoints allow. Safe to call while training is running."""
    os.makedirs("figures", exist_ok=True)
    for model in MODELS:
        config = TrainConfig(model=model)
        if not os.path.exists(config.checkpoint_path):
            continue
        plot_learning_curve(config.log_path, f"figures/{model}_learning_curve.png", label=model)
        plot_generalisation(model, config.checkpoint_path, f"figures/{model}_generalisation.png")
        if model.startswith("ntm"):
            for length in (20, 40):
                plot_memory_use(model, config.checkpoint_path, f"figures/{model}_memory_length{length}.png", length)
    plot_all_learning_curves({m: TrainConfig(model=m).log_path for m in MODELS}, "figures/learning_curves.png")


def train_all(sequences, batch_size, refresh, seed=TrainConfig.seed):
    """Train every model at once, redrawing the figures every `refresh` seconds.

    Each model writes its own log, checkpoint and figures, so nothing collides, and the figures
    are always current: stop this whenever you like and keep what has been drawn.
    """
    running = {}
    threads_each = max(1, (os.cpu_count() or 4) // len(MODELS))  # the models run side by side
    for model in MODELS:
        os.makedirs(os.path.dirname(TrainConfig(model=model).log_path), exist_ok=True)
        log = open(f"results/copy/{model}/train.log", "w")
        command = [sys.executable, __file__, "train", "--model", model, "--seed", str(seed),
                   "--sequences", str(sequences), "--batch-size", str(batch_size),
                   "--threads", str(threads_each)]
        running[model] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                          env={**os.environ, "PYTHONUNBUFFERED": "1"})
        print(f"training {model}, logging to results/copy/{model}/train.log")

    print(f"redrawing every figure every {refresh // 60} minutes; ctrl-c to stop")
    try:
        while any(process.poll() is None for process in running.values()):
            time.sleep(refresh)
            draw_every_figure()
            print(f"{time.strftime('%H:%M')} figures updated")
    except KeyboardInterrupt:
        for process in running.values():
            process.terminate()
    draw_every_figure()
    print("figures written to figures/")


def main():
    parser = argparse.ArgumentParser(description="NTM and LSTM on the copy task")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["demo", "train", "plot", "try", "memory", "compare", "all"]:
        command = commands.add_parser(name)
        command.add_argument("--model", choices=list(MODELS), default="lstm")
    commands.choices["train"].add_argument("--sequences", type=int, default=TrainConfig.total_sequences)
    commands.choices["train"].add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    commands.choices["train"].add_argument("--learning-rate", type=float)
    commands.choices["train"].add_argument("--seed", type=int, default=TrainConfig.seed)
    commands.choices["train"].add_argument("--device", choices=["cpu", "cuda"], default="")
    commands.choices["train"].add_argument("--compile", action="store_true", help="torch.compile: faster per step after a slow warmup")
    commands.choices["train"].add_argument("--threads", type=int, default=TrainConfig.threads)
    commands.choices["try"].add_argument("vectors", nargs="*", help="8-bit vectors like 10110010")
    commands.choices["try"].add_argument("--random", type=int, help="use a random sequence of this length")
    commands.choices["memory"].add_argument("--length", type=int, default=20, help="sequence length to trace")
    commands.choices["all"].add_argument("--sequences", type=int, default=TrainConfig.total_sequences)
    commands.choices["all"].add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    commands.choices["all"].add_argument("--seed", type=int, default=TrainConfig.seed)
    commands.choices["all"].add_argument("--refresh", type=int, default=900, help="seconds between figure redraws")
    args = parser.parse_args()

    config = TrainConfig(model=args.model)

    if args.command == "all":
        train_all(args.sequences, args.batch_size, args.refresh, args.seed)
        return

    if args.command == "compare":
        os.makedirs("figures", exist_ok=True)
        logs = {model: TrainConfig(model=model).log_path for model in MODELS}
        plot_all_learning_curves(logs, "figures/learning_curves.png")
        print("saved figures/learning_curves.png")
        return

    if args.command == "demo":
        torch.manual_seed(TrainConfig.seed)
        demo(args.model)
        return

    if args.command == "train":
        config.total_sequences = args.sequences
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate or LEARNING_RATES[args.model]
        config.seed = args.seed
        config.device = args.device
        config.compile = args.compile
        config.threads = args.threads
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
