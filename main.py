"""Command line for the paper's tasks.

    uv run main.py demo                        run one batch through an untrained model
    uv run main.py train --model ntm-ff        train (also: lstm, ntm-lstm)
    uv run main.py plot --model ntm-ff         draw Figures 3 and 5 into figures/
    uv run main.py memory --model ntm-ff       the paper's Figure 6: what the heads read and wrote
    uv run main.py compare                     all three learning curves on one plot (Figure 3)
    uv run main.py all                         train all three and keep every figure up to date

Every command takes --task; it defaults to copy. Results go to results/<task>/<model>/ and
figures to figures/<task>_<model>_*.png.
"""

import argparse
import os
import subprocess
import sys
import time

import torch

from src.models.build import MODELS, build_model
from src.plots import plot_all_learning_curves, plot_learning_curve, task_plot
from src.tasks.registry import TASKS, get_task
from src.train import TrainConfig, train


def demo(task_name, model_name):
    task = get_task(task_name)
    x, target, mask = task.batch(batch_size=2, min_len=3, max_len=3)  # short, so it fits on screen
    # One row per timestep. Floats rather than ints: repeat copy feeds in a fractional scalar.
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=120)
    print("input x:", tuple(x.shape), f"= (batch, timesteps, {task.INPUT_SIZE} input channels)")
    print(x[0])

    model = build_model(model_name, task.INPUT_SIZE, task.OUTPUT_SIZE, task_name)
    print(f"{model_name}: {model.num_parameters():,} parameters")
    logits = model(x)
    print("output logits:", tuple(logits.shape), f"= (batch, timesteps, {task.OUTPUT_SIZE} output channels)")

    print(f"prediction on the {int(mask[0].sum())} scored steps (untrained, so random):")
    print((logits[0][mask[0]] > 0).int())
    print("target:")
    print(target[0][mask[0]].int())


def figure_dir(task_name):
    """Where a task's figures live."""
    os.makedirs(f"figures/{task_name}", exist_ok=True)
    return f"figures/{task_name}"


def draw_every_figure(task_name):
    """Redraw whatever the saved checkpoints allow. Safe to call while training is running."""
    os.makedirs("figures", exist_ok=True)
    generalisation = task_plot(task_name, "plot_generalisation")
    memory_use = task_plot(task_name, "plot_memory_use")
    for model in MODELS:
        config = TrainConfig(model=model, task=task_name)
        if not os.path.exists(config.checkpoint_path):
            continue
        prefix = f"{figure_dir(task_name)}/{model}"
        plot_learning_curve(config.log_path, f"{prefix}_learning_curve.png", label=model)
        if generalisation:
            generalisation(model, config.checkpoint_path, f"{prefix}_generalisation.png")
        if model.startswith("ntm") and memory_use:
            for length in (20, 40):
                memory_use(model, config.checkpoint_path, f"{prefix}_memory_length{length}.png", length)
    logs = {model: TrainConfig(model=model, task=task_name).log_path for model in MODELS}
    plot_all_learning_curves(logs, f"{figure_dir(task_name)}/learning_curves.png")


def train_all(task_name, sequences, batch_size, refresh, seed=TrainConfig.seed):
    """Train every model at once, redrawing the figures every `refresh` seconds.

    Each model writes its own log, checkpoint and figures, so nothing collides, and the figures
    are always current: stop this whenever you like and keep what has been drawn.
    """
    running = {}
    threads_each = max(1, (os.cpu_count() or 4) // len(MODELS))  # the models run side by side
    for model in MODELS:
        results = TrainConfig(model=model, task=task_name).results_path
        os.makedirs(results, exist_ok=True)
        log = open(f"{results}/train.log", "w")
        command = [sys.executable, __file__, "train", "--model", model, "--task", task_name,
                   "--seed", str(seed), "--sequences", str(sequences),
                   "--batch-size", str(batch_size), "--threads", str(threads_each)]
        running[model] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                          env={**os.environ, "PYTHONUNBUFFERED": "1"})
        print(f"training {model}, logging to {results}/train.log")

    print(f"redrawing every figure every {refresh // 60} minutes; ctrl-c to stop")
    try:
        while any(process.poll() is None for process in running.values()):
            time.sleep(refresh)
            draw_every_figure(task_name)
            print(f"{time.strftime('%H:%M')} figures updated")
    except KeyboardInterrupt:
        for process in running.values():
            process.terminate()
    draw_every_figure(task_name)
    print("figures written to figures/")


def main():
    parser = argparse.ArgumentParser(description="NTM and LSTM on the paper's tasks")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["demo", "train", "plot", "memory", "compare", "all"]:
        command = commands.add_parser(name)
        command.add_argument("--model", choices=list(MODELS), default="lstm")
        command.add_argument("--task", choices=list(TASKS), default="copy")
    commands.choices["train"].add_argument("--sequences", type=int, default=TrainConfig.total_sequences)
    commands.choices["train"].add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    commands.choices["train"].add_argument("--learning-rate", type=float)
    commands.choices["train"].add_argument("--seed", type=int, default=TrainConfig.seed)
    commands.choices["train"].add_argument("--device", choices=["cpu", "cuda"], default="")
    commands.choices["train"].add_argument("--compile", action="store_true", help="torch.compile: faster per step after a slow warmup")
    commands.choices["train"].add_argument("--threads", type=int, default=TrainConfig.threads)
    commands.choices["memory"].add_argument("--length", type=int, default=20, help="sequence length to trace")
    commands.choices["all"].add_argument("--sequences", type=int, default=TrainConfig.total_sequences)
    commands.choices["all"].add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    commands.choices["all"].add_argument("--seed", type=int, default=TrainConfig.seed)
    commands.choices["all"].add_argument("--refresh", type=int, default=900, help="seconds between figure redraws")
    args = parser.parse_args()

    config = TrainConfig(model=args.model, task=args.task)

    if args.command == "all":
        train_all(args.task, args.sequences, args.batch_size, args.refresh, args.seed)
        return

    if args.command == "compare":
        os.makedirs("figures", exist_ok=True)
        logs = {model: TrainConfig(model=model, task=args.task).log_path for model in MODELS}
        out = f"{figure_dir(args.task)}/learning_curves.png"
        plot_all_learning_curves(logs, out)
        print(f"saved {out}")
        return

    if args.command == "demo":
        torch.manual_seed(TrainConfig.seed)
        demo(args.task, args.model)
        return

    if args.command == "train":
        config.total_sequences = args.sequences
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate or 0.0
        config.seed = args.seed
        config.device = args.device
        config.compile = args.compile
        config.threads = args.threads
        train(config)
        return

    if not os.path.exists(config.checkpoint_path):
        print(f"Nothing saved yet: train {args.model} on {args.task} first, "
              f"or wait for its first {config.log_every:,} sequences.")
        return

    prefix = f"{figure_dir(args.task)}/{args.model}"
    if args.command == "plot":
        plot_learning_curve(config.log_path, f"{prefix}_learning_curve.png", label=args.model)
        print(f"saved {prefix}_learning_curve.png")
        generalisation = task_plot(args.task, "plot_generalisation")
        if generalisation:
            generalisation(args.model, config.checkpoint_path, f"{prefix}_generalisation.png")
            print(f"saved {prefix}_generalisation.png")
    elif args.command == "memory":
        if not args.model.startswith("ntm"):
            raise SystemExit("memory plots need an NTM: --model ntm-ff or ntm-lstm")
        memory_use = task_plot(args.task, "plot_memory_use")
        if not memory_use:
            raise SystemExit(f"{args.task} has no memory figure")
        out = f"{prefix}_memory_length{args.length}.png"
        memory_use(args.model, config.checkpoint_path, out, args.length)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
