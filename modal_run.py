"""Train on a Modal GPU and bring the results back.

    uv run --with modal modal run modal_run.py --model lstm
    uv run --with modal modal run modal_run.py --model ntm-ff --task repeat-copy

Each step replays as a single CUDA graph, which is what makes a GPU worth using here: eager, the
NTM is several times slower on a GPU than on a laptop CPU, because it launches a few hundred tiny
kernels per sequence. Results land in results/copy/<model>/ exactly as a local run leaves them.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "matplotlib")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("main.py", remote_path="/root/main.py")
)

volume = modal.Volume.from_name("ntm-results", create_if_missing=True)
app = modal.App("ntm-tasks", image=image)


@app.function(gpu="A10G", timeout=8 * 3600, volumes={"/root/results": volume})
def train_on_gpu(model: str, task: str, sequences: int, seed: int):
    import sys

    sys.path.insert(0, "/root")
    from src.train import TrainConfig, train

    config = TrainConfig(
        model=model,
        task=task,
        total_sequences=sequences,
        seed=seed,
        device="cuda",
    )
    train(config)
    volume.commit()


@app.local_entrypoint()
def main(model: str = "lstm", task: str = "copy", sequences: int = 1_000_000, seed: int = 0):
    import pathlib

    train_on_gpu.remote(model, task, sequences, seed)

    local = pathlib.Path("results") / task / model
    local.mkdir(parents=True, exist_ok=True)
    for name in ("log.csv", "model.pt", "run.json"):
        with open(local / name, "wb") as f:
            for chunk in volume.read_file(f"{task}/{model}/{name}"):
                f.write(chunk)
    print(f"downloaded to {local}/  — now run: uv run main.py plot --task {task} --model {model}")
