"""Training on a rented GPU, and getting the results back.

    uv run --with modal modal run src/remote.py::train --model ntm-ff --task copy
    uv run --with modal modal run src/remote.py::fetch

`modal` is deliberately not a project dependency: everything in main.py runs without it, and
only these two commands need an account. Pass --detach to `modal run` for a long run, or the
job is cancelled when the laptop that launched it goes away.

Each step replays as a single CUDA graph, one captured per sequence length, which is what makes
a GPU worth using here: eager, the NTM launches a few hundred tiny kernels per sequence and runs
several times slower on a GPU than on a laptop CPU.

Results are written into a Modal volume as training goes, so they survive the client dropping;
`fetch` copies whatever is in there into results/<task>/<model>/ and is safe to re-run.
"""

import pathlib

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "matplotlib")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("main.py", remote_path="/root/main.py")
)

volume = modal.Volume.from_name("ntm-results", create_if_missing=True)
app = modal.App("ntm-tasks", image=image)

RESULTS = pathlib.Path("results")


@app.function(gpu="A10G", timeout=8 * 3600, volumes={"/root/results": volume})
def train_on_gpu(model: str, task: str, sequences: int, seed: int):
    import sys

    sys.path.insert(0, "/root")
    from src.train import TrainConfig, train

    train(TrainConfig(model=model, task=task, total_sequences=sequences, seed=seed, device="cuda"))
    volume.commit()


def download(path: str):
    local = RESULTS / path
    local.parent.mkdir(parents=True, exist_ok=True)
    with open(local, "wb") as f:
        for chunk in volume.read_file(path):
            f.write(chunk)
    return local


@app.local_entrypoint()
def train(model: str = "lstm", task: str = "copy", sequences: int = 1_000_000, seed: int = 0):
    train_on_gpu.remote(model, task, sequences, seed)
    for name in ("log.csv", "model.pt", "run.json"):
        print(f"downloaded {download(f'{task}/{model}/{name}')}")
    print(f"now run: uv run main.py plot --task {task} --model {model}")


@app.local_entrypoint()
def fetch():
    """Everything in the volume, for when a run is still going or its client dropped."""
    for entry in volume.listdir("/", recursive=True):
        if entry.type.name == "FILE":
            print(f"{entry.path} -> {download(entry.path)}")
