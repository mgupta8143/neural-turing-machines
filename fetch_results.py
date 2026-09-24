"""Download every result from the Modal volume, in case a training client dropped.

    uv run --with modal python fetch_results.py

Training writes its log, checkpoint and run.json into a Modal volume as it goes, so results
survive even if the laptop that launched the run goes to sleep. This copies whatever is there
into results/<task>/<model>/, and is safe to re-run.
"""

import pathlib

import modal

volume = modal.Volume.from_name("ntm-results", create_if_missing=True)

for entry in volume.listdir("/", recursive=True):
    if entry.type.name != "FILE":
        continue
    local = pathlib.Path("results") / entry.path
    local.parent.mkdir(parents=True, exist_ok=True)
    with open(local, "wb") as f:
        for chunk in volume.read_file(entry.path):
            f.write(chunk)
    print(f"{entry.path} -> {local}")
