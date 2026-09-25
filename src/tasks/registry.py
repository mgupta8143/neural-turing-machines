"""The tasks you can train on.

A task is a module with these, and nothing in the training loop, the models or the plots
needs to know which task it is driving:

    INPUT_SIZE, OUTPUT_SIZE   channel counts, which decide the model's input and readout sizes
    TIMESTEPS                 every timestep count a batch can have; the CUDA-graph path
                              captures one graph per entry, so it has to be known up front
    batch(batch_size)         -> x (batch, T, INPUT_SIZE), target (batch, T, OUTPUT_SIZE)
                                 and mask (batch, T), true on the timesteps that are scored
    BITS, PROBE_CASES         the data channels to score and the examples to score them on,
                              which is all src/probe.py needs to log diagnostics for any task

Training only ever calls batch(batch_size). A task also takes min_len and max_len, which the
demo and the task's own figures use to pin an example to a chosen length.
"""

from src.tasks.copy import data as copy_data
from src.tasks.priority_sort import data as priority_sort_data

TASKS = {
    "copy": copy_data,  # paper, Section 4.1
    "priority-sort": priority_sort_data,  # paper, Section 4.5
}


def get_task(name):
    if name not in TASKS:
        raise SystemExit(f"unknown task '{name}': choose from {', '.join(TASKS)}")
    return TASKS[name]
