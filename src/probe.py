"""Diagnostics logged beside the learning curve.

The cost curve alone cannot tell you that a model has learned something that will not survive
a longer sequence. Repeat copy reached 0.0001 bits and was still wrong on a fifth of the bits
at twice the length, and the reason was only visible in the write head: it kept a sharply
peaked weighting through the output phase, so it erased its own data once the sequence ran
past the 128 memory locations. The copy task, which generalises, lets the weighting go diffuse
instead (peak 0.41 against 0.95).

So every run records, next to the cost: what the write head does after the input ends, and the
error beyond the training range. Both are cheap, and either would have caught that immediately.
"""

import torch


def write_head_stats(model, x, input_steps):
    """What the write head did during the output phase of this one sequence."""
    trace = {}
    with torch.no_grad():
        model(x, trace=trace)

    head = model.write_heads[0]
    weightings = torch.stack(trace["write_weightings"])[input_steps:]  # (steps, locations)
    parameters = torch.stack(trace["write_parameters"])[input_steps:]
    key, strength, gate, shifts, sharpness, erase, add = torch.split(parameters, head.sizes, dim=1)

    where = weightings.argmax(dim=1)
    hops = (where[1:] - where[:-1]) % model.memory_locations
    return {
        "gate": torch.sigmoid(gate).mean().item(),  # 0 = ignore content, stay where you were
        "peak_w": weightings.max(dim=1).values.mean().item(),  # 1 = one location takes the write
        "erase": torch.sigmoid(erase).mean().item(),
        "hop": torch.minimum(hops, model.memory_locations - hops).float().mean().item(),
    }


def bit_error(model, x, target, mask, channels):
    with torch.no_grad():
        bits = (model(x)[0][mask[0]][:, :channels] > 0).float()
    return 100 * (bits != target[0][mask[0]][:, :channels]).float().mean().item()


def probe(model, task, device="cpu"):
    """One diagnostics row, or None for a task or model that does not support it."""
    if not hasattr(model, "write_heads") or not hasattr(task, "PROBE_CASES"):
        return None
    was_training = model.training
    model.eval()
    row = {}
    for i, (label, build) in enumerate(task.PROBE_CASES):
        x, target, mask = (t.to(device) for t in build())
        row[f"{label}_err"] = bit_error(model, x, target, mask, task.BITS)
        if i == 0:  # head behaviour only needs the in-range case
            row.update(write_head_stats(model, x, int((~mask[0]).sum())))
    model.train(was_training)
    return row
