"""Piecewise linear schedules.

A segment ``{from, to, steps}`` uses ``steps`` optimizer steps when ``steps > 1``.
A fraction in ``(0, 1)`` takes that share of the steps still remaining.
``steps: -1`` consumes whatever is left.
"""

from __future__ import annotations

import numpy as np


def piecewise_linear(segments: list[dict], total_steps: int) -> np.ndarray:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    remaining = total_steps
    chunks = []
    for segment in segments:
        count = _segment_length(segment["steps"], remaining)
        if count <= 0:
            continue
        remaining -= count
        chunks.append(np.linspace(float(segment["from"]), float(segment["to"]), count))
        if remaining <= 0:
            break
    if not chunks:
        raise ValueError("Learning-rate schedule produced no steps")
    values = np.concatenate(chunks)
    if len(values) < total_steps:
        values = np.concatenate([values, np.full(total_steps - len(values), values[-1])])
    return values[:total_steps]


def _segment_length(steps, remaining: int) -> int:
    steps = float(steps)
    if steps == -1:
        return remaining
    if steps == 0:
        return 0
    if steps < 1:
        return int(remaining * steps)
    return min(remaining, int(steps))
