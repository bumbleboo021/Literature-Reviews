"""Collocation grids, including the split nozzle regions on pp. 16-17."""

import numpy as np


def collocation(start, stop, count, kind="uniform", seed=0):
    if not np.isfinite(start+stop) or stop <= start or not isinstance(count, int) or count < 2:
        raise ValueError("Need finite start < stop and integer count >= 2.")
    if kind == "uniform":
        return np.linspace(start, stop, count)
    if kind == "chebyshev":
        nodes = -np.cos(np.linspace(0, np.pi, count))
        return start+(stop-start)*(nodes+1)/2
    if kind == "random":
        return np.sort(np.random.default_rng(seed).uniform(start, stop, count))
    raise ValueError("Grid must be uniform, chebyshev, or random.")


def nozzle_stage_grid(stage=1, points_per_region=20):
    if stage == 1:
        return collocation(0, 0.4, points_per_region)
    if stage == 2:
        return np.r_[collocation(0, 0.4, points_per_region), collocation(0.6, 0.9, points_per_region)]
    raise ValueError("Nozzle stage must be 1 or 2.")
