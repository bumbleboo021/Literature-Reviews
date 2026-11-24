"""Pinned, floating, optimized initial/boundary value handling, Eqs. (23)-(27)."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Boundary:
    kind: str = "floating"
    x0: float = 0.0
    values: object = (1.0,)

    def __post_init__(self):
        self.kind = self.kind.lower()
        if self.kind not in ("pinned", "floating", "optimized"):
            raise ValueError("Boundary must be pinned, floating, or optimized.")
        self.values = np.atleast_1d(np.asarray(self.values, float)).copy()
        if self.values.ndim != 1 or not len(self.values) or not np.all(np.isfinite(self.values)):
            raise ValueError("Boundary values must be a finite nonempty vector.")
        if not np.isfinite(self.x0):
            raise ValueError("Boundary input must be finite.")

