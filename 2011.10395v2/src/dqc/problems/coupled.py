"""Two strongly coupled equations, Eqs. (33)-(34)."""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm
from .base import Problem


@dataclass
class CoupledSystem(Problem):
    lambda1: float = 5.0
    lambda2: float = 3.0
    u0: tuple = (0.5, 0.0)
    n_outputs = 2
    labels = ("u1", "u2")

    @property
    def matrix(self):
        return np.array([[self.lambda2, self.lambda1], [-self.lambda1, -self.lambda2]])

    @property
    def initial_values(self):
        return self.u0

    def residual(self, x, e):
        return e.dx-e.value @ self.matrix.T

    def partials(self, x, e):
        shape = (len(x), 2, 2)
        return np.broadcast_to(-self.matrix, shape), np.broadcast_to(np.eye(2), shape)

    def reference(self, x):
        return np.stack([expm(self.matrix*t) @ np.asarray(self.u0) for t in x])

