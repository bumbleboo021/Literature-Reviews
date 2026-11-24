"""Regularization schedules from Sec. III F, Eqs. (28)-(29)."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Constant:
    weight: float = 1.0

    def __post_init__(self):
        if self.weight < 0 or not np.isfinite(self.weight):
            raise ValueError("Weight must be finite and nonnegative.")

    def __call__(self, step):
        return self.weight


@dataclass(frozen=True)
class LinearDecay:
    total_steps: int
    weight: float = 1.0

    def __post_init__(self):
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        Constant(self.weight)

    def __call__(self, step):
        return self.weight*max(0., 1-max(0, step)/self.total_steps)


@dataclass(frozen=True)
class TanhDecay:
    """Eq. (29): weight * [1 - tanh((step-drop)/(width*total))].

    Starts close to 2*weight (not weight); intentionally matches the equation.
    """

    total_steps: int
    drop_step: float
    width: float = 0.05
    weight: float = 1.0

    def __post_init__(self):
        if self.total_steps <= 0 or self.width <= 0 or not np.isfinite(self.drop_step+self.width):
            raise ValueError("Invalid tanh schedule parameters.")
        Constant(self.weight)

    def __call__(self, step):
        return self.weight*(1-np.tanh((step-self.drop_step)/(self.width*self.total_steps)))

