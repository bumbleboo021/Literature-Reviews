"""Strongly coupled ODE system — Section IV, Eqs. (33)–(34), Fig. 8.

    du1/dx - lam1 * u2 - lam2 * u1 = 0,   u1(0) = u1_0
    du2/dx + lam2 * u2 + lam1 * u1 = 0,   u2(0) = u2_0

For |lam1| > |lam2| (paper uses lam1 = 5, lam2 = 3, u1_0 = 0.5, u2_0 = 0),
let omega = sqrt(lam1^2 - lam2^2) = 4. The exact solution is:
    u1(x) = u1_0 * cos(omega * x) + ((lam2 * u1_0 + lam1 * u2_0) / omega) * sin(omega * x)
    u2(x) = u2_0 * cos(omega * x) - ((lam1 * u1_0 + lam2 * u2_0) / omega) * sin(omega * x)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class CoupledODEProblem:
    """Problem definition for Eqs. (33)–(34)."""

    lam1: float = 5.0
    lam2: float = 3.0
    u1_0: float = 0.5
    u2_0: float = 0.0
    x_min: float = 0.0
    x_max: float = 0.9
    n_points: int = 20

    def grid(self, n_points: int | None = None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        m = self.n_points if n_points is None else n_points
        return torch.linspace(self.x_min, self.x_max, m, dtype=dtype)

    def residual(
        self, x: torch.Tensor, fs: list[torch.Tensor], dfs: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        u1, u2 = fs[0], fs[1]
        du1, du2 = dfs[0], dfs[1]
        f1 = du1 - self.lam1 * u2 - self.lam2 * u1
        f2 = du2 + self.lam2 * u2 + self.lam1 * u1
        return [f1, f2]

    def true_solution(self, x: torch.Tensor) -> list[torch.Tensor]:
        omega = math.sqrt(self.lam1**2 - self.lam2**2)
        cos_wx = torch.cos(omega * x)
        sin_wx = torch.sin(omega * x)
        u1 = self.u1_0 * cos_wx + ((self.lam2 * self.u1_0 + self.lam1 * self.u2_0) / omega) * sin_wx
        u2 = self.u2_0 * cos_wx - ((self.lam1 * self.u1_0 + self.lam2 * self.u2_0) / omega) * sin_wx
        return [u1, u2]
