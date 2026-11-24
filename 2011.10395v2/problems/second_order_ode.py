"""Second-order damped harmonic oscillator ODE (extension benchmark).

    d^2u/dx^2 + 2*gamma * du/dx + omega0^2 * u = 0,   u(0) = 1,  u'(0) = 0

Exact analytical solution (underdamped: omega_d = sqrt(omega0^2 - gamma^2)):
    u(x) = exp(-gamma * x) * (cos(omega_d * x) + (gamma / omega_d) * sin(omega_d * x))
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class SecondOrderHarmonicProblem:
    """Underdamped harmonic oscillator with exact initial value & slope."""

    gamma: float = 1.0
    omega0: float = 6.0
    u0: float = 1.0
    du0: float = 0.0
    x_min: float = 0.0
    x_max: float = 0.9
    n_points: int = 25

    def grid(self, n_points: int | None = None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        m = self.n_points if n_points is None else n_points
        return torch.linspace(self.x_min, self.x_max, m, dtype=dtype)

    def residual(
        self, x: torch.Tensor, u: torch.Tensor, du_dx: torch.Tensor, d2u_dx2: torch.Tensor
    ) -> torch.Tensor:
        return d2u_dx2 + 2.0 * self.gamma * du_dx + (self.omega0**2) * u

    def true_solution(self, x: torch.Tensor) -> torch.Tensor:
        omega_d = math.sqrt(self.omega0**2 - self.gamma**2)
        c1 = self.u0
        c2 = (self.du0 + self.gamma * self.u0) / omega_d
        return torch.exp(-self.gamma * x) * (c1 * torch.cos(omega_d * x) + c2 * torch.sin(omega_d * x))
