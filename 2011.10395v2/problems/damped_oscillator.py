"""Single damped oscillator ODE — Section IV, Eqs. (30)–(31), Fig. 6.

    du/dx + lam * u * (kappa + tan(lam * x)) = 0,   u(0) = u0

Analytical solution (for u0 = 1):
    u(x) = u0 * exp(-kappa * lam * x) * cos(lam * x)
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class DampedOscillatorProblem:
    """Problem definition for Eqs. (30)–(31)."""

    lam: float = 8.0
    kappa: float = 0.1
    u0: float = 1.0
    x_min: float = 0.0
    x_max: float = 0.9
    n_points: int = 20
    pole_free: bool = True

    def grid(self, n_points: int | None = None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """Equidistant training/evaluation grid on [x_min, x_max]."""
        m = self.n_points if n_points is None else n_points
        return torch.linspace(self.x_min, self.x_max, m, dtype=dtype)

    def residual(
        self, x: torch.Tensor, fs: list[torch.Tensor], dfs: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """ODE residual for Eq. (30).

        Multiplying Eq. (30) by cos(lam * x) gives the algebraically equivalent,
        pole-free form:
            cos(lam * x) * du/dx + lam * u * (kappa * cos(lam * x) + sin(lam * x)) = 0
        which avoids huge MSE spikes near the tan(lam * x) singularities while
        preserving the exact same solution u(x) = u0 * exp(-kappa * lam * x) * cos(lam * x).
        """
        u, du_dx = fs[0], dfs[0]
        if self.pole_free:
            cos_lx = torch.cos(self.lam * x)
            sin_lx = torch.sin(self.lam * x)
            return [cos_lx * du_dx + self.lam * u * (self.kappa * cos_lx + sin_lx)]
        return [du_dx + self.lam * u * (self.kappa + torch.tan(self.lam * x))]

    def true_solution(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Analytical solution u(x) = u0 * exp(-kappa * lam * x) * cos(lam * x) (Eq. 31)."""
        return [self.u0 * torch.exp(-self.kappa * self.lam * x) * torch.cos(self.lam * x)]
