"""Highly nontrivial nonlinear ODE — Section IV, Eq. (32), Fig. 7.

    du/dx - 4*u + 6*u^2 - sin(50*x) - u*cos(25*x) + 0.5 = 0,   u(0) = 0.75

No closed-form solution exists; reference values are computed to high precision
with `scipy.integrate.solve_ivp(..., method="DOP853")`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
import torch


@dataclass(frozen=True)
class NontrivialODEProblem:
    """Problem definition for Eq. (32)."""

    u0: float = 0.75
    x_min: float = 0.0
    x_max: float = 0.9
    n_points: int = 100

    def grid(self, n_points: int | None = None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        m = self.n_points if n_points is None else n_points
        return torch.linspace(self.x_min, self.x_max, m, dtype=dtype)

    def residual(
        self, x: torch.Tensor, fs: list[torch.Tensor], dfs: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        u, du_dx = fs[0], dfs[0]
        return [du_dx - 4.0 * u + 6.0 * u**2 - torch.sin(50.0 * x) - u * torch.cos(25.0 * x) + 0.5]

    def true_solution(self, x: torch.Tensor) -> list[torch.Tensor]:
        """High-accuracy numerical reference via DOP853."""
        x_np = x.detach().cpu().numpy().astype(np.float64)

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            u = y[0]
            return np.array([4.0 * u - 6.0 * u**2 + np.sin(50.0 * t) + u * np.cos(25.0 * t) - 0.5])

        sol = solve_ivp(
            rhs,
            t_span=(float(self.x_min), float(max(self.x_max, x_np.max()))),
            y0=[self.u0],
            t_eval=x_np,
            method="DOP853",
            rtol=1e-11,
            atol=1e-13,
        )
        if not sol.success:
            raise RuntimeError(f"Reference ODE integration failed: {sol.message}")
        return [torch.tensor(sol.y[0], dtype=x.dtype)]
