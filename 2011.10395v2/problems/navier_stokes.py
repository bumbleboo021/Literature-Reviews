"""Quasi-1D stationary Navier-Stokes convergent-divergent nozzle — Section IV, Eqs. (40)–(50), Fig. 10.

Nozzle shape (Eq. 43):
    A(x) = 1 + 4.95 * (2x - 1)^2,   0 <= x <= 1

Stationary inviscid quasi-1D Navier-Stokes equations (Eqs. 40–42 with d/dt = 0, gamma = 1.4):
    F_rho = rho * dV/dx + rho * V * d(ln A)/dx + V * drho/dx = 0
    F_T   = V * dT/dx + (gamma - 1) * T * (dV/dx + V * d(ln A)/dx) = 0
    F_V   = rho * V * dV/dx + (1 / gamma) * (rho * dT/dx + T * drho/dx) = 0

Boundary conditions (Eq. 44):
    rho(0) = 1.0,  T(0) = 1.0,  V(0) = 0.1
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
import torch


@dataclass(frozen=True)
class NavierStokesNozzleProblem:
    """Problem definition for the convergent-divergent nozzle (Fig. 10)."""

    gamma: float = 1.4
    rho0: float = 1.0
    T0: float = 1.0
    V0: float = 0.1

    @staticmethod
    def area(x: torch.Tensor) -> torch.Tensor:
        """Nozzle cross-sectional area A(x) = 1 + 4.95*(2x - 1)^2 (Eq. 43)."""
        return 1.0 + 4.95 * (2.0 * x - 1.0) ** 2

    @staticmethod
    def d_ln_area_dx(x: torch.Tensor) -> torch.Tensor:
        """Analytical derivative d(ln A)/dx = A'(x) / A(x)."""
        area = 1.0 + 4.95 * (2.0 * x - 1.0) ** 2
        d_area = 19.8 * (2.0 * x - 1.0)
        return d_area / area

    def residual(
        self, x: torch.Tensor, fs: list[torch.Tensor], dfs: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Pole-free stationary Navier-Stokes residuals for [rho, T, V]."""
        rho, temp, vel = fs[0], fs[1], fs[2]
        drho, dtemp, dvel = dfs[0], dfs[1], dfs[2]
        d_lna = self.d_ln_area_dx(x)

        f_rho = rho * dvel + rho * vel * d_lna + vel * drho
        f_temp = vel * dtemp + (self.gamma - 1.0) * temp * (dvel + vel * d_lna)
        f_vel = rho * vel * dvel + (1.0 / self.gamma) * (rho * dtemp + temp * drho)
        return [f_rho, f_temp, f_vel]

    def true_solution(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Exact analytical isentropic subsonic-supersonic nozzle solution (Anderson [2]).

        Solves the area-Mach relation at every x (subsonic M < 1 for x < 0.5,
        sonic M = 1 at x = 0.5, supersonic M > 1 for x > 0.5), scaled to match
        the inlet stagnation state.
        """
        g = self.gamma
        exp_pow = (g + 1.0) / (g - 1.0)
        x_np = x.detach().cpu().numpy().astype(np.float64)
        areas = 1.0 + 4.95 * (2.0 * x_np - 1.0) ** 2

        def area_mach_eq(mach: float, a_ratio: float) -> float:
            return (1.0 / mach**2) * (
                (2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * mach**2)
            ) ** exp_pow - a_ratio**2

        machs = np.empty_like(x_np)
        for i, (xi, a_i) in enumerate(zip(x_np, areas)):
            if abs(xi - 0.5) < 1e-12:
                machs[i] = 1.0
            elif xi < 0.5:
                machs[i] = brentq(area_mach_eq, 1e-6, 1.0 - 1e-9, args=(a_i,))
            else:
                machs[i] = brentq(area_mach_eq, 1.0 + 1e-9, 10.0, args=(a_i,))

        # Isentropic relations normalized to reservoir (stagnation) conditions
        # (rho_0 = 1, T_0 = 1, V = M * sqrt(T), matching Anderson [2] & Fig. 9c/10b)
        temp = 1.0 / (1.0 + 0.5 * (g - 1.0) * machs**2)
        rho = temp ** (1.0 / (g - 1.0))
        vel = machs * np.sqrt(temp)

        return [
            torch.tensor(rho, dtype=x.dtype),
            torch.tensor(temp, dtype=x.dtype),
            torch.tensor(vel, dtype=x.dtype),
        ]
