"""Boundary handling strategies from Section III.E of arXiv:2011.10395v2.

Supports all three methods described in the paper:
  - 'pinned'    (Eq. 24): f(x) = <C>(x), boundary enforced via loss term eta*(f(x0) - u0)^2
  - 'floating'  (Eq. 25-26): f(x) = (u0 - <C>(x0)) + <C>(x), exact match at x0, no boundary loss
  - 'optimized' (Eq. 27): f(x) = f_c + <C>(x), f_c trained via Adam + boundary loss term
"""

from __future__ import annotations

from typing import Literal
import torch

from dqc.circuit import DifferentiableQuantumCircuit

BoundaryMode = Literal["pinned", "floating", "optimized"]


class BoundaryHandler:
    """Wraps a DQC to enforce an initial/boundary condition u(x0) = u0."""

    def __init__(
        self,
        mode: BoundaryMode = "floating",
        x0: float = 0.0,
        u0: float = 1.0,
        eta: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if mode not in ("pinned", "floating", "optimized"):
            raise ValueError(f"Unknown boundary mode '{mode}'. Choose pinned, floating, or optimized.")
        self.mode: BoundaryMode = mode
        self.x0 = torch.tensor(x0, dtype=dtype)
        self.u0 = torch.tensor(u0, dtype=dtype)
        self.eta = float(eta)
        self.dtype = dtype

        self.fc: torch.Tensor | None = None
        if self.mode == "optimized":
            self.fc = torch.tensor(0.0, dtype=dtype, requires_grad=True)

    def extra_params(self) -> list[torch.Tensor]:
        """Return any additional classical parameters to optimize (only `fc` for optimized mode)."""
        return [self.fc] if self.fc is not None else []

    def apply(
        self, dqc: DifferentiableQuantumCircuit, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate boundary-adjusted (f(x), df/dx, boundary_loss) on grid x."""
        raw_f, df_dx = dqc.eval_with_derivative(x)
        zero = torch.tensor(0.0, dtype=self.dtype)

        if self.mode == "pinned":
            f0 = dqc.raw_eval(self.x0)
            b_loss = self.eta * (f0 - self.u0) ** 2
            return raw_f, df_dx, b_loss

        if self.mode == "floating":
            # Eq. (25)-(26): f_b = u0 - <C>(x0), f(x) = f_b + <C>(x)
            f0_raw = dqc.raw_eval(self.x0)
            fb = self.u0 - f0_raw
            return raw_f + fb, df_dx, zero

        # mode == "optimized" (Eq. 27)
        assert self.fc is not None
        f0 = dqc.raw_eval(self.x0) + self.fc
        b_loss = self.eta * (f0 - self.u0) ** 2
        return raw_f + self.fc, df_dx, b_loss

    def eval_only(self, dqc: DifferentiableQuantumCircuit, x: torch.Tensor) -> torch.Tensor:
        """Evaluate f(x) without computing df/dx (useful for plotting / evaluation)."""
        raw_f = dqc.raw_eval(x)
        if self.mode == "pinned":
            return raw_f
        if self.mode == "floating":
            fb = self.u0 - dqc.raw_eval(self.x0)
            return raw_f + fb
        assert self.fc is not None
        return raw_f + self.fc
