"""Loss functions, regularization schedules, and evaluation metrics.

Maps to Sections III.D (Eqs. 19–22), III.F (Eqs. 28–29), and IV (L_F, L_D, L_Q)
of arXiv:2011.10395v2.
"""

from __future__ import annotations

import math
from typing import Literal
import torch

LossType = Literal["mse", "mae"]


def distance(a: torch.Tensor, b: torch.Tensor | float, loss_type: LossType = "mse") -> torch.Tensor:
    """Pointwise distance L(a, b) — Eq. (22) for MSE or |a - b| for MAE."""
    if loss_type == "mse":
        return (a - b) ** 2
    if loss_type == "mae":
        return torch.abs(a - b)
    raise ValueError(f"Unknown loss_type '{loss_type}'. Choose 'mse' or 'mae'.")


def differential_loss(
    residuals: torch.Tensor | list[torch.Tensor],
    loss_type: LossType = "mse",
) -> torch.Tensor:
    """Differential loss L^(diff) — Eq. (20).

    For a single ODE residual F of shape (M,), returns mean(L(F, 0)).
    For a system of K ODEs [F_1, ..., F_K], sums the per-equation mean losses.
    """
    if isinstance(residuals, torch.Tensor):
        residuals = [residuals]
    return sum(torch.mean(distance(res, 0.0, loss_type=loss_type)) for res in residuals)


def regularization_schedule(
    step: int,
    n_iter: int,
    kind: Literal["linear", "reverse_sigmoid"] = "reverse_sigmoid",
    n_drop: int = 150,
    delta: float = 0.1,
) -> float:
    """Iteration-dependent regularization weight zeta(n_j) — Section III.F, Eq. (29).

    - 'linear':          1 - step / n_iter
    - 'reverse_sigmoid': 1 - tanh((step - n_drop) / (delta * n_iter))
    """
    if kind == "linear":
        return max(0.0, 1.0 - step / max(1, n_iter))
    if kind == "reverse_sigmoid":
        return float(1.0 - math.tanh((step - n_drop) / (delta * max(1, n_iter))))
    raise ValueError(f"Unknown schedule kind '{kind}'")


def regularization_loss(
    f_reg: torch.Tensor,
    u_reg: torch.Tensor,
    weight: float,
    loss_type: LossType = "mse",
) -> torch.Tensor:
    """Regularization loss L^(reg) — Eq. (28)."""
    if weight <= 0.0:
        return torch.tensor(0.0, dtype=f_reg.dtype)
    return weight * torch.sum(distance(f_reg, u_reg, loss_type=loss_type))


def solution_quality(
    f_preds: torch.Tensor | list[torch.Tensor],
    u_trues: torch.Tensor | list[torch.Tensor],
) -> float:
    """Quality of solution L_Q = (1/M) sum_i (f(x_i) - u(x_i))^2 (Section IV, p. 12).

    For multi-function systems, returns the sum of per-function MSEs against true solutions.
    """
    if isinstance(f_preds, torch.Tensor):
        f_preds = [f_preds]
    if isinstance(u_trues, torch.Tensor):
        u_trues = [u_trues]
    total = sum(torch.mean((fp - ut) ** 2).item() for fp, ut in zip(f_preds, u_trues))
    return float(total)
