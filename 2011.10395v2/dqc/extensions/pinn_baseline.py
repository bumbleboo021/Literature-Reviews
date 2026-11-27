"""Classical Chebyshev-Feature PINN baseline for fair comparison against Quantum DQC.

Maps scalar x in (-1, 1) through a Chebyshev polynomial feature layer
[T_1(x), ..., T_K(x)] into a Tanh MLP with exact floating boundary handling,
mirroring the quantum Chebyshev Tower feature map (Eq. 15 of arXiv:2011.10395v2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn

from dqc.losses import LossType, differential_loss, solution_quality
from dqc.solver import ResidualFn, TrueSolutionFn


class ChebyshevPINN(nn.Module):
    """Classical PINN with Chebyshev input features [T_1(x), ..., T_K(x)] in float64."""

    def __init__(
        self,
        degree: int = 12,
        hidden_dims: tuple[int, ...] = (32, 32),
        seed: int | None = 42,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.degree = degree
        self.dtype = dtype

        layers: list[nn.Module] = []
        in_dim = degree
        for h in hidden_dims:
            lin = nn.Linear(in_dim, h, dtype=dtype)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(nn.Tanh())
            in_dim = h
        out_lin = nn.Linear(in_dim, 1, dtype=dtype)
        nn.init.xavier_uniform_(out_lin.weight)
        nn.init.zeros_(out_lin.bias)
        layers.append(out_lin)
        self.net = nn.Sequential(*layers)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _chebyshev_features(self, x_1d: torch.Tensor) -> torch.Tensor:
        """Map x of shape (M,) -> [T_1(x), ..., T_K(x)] of shape (M, degree)."""
        theta = torch.arccos(x_1d)
        ks = torch.arange(1, self.degree + 1, dtype=self.dtype)
        return torch.cos(theta.unsqueeze(-1) * ks.unsqueeze(0))

    def raw_eval(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.to(dtype=self.dtype)
        scalar = x_in.ndim == 0
        if scalar:
            x_in = x_in.unsqueeze(0)
        feats = self._chebyshev_features(x_in)
        out = self.net(feats).squeeze(-1)
        return out.squeeze(0) if scalar else out

    def eval_with_derivative(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_req = x.detach().clone().to(dtype=self.dtype).requires_grad_(True)
        f_val = self.raw_eval(x_req)
        df_dx = torch.autograd.grad(f_val.sum(), x_req, create_graph=True)[0]
        return f_val, df_dx


@dataclass
class PINNResult:
    """Training history and final model for a PINN run."""

    model: ChebyshevPINN
    x0: float
    u0: float
    loss_full: list[float] = field(default_factory=list)
    loss_quality: list[float] = field(default_factory=list)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x0_t = torch.tensor(self.x0, dtype=self.model.dtype)
            fb = self.u0 - self.model.raw_eval(x0_t)
            return self.model.raw_eval(x) + fb


def train_pinn_adam(
    model: ChebyshevPINN,
    x_grid: torch.Tensor,
    residual_fn: ResidualFn,
    x0: float = 0.0,
    u0: float = 1.0,
    true_solution_fn: TrueSolutionFn | None = None,
    n_iter: int = 250,
    lr: float = 1e-2,
    loss_type: LossType = "mse",
) -> PINNResult:
    """Train a `ChebyshevPINN` with Adam and exact floating boundary handling."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    result = PINNResult(model=model, x0=x0, u0=u0)
    x0_t = torch.tensor(x0, dtype=model.dtype)
    u0_t = torch.tensor(u0, dtype=model.dtype)
    u_true = true_solution_fn(x_grid)[0] if true_solution_fn is not None else None

    for _ in range(n_iter):
        optimizer.zero_grad()
        raw_f, df_dx = model.eval_with_derivative(x_grid)
        f = raw_f + (u0_t - model.raw_eval(x0_t))

        l_full = differential_loss(residual_fn(x_grid, [f], [df_dx]), loss_type=loss_type)
        l_full.backward()
        optimizer.step()

        result.loss_full.append(float(l_full.item()))
        if u_true is not None:
            with torch.no_grad():
                result.loss_quality.append(solution_quality(f.detach(), u_true))

    return result


def train_pinn_lbfgs(
    model: ChebyshevPINN,
    x_grid: torch.Tensor,
    residual_fn: ResidualFn,
    x0: float = 0.0,
    u0: float = 1.0,
    true_solution_fn: TrueSolutionFn | None = None,
    n_iter: int = 200,
    loss_type: LossType = "mse",
) -> PINNResult:
    """Refine a `ChebyshevPINN` with SciPy L-BFGS-B and exact floating boundary handling."""
    result = PINNResult(model=model, x0=x0, u0=u0)
    x0_t = torch.tensor(x0, dtype=model.dtype)
    u0_t = torch.tensor(u0, dtype=model.dtype)
    u_true = true_solution_fn(x_grid)[0] if true_solution_fn is not None else None
    params = list(model.parameters())

    def _pack() -> np.ndarray:
        return np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in params]).astype(np.float64)

    def _unpack(vec: np.ndarray) -> None:
        offset = 0
        with torch.no_grad():
            for p in params:
                n = p.numel()
                p.copy_(torch.tensor(vec[offset : offset + n], dtype=model.dtype).reshape(p.shape))
                offset += n

    def loss_and_grad(vec: np.ndarray) -> tuple[float, np.ndarray]:
        _unpack(vec)
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        raw_f, df_dx = model.eval_with_derivative(x_grid)
        f = raw_f + (u0_t - model.raw_eval(x0_t))
        l_full = differential_loss(residual_fn(x_grid, [f], [df_dx]), loss_type=loss_type)
        l_full.backward()
        grad = np.concatenate([p.grad.detach().cpu().numpy().reshape(-1) for p in params]).astype(np.float64)
        return float(l_full.item()), grad

    def callback(xk: np.ndarray) -> None:
        _unpack(xk)
        raw_f, df_dx = model.eval_with_derivative(x_grid)
        f = raw_f + (u0_t - model.raw_eval(x0_t))
        lf = float(differential_loss(residual_fn(x_grid, [f], [df_dx]), loss_type=loss_type).item())
        result.loss_full.append(lf)
        if u_true is not None:
            with torch.no_grad():
                result.loss_quality.append(solution_quality(f.detach(), u_true))

    opt = minimize(
        loss_and_grad,
        _pack(),
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={"maxiter": n_iter, "ftol": 1e-15, "gtol": 1e-12},
    )
    _unpack(opt.x)
    return result
