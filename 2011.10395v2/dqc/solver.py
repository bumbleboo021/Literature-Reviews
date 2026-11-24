"""Hybrid quantum-classical Adam training loop — Section III.H & Fig. 1."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import torch

from dqc.boundary import BoundaryHandler
from dqc.circuit import DifferentiableQuantumCircuit
from dqc.losses import LossType, differential_loss, solution_quality

# ResidualFn takes (x_grid, list_of_f, list_of_df_dx) -> list of ODE residuals [F_1, ..., F_K]
ResidualFn = Callable[
    [torch.Tensor, list[torch.Tensor], list[torch.Tensor]],
    list[torch.Tensor],
]
# TrueSolutionFn takes x_grid -> list of true solution tensors [u_1(x), ..., u_K(x)]
TrueSolutionFn = Callable[[torch.Tensor], list[torch.Tensor]]
# RegLossFn takes (step, circuits, boundaries) -> scalar regularization loss tensor
RegLossFn = Callable[
    [int, list[DifferentiableQuantumCircuit], list[BoundaryHandler]],
    torch.Tensor,
]


@dataclass
class SolveResult:
    """Container for training history metrics (L_F, L_D, L_Q) and final circuits."""

    circuits: list[DifferentiableQuantumCircuit]
    boundaries: list[BoundaryHandler]
    loss_full: list[float] = field(default_factory=list)
    loss_diff: list[float] = field(default_factory=list)
    loss_quality: list[float] = field(default_factory=list)

    def predict(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Evaluate all trained trial functions f_k(x) without tracking gradients."""
        with torch.no_grad():
            return [b.eval_only(c, x) for c, b in zip(self.circuits, self.boundaries)]


class DQCSolver:
    """Hybrid quantum-classical DQC ODE solver using Adam (Fig. 1 workflow)."""

    def __init__(
        self,
        circuits: DifferentiableQuantumCircuit | list[DifferentiableQuantumCircuit],
        boundaries: BoundaryHandler | list[BoundaryHandler],
        residual_fn: ResidualFn,
        true_solution_fn: TrueSolutionFn | None = None,
        reg_loss_fn: RegLossFn | None = None,
    ) -> None:
        self.circuits = [circuits] if isinstance(circuits, DifferentiableQuantumCircuit) else list(circuits)
        self.boundaries = [boundaries] if isinstance(boundaries, BoundaryHandler) else list(boundaries)
        if len(self.circuits) != len(self.boundaries):
            raise ValueError("circuits and boundaries must have the same length")
        self.residual_fn = residual_fn
        self.true_solution_fn = true_solution_fn
        self.reg_loss_fn = reg_loss_fn

    def _trainable_params(self) -> list[torch.Tensor]:
        params: list[torch.Tensor] = []
        for c, b in zip(self.circuits, self.boundaries):
            params.append(c.params)
            params.extend(b.extra_params())
        return params

    def solve(
        self,
        x_grid: torch.Tensor,
        n_iter: int = 250,
        lr: float = 0.01,
        loss_type: LossType = "mse",
        log_every: int = 0,
    ) -> SolveResult:
        """Run the Adam optimization loop for `n_iter` iterations."""
        optimizer = torch.optim.Adam(self._trainable_params(), lr=lr)
        result = SolveResult(circuits=self.circuits, boundaries=self.boundaries)

        u_trues = self.true_solution_fn(x_grid) if self.true_solution_fn is not None else None

        for step in range(n_iter):
            optimizer.zero_grad()

            fs: list[torch.Tensor] = []
            dfs: list[torch.Tensor] = []
            b_losses: list[torch.Tensor] = []

            for circuit, boundary in zip(self.circuits, self.boundaries):
                f_k, df_k, b_loss_k = boundary.apply(circuit, x_grid)
                fs.append(f_k)
                dfs.append(df_k)
                b_losses.append(b_loss_k)

            residuals = self.residual_fn(x_grid, fs, dfs)
            l_diff = differential_loss(residuals, loss_type=loss_type)
            l_boundary = sum(b_losses)
            l_d = l_diff + l_boundary

            if self.reg_loss_fn is not None:
                l_reg = self.reg_loss_fn(step, self.circuits, self.boundaries)
                l_full = l_d + l_reg
            else:
                l_full = l_d

            l_full.backward()
            optimizer.step()

            result.loss_full.append(float(l_full.item()))
            result.loss_diff.append(float(l_d.item()))
            if u_trues is not None:
                with torch.no_grad():
                    lq = solution_quality([f.detach() for f in fs], u_trues)
                result.loss_quality.append(lq)

            if log_every > 0 and (step % log_every == 0 or step == n_iter - 1):
                msg = f"iter {step + 1:4d}/{n_iter} | L_F = {result.loss_full[-1]:.3e}"
                if result.loss_quality:
                    msg += f" | L_Q = {result.loss_quality[-1]:.3e}"
                print(msg, flush=True)

        return result
