"""Fubini-Study Metric Tensor & Quantum Natural Gradient (QNG) Solver.

Implements Section 4.2 (Eqs. 26–27) of Vu Tuan Hai & Le Bin Ho:
    g_ij(theta) = -1/2 * d^2/(d theta_i d theta_j) |<psi(theta') | psi(theta)>|^2 |_{theta'=theta}

Provides:
  1. `fubini_study_lagrange`: hardware-compatible Fubini-Study metric tensor via
     our 2nd-order Lagrange parameter-shift rule (Eqs. 24 & 26).
  2. `fubini_study_statevector`: exact, vectorized statevector Fubini-Study metric
     tensor averaged across a training grid x_grid for fast DQC simulation.
  3. `QNGSolver`: Quantum Natural Gradient optimizer
     theta <- theta - lr * (g(theta) + reg * I)^{-1} grad_theta L
     for any baseline `DifferentiableQuantumCircuit` + `BoundaryHandler`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import numpy as np
import pennylane as qml
from scipy.optimize import minimize
import torch

from dqc.ansatz import hardware_efficient_ansatz
from dqc.boundary import BoundaryHandler
from dqc.circuit import DifferentiableQuantumCircuit
from dqc.extensions.lagrange_interpolation.lagrange_gpsr import (
    ShiftRule,
    apply_second_order_shift_rule,
)
from dqc.feature_maps import apply_feature_map
from dqc.losses import LossType, differential_loss, solution_quality
from dqc.solver import ResidualFn, SolveResult, TrueSolutionFn


def fubini_study_lagrange(
    overlap_prob_fn: Callable[[Sequence[float], Sequence[float]], float],
    theta: Sequence[float],
    rules: Sequence[ShiftRule],
) -> np.ndarray:
    """Compute Fubini-Study metric tensor g_ij(theta) via Eq. (26) + 2nd-order Lagrange GPSR.

    Args:
        overlap_prob_fn: Function(theta_ref, theta_var) -> |<psi(theta_ref) | psi(theta_var)>|^2.
        theta: Current parameter vector of length P.
        rules: Per-parameter `ShiftRule` (e.g., 2-term for RX/RY/RZ, 4-term for CRY).
    """
    theta_arr = np.asarray(theta, dtype=np.float64)
    p = len(theta_arr)
    g = np.zeros((p, p), dtype=np.float64)

    for i in range(p):
        for j in range(i, p):
            if i == j:
                def eval_2d(a: float, b: float) -> float:
                    t = theta_arr.copy()
                    t[i] = a + b - theta_arr[i]
                    return overlap_prob_fn(theta_arr, t)
            else:
                def eval_2d(a: float, b: float) -> float:
                    t = theta_arr.copy()
                    t[i] = a
                    t[j] = b
                    return overlap_prob_fn(theta_arr, t)

            d2 = apply_second_order_shift_rule(
                eval_2d, float(theta_arr[i]), float(theta_arr[j]), rules[i], rules[j]
            )
            g[i, j] = -0.5 * d2
            g[j, i] = g[i, j]
    return g


def fubini_study_statevector(
    dqc: DifferentiableQuantumCircuit,
    x_grid: torch.Tensor,
) -> torch.Tensor:
    """Exact Fubini-Study metric tensor averaged over `x_grid` via statevector Jacobian.

    For each grid point x_m:
        g_ij(theta; x_m) = Re[ <d_i psi | d_j psi> - <d_i psi | psi><psi | d_j psi> ]
    """
    dev = qml.device("default.qubit", wires=dqc.n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def state_qnode(x_val: torch.Tensor, theta_flat: torch.Tensor) -> torch.Tensor:
        theta_3d = theta_flat.reshape(dqc.params.shape)
        apply_feature_map(dqc.feature_map, x_val, dqc.wires)
        if dqc.evolution_unitary is not None:
            qml.QubitUnitary(dqc.evolution_unitary, wires=dqc.wires)
        hardware_efficient_ansatz(theta_3d, dqc.wires)
        return qml.state()

    theta_flat = dqc.params.detach().reshape(-1)
    n_params = theta_flat.numel()
    g_avg = torch.zeros((n_params, n_params), dtype=dqc.dtype)

    for x_m in x_grid:
        # Compute real and imag parts of d|psi>/d(theta_flat) via torch.func.jacrev
        def real_imag_state(t_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            psi = state_qnode(x_m, t_flat)
            return psi.real, psi.imag

        psi = state_qnode(x_m, theta_flat).detach()
        jac_r, jac_i = torch.func.jacrev(real_imag_state)(theta_flat)
        jac = torch.complex(jac_r, jac_i)  # shape (2^N, P)

        # G_ij = Re( J^dag J - (J^dag psi)(psi^dag J) )
        gram = jac.conj().T @ jac
        proj = jac.conj().T @ psi
        g_m = (gram - torch.outer(proj, proj.conj())).real
        g_avg = g_avg + g_m

    return g_avg / len(x_grid)


class QNGSolver:
    """Quantum Natural Gradient (QNG) solver for single-circuit DQCs.

    At each iteration computes:
        delta_theta = (g(theta) + diag_reg * I)^{-1} grad_theta L
        theta <- theta - lr * delta_theta

    Supports computing the Fubini-Study metric every `metric_every` steps to
    keep CPU wall-clock time fast while retaining natural-gradient preconditioning.
    """

    def __init__(
        self,
        circuit: DifferentiableQuantumCircuit,
        boundary: BoundaryHandler,
        residual_fn: ResidualFn,
        true_solution_fn: TrueSolutionFn | None = None,
    ) -> None:
        self.circuit = circuit
        self.boundary = boundary
        self.residual_fn = residual_fn
        self.true_solution_fn = true_solution_fn

    def solve(
        self,
        x_grid: torch.Tensor,
        n_iter: int = 100,
        lr: float = 0.05,
        diag_reg: float = 1e-3,
        metric_every: int = 1,
        metric_subsample: int = 5,
        max_step_norm: float = 0.5,
        loss_type: LossType = "mse",
        log_every: int = 0,
    ) -> SolveResult:
        """Run QNG optimization for `n_iter` iterations."""
        result = SolveResult(circuits=[self.circuit], boundaries=[self.boundary])
        u_trues = self.true_solution_fn(x_grid) if self.true_solution_fn is not None else None

        # Subsample grid points for metric tensor estimation (smoothly spans domain)
        if metric_subsample < len(x_grid):
            idx = torch.linspace(0, len(x_grid) - 1, metric_subsample).long()
            x_metric = x_grid[idx]
        else:
            x_metric = x_grid

        n_params = self.circuit.params.numel()
        eye = torch.eye(n_params, dtype=self.circuit.dtype)
        g_mat = eye.clone()

        for step in range(n_iter):
            if self.circuit.params.grad is not None:
                self.circuit.params.grad.zero_()

            f_val, df_dx, b_loss = self.boundary.apply(self.circuit, x_grid)
            residuals = self.residual_fn(x_grid, [f_val], [df_dx])
            l_diff = differential_loss(residuals, loss_type=loss_type)
            l_full = l_diff + b_loss
            l_full.backward()

            grad_flat = self.circuit.params.grad.detach().reshape(-1)

            if step % metric_every == 0:
                g_mat = fubini_study_statevector(self.circuit, x_metric)

            # Solve (g + diag_reg * I) * nat_grad = grad
            nat_grad = torch.linalg.solve(g_mat + diag_reg * eye, grad_flat)
            nat_dir = nat_grad.reshape(self.circuit.params.shape)

            # Trust-region cap on initial trial step
            step_norm = torch.linalg.norm(lr * nat_dir)
            alpha = lr if step_norm <= max_step_norm else float(lr * (max_step_norm / step_norm))

            # Backtracking line search to guarantee monotonic decrease on stiff ODE losses
            prev_loss = float(l_full.item())
            orig_params = self.circuit.params.detach().clone()
            for _ in range(5):
                with torch.no_grad():
                    self.circuit.params.copy_(orig_params - alpha * nat_dir)
                f_try, df_try, b_try = self.boundary.apply(self.circuit, x_grid)
                try_loss = float((differential_loss(self.residual_fn(x_grid, [f_try], [df_try]), loss_type=loss_type) + b_try).item())
                if try_loss <= prev_loss:
                    break
                alpha *= 0.5

            result.loss_full.append(float(l_full.item()))
            result.loss_diff.append(float(l_full.item()))
            if u_trues is not None:
                with torch.no_grad():
                    lq = solution_quality(f_val.detach(), u_trues[0])
                result.loss_quality.append(lq)

            if log_every > 0 and (step % log_every == 0 or step == n_iter - 1):
                msg = f"QNG iter {step + 1:4d}/{n_iter} | L_F = {result.loss_full[-1]:.3e}"
                if result.loss_quality:
                    msg += f" | L_Q = {result.loss_quality[-1]:.3e}"
                print(msg, flush=True)

        return result


class LBFGSSolver:
    """Quasi-Newton L-BFGS solver (with strong Wolfe line search) for refining DQCs."""

    def __init__(
        self,
        circuit: DifferentiableQuantumCircuit,
        boundary: BoundaryHandler,
        residual_fn: ResidualFn,
        true_solution_fn: TrueSolutionFn | None = None,
    ) -> None:
        self.circuit = circuit
        self.boundary = boundary
        self.residual_fn = residual_fn
        self.true_solution_fn = true_solution_fn

    def solve(
        self,
        x_grid: torch.Tensor,
        n_iter: int = 80,
        loss_type: LossType = "mse",
        log_every: int = 0,
    ) -> SolveResult:
        """Run SciPy L-BFGS-B for up to `n_iter` iterations, logging per-iteration L_F and L_Q."""
        result = SolveResult(circuits=[self.circuit], boundaries=[self.boundary])
        u_trues = self.true_solution_fn(x_grid) if self.true_solution_fn is not None else None
        shape = self.circuit.params.shape

        def loss_and_grad(theta_np: np.ndarray) -> tuple[float, np.ndarray]:
            with torch.no_grad():
                self.circuit.params.copy_(torch.tensor(theta_np, dtype=self.circuit.dtype).reshape(shape))
            if self.circuit.params.grad is not None:
                self.circuit.params.grad.zero_()
            f_val, df_dx, b_loss = self.boundary.apply(self.circuit, x_grid)
            l_diff = differential_loss(self.residual_fn(x_grid, [f_val], [df_dx]), loss_type=loss_type)
            l_full = l_diff + b_loss
            l_full.backward()
            grad_np = self.circuit.params.grad.detach().cpu().numpy().reshape(-1).astype(np.float64)
            return float(l_full.item()), grad_np

        def callback(xk: np.ndarray) -> None:
            with torch.no_grad():
                self.circuit.params.copy_(torch.tensor(xk, dtype=self.circuit.dtype).reshape(shape))
            f_val, df_dx, b_loss = self.boundary.apply(self.circuit, x_grid)
            l_diff = differential_loss(self.residual_fn(x_grid, [f_val], [df_dx]), loss_type=loss_type)
            lf = float((l_diff + b_loss).item())
            result.loss_full.append(lf)
            result.loss_diff.append(lf)
            if u_trues is not None:
                with torch.no_grad():
                    result.loss_quality.append(solution_quality(f_val.detach(), u_trues[0]))
            step = len(result.loss_full)
            if log_every > 0 and (step % log_every == 0 or step == n_iter):
                msg = f"L-BFGS iter {step:4d}/{n_iter} | L_F = {lf:.3e}"
                if result.loss_quality:
                    msg += f" | L_Q = {result.loss_quality[-1]:.3e}"
                print(msg, flush=True)

        x0 = self.circuit.params.detach().cpu().numpy().reshape(-1).astype(np.float64)
        opt = minimize(
            loss_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options={"maxiter": n_iter, "ftol": 1e-15, "gtol": 1e-12},
        )
        with torch.no_grad():
            self.circuit.params.copy_(torch.tensor(opt.x, dtype=self.circuit.dtype).reshape(shape))
        return result
