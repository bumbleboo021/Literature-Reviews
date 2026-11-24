"""Unit tests for boundary handlers, losses, and a fast end-to-end ODE smoke test."""

import torch

from dqc.boundary import BoundaryHandler
from dqc.circuit import DifferentiableQuantumCircuit
from dqc.losses import regularization_schedule
from dqc.solver import DQCSolver


def test_floating_boundary_matches_initial_condition_exactly():
    """Eq. (25)-(26): floating boundary must satisfy f(x0) = u0 exactly for any random theta."""
    dqc = DifferentiableQuantumCircuit(n_qubits=3, depth=2, seed=99)
    boundary = BoundaryHandler(mode="floating", x0=0.0, u0=0.75)

    xs = torch.tensor([0.0, 0.3, 0.6], dtype=torch.float64)
    f, df_dx, b_loss = boundary.apply(dqc, xs)

    assert torch.allclose(f[0], torch.tensor(0.75, dtype=torch.float64), atol=1e-12)
    assert b_loss.item() == 0.0
    assert df_dx.shape == xs.shape


def test_optimized_boundary_exposes_trainable_shift():
    boundary = BoundaryHandler(mode="optimized", x0=0.0, u0=1.0, eta=2.0)
    extra = boundary.extra_params()
    assert len(extra) == 1
    assert extra[0].requires_grad


def test_regularization_schedule_decreases():
    w_start = regularization_schedule(step=0, n_iter=600, kind="reverse_sigmoid", n_drop=150, delta=0.1)
    w_mid = regularization_schedule(step=150, n_iter=600, kind="reverse_sigmoid", n_drop=150, delta=0.1)
    w_end = regularization_schedule(step=500, n_iter=600, kind="reverse_sigmoid", n_drop=150, delta=0.1)
    assert w_start > w_mid > w_end
    assert w_end < 1e-3


def test_solver_smoke_test_exponential_decay():
    """Solve du/dx + 2u = 0, u(0) = 1 (exact: u(x) = exp(-2x)) in 80 Adam steps."""
    dqc = DifferentiableQuantumCircuit(
        n_qubits=3, depth=2, feature_map="chebyshev_tower", seed=42
    )
    boundary = BoundaryHandler(mode="floating", x0=0.0, u0=1.0)

    def residual_fn(x, fs, dfs):
        u, du_dx = fs[0], dfs[0]
        return [du_dx + 2.0 * u]

    def true_fn(x):
        return [torch.exp(-2.0 * x)]

    x_grid = torch.linspace(0.0, 0.8, 12, dtype=torch.float64)
    solver = DQCSolver(dqc, boundary, residual_fn=residual_fn, true_solution_fn=true_fn)
    res = solver.solve(x_grid, n_iter=80, lr=0.05)

    assert res.loss_full[-1] < res.loss_full[0] * 0.01
    assert res.loss_quality[-1] < 1e-3
