"""Unit tests for Fubini-Study metric tensor (Eq. 27) and QNGSolver."""

import math
import numpy as np
import pennylane as qml
import torch

from dqc.boundary import BoundaryHandler
from dqc.circuit import DifferentiableQuantumCircuit
from dqc.extensions.lagrange_gpsr import derive_lagrange_shift_rule
from dqc.extensions.qng_solver import (
    QNGSolver,
    fubini_study_lagrange,
    fubini_study_statevector,
)


def test_fubini_study_matches_paper_fig2_eq27():
    """Verify Eq. (27) & Fig. 2 of Vu Tuan Hai & Le Bin Ho using 2nd-order Lagrange GPSR.

    Circuit: RX(theta_x) on wire 0, RZ(theta_z) on wire 1, CRY(theta_y) on [0, 1].
    Exact Fubini-Study metric tensor:
        g = diag(1/4, 0, (1/4) * sin^2(theta_x / 2))
    """
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def state_fn(theta):
        qml.RX(theta[0], wires=0)
        qml.RZ(theta[1], wires=1)
        qml.CRY(theta[2], wires=[0, 1])
        return qml.state()

    def overlap_prob(t_ref, t_var) -> float:
        psi_ref = state_fn(t_ref)
        psi_var = state_fn(t_var)
        return float(np.abs(np.vdot(psi_ref, psi_var)) ** 2)

    theta = [1.2, 0.7, 0.9]
    r_2term = derive_lagrange_shift_rule([-0.5, 0.5])       # RX, RZ (n=2)
    r_4term = derive_lagrange_shift_rule([-0.5, 0.0, 0.5])  # CRY    (n=3)

    g_lagrange = fubini_study_lagrange(overlap_prob, theta, [r_2term, r_2term, r_4term])
    expected = np.diag([0.25, 0.0, 0.25 * math.sin(theta[0] / 2.0) ** 2])
    assert np.allclose(g_lagrange, expected, atol=1e-10)


def test_fubini_study_statevector_matches_lagrange_on_dqc():
    """Verify fast statevector Fubini-Study metric matches 2nd-order Lagrange GPSR."""
    dqc = DifferentiableQuantumCircuit(n_qubits=2, depth=1, feature_map="chebyshev_tower", seed=5)
    x_pt = torch.tensor([0.35], dtype=torch.float64)

    g_sv = fubini_study_statevector(dqc, x_pt).numpy()
    assert g_sv.shape == (6, 6)
    # Symmetric and positive semi-definite
    assert np.allclose(g_sv, g_sv.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(g_sv)
    assert eigvals.min() >= -1e-10


def test_qng_solver_converges_on_exponential_decay():
    """Smoke test: QNGSolver solves du/dx + 2u = 0, u(0) = 1 in 40 steps."""
    dqc = DifferentiableQuantumCircuit(n_qubits=3, depth=2, feature_map="chebyshev_tower", seed=42)
    boundary = BoundaryHandler(mode="floating", x0=0.0, u0=1.0)

    def residual_fn(x, fs, dfs):
        return [dfs[0] + 2.0 * fs[0]]

    def true_fn(x):
        return [torch.exp(-2.0 * x)]

    x_grid = torch.linspace(0.0, 0.8, 12, dtype=torch.float64)
    res = QNGSolver(dqc, boundary, residual_fn, true_fn).solve(
        x_grid,
        n_iter=60,
        lr=0.05,
        diag_reg=1e-2,
        metric_every=5,
        metric_subsample=3,
        max_step_norm=0.3,
    )
    assert res.loss_full[-1] < res.loss_full[0] * 1e-2
    assert res.loss_quality[-1] < 1e-3
