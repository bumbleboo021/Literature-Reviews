"""Unit tests for ControlledRotationDQC (CR-HEA) and 2nd-order ODE problem."""

import torch

from dqc.extensions.cr_ansatz import ControlledRotationDQC
from problems.second_order_ode import SecondOrderHarmonicProblem


def test_cr_dqc_first_and_second_derivatives_match_fd():
    dqc = ControlledRotationDQC(n_qubits=3, depth=2, feature_map="chebyshev_tower", seed=42)
    xs = torch.tensor([0.1, 0.4, 0.7], dtype=torch.float64)

    f, df, d2f = dqc.eval_with_second_derivative(xs)

    eps = 1e-4
    with torch.no_grad():
        f_p = dqc.raw_eval(xs + eps)
        f_m = dqc.raw_eval(xs - eps)
        df_fd = (f_p - f_m) / (2.0 * eps)
        d2f_fd = (f_p - 2.0 * f + f_m) / (eps**2)

    assert torch.allclose(df, df_fd, atol=1e-5)
    assert torch.allclose(d2f, d2f_fd, atol=1e-4)

    # Verify gradients flow to BOTH single-qubit and CRY entangling parameters
    loss = torch.mean(d2f**2 + f**2)
    loss.backward()
    assert dqc.rot_params.grad is not None and torch.isfinite(dqc.rot_params.grad).all()
    assert dqc.ent_params.grad is not None and torch.isfinite(dqc.ent_params.grad).all()


def test_second_order_harmonic_exact_solution_zero_residual():
    prob = SecondOrderHarmonicProblem(gamma=1.0, omega0=6.0, u0=1.0, du0=0.0)
    x = prob.grid().requires_grad_(True)
    u = prob.true_solution(x)
    du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    d2u = torch.autograd.grad(du.sum(), x)[0]
    res = prob.residual(x, u, du, d2u)
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-11)
    assert abs(u[0].item() - 1.0) < 1e-12
    assert abs(du[0].item() - 0.0) < 1e-12
