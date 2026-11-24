"""Unit tests verifying exact analytical derivatives df/dx and dL/dtheta."""

import pytest
import torch

from dqc.circuit import DifferentiableQuantumCircuit


@pytest.mark.parametrize("feature_map", ["product", "chebyshev", "chebyshev_tower"])
def test_autograd_matches_finite_difference_and_parameter_shift(feature_map):
    """Verify three independent ways of computing df/dx agree to ~1e-7:
      1. PyTorch autograd through PennyLane backprop (`eval_with_derivative`)
      2. Explicit hardware parameter-shift rule (`df_dx_parameter_shift`, Eq. 2)
      3. Central finite differences (f(x+eps) - f(x-eps)) / (2*eps)
    """
    dqc = DifferentiableQuantumCircuit(
        n_qubits=3, depth=2, feature_map=feature_map, seed=123
    )
    xs = torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.float64)

    f_val, df_autograd = dqc.eval_with_derivative(xs)
    df_pshift = dqc.df_dx_parameter_shift(xs)

    eps = 1e-6
    with torch.no_grad():
        df_fd = (dqc.raw_eval(xs + eps) - dqc.raw_eval(xs - eps)) / (2.0 * eps)

    assert f_val.shape == xs.shape
    assert df_autograd.shape == xs.shape
    assert torch.allclose(df_autograd, df_pshift, atol=1e-7)
    assert torch.allclose(df_autograd, df_fd, atol=1e-6)


def test_second_order_gradient_wrt_theta_flows_and_matches_fd():
    """Training an ODE loss requires d/dtheta of a function depending on df/dx.
    Verify `create_graph=True` gives exact gradients wrt variational params theta.
    """
    dqc = DifferentiableQuantumCircuit(
        n_qubits=2, depth=1, feature_map="chebyshev_tower", seed=7
    )
    xs = torch.tensor([0.1, 0.4, 0.7], dtype=torch.float64)

    f_val, df_dx = dqc.eval_with_derivative(xs)
    loss = torch.mean((df_dx + f_val) ** 2)
    loss.backward()

    grad_auto = dqc.params.grad.detach().clone()
    assert grad_auto.shape == dqc.params.shape
    assert torch.isfinite(grad_auto).all()

    # Cross-check one parameter entry with finite differences
    eps = 1e-6
    idx = (0, 0, 1)
    p_plus = dqc.params.detach().clone()
    p_minus = dqc.params.detach().clone()
    p_plus[idx] += eps
    p_minus[idx] -= eps

    f_p, df_p = dqc.eval_with_derivative(xs, params=p_plus)
    f_m, df_m = dqc.eval_with_derivative(xs, params=p_minus)
    loss_p = torch.mean((df_p + f_p) ** 2)
    loss_m = torch.mean((df_m + f_m) ** 2)
    fd_entry = (loss_p - loss_m) / (2.0 * eps)

    assert torch.allclose(grad_auto[idx], fd_entry, atol=1e-5)
