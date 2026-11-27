"""Unit tests for the classical ChebyshevPINN baseline."""

import torch

from dqc.extensions.pinn_baseline import ChebyshevPINN, train_pinn_adam


def test_chebyshev_pinn_derivative_matches_fd():
    cpinn = ChebyshevPINN(degree=6, hidden_dims=(16,), seed=42)
    xs = torch.tensor([0.0, 0.3, 0.6, 0.85], dtype=torch.float64)
    f, df = cpinn.eval_with_derivative(xs)

    eps = 1e-6
    with torch.no_grad():
        df_fd = (cpinn.raw_eval(xs + eps) - cpinn.raw_eval(xs - eps)) / (2.0 * eps)
    assert torch.allclose(df, df_fd, atol=1e-6)


def test_chebyshev_pinn_solves_exponential_decay_with_exact_boundary():
    cpinn = ChebyshevPINN(degree=6, hidden_dims=(16, 16), seed=42)
    x_grid = torch.linspace(0.0, 0.8, 15, dtype=torch.float64)

    def res_fn(x, fs, dfs):
        return [dfs[0] + 2.0 * fs[0]]

    def true_fn(x):
        return [torch.exp(-2.0 * x)]

    res = train_pinn_adam(
        cpinn, x_grid, res_fn, x0=0.0, u0=1.0, true_solution_fn=true_fn, n_iter=120, lr=0.02
    )
    pred = res.predict(x_grid)
    assert abs(pred[0].item() - 1.0) < 1e-12
    assert res.loss_quality[-1] < 1e-4
