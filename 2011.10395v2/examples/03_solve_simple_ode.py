"""Example 03: End-to-End Solution of a Simple ODE (du/dx + 2u = 0, u(0) = 1).

Run with:
    python 2011.10395v2/examples/03_solve_simple_ode.py
"""

import torch

from dqc.boundary import BoundaryHandler
from dqc.circuit import DifferentiableQuantumCircuit
from dqc.solver import DQCSolver


def main() -> None:
    # 1. Build a small 3-qubit, depth-2 DQC with Chebyshev Tower feature map
    dqc = DifferentiableQuantumCircuit(
        n_qubits=3,
        depth=2,
        feature_map="chebyshev_tower",
        seed=42,
    )

    # 2. Floating boundary guarantees f(0) = 1.0 exactly at every step (Eq. 25-26)
    boundary = BoundaryHandler(mode="floating", x0=0.0, u0=1.0)

    # 3. Define the ODE residual F[du/dx, u, x] = du/dx + 2u = 0
    def residual_fn(x, fs, dfs):
        u, du_dx = fs[0], dfs[0]
        return [du_dx + 2.0 * u]

    def true_solution_fn(x):
        return [torch.exp(-2.0 * x)]

    # 4. Train with Adam on 15 equidistant points in [0, 0.8]
    x_grid = torch.linspace(0.0, 0.8, 15, dtype=torch.float64)
    solver = DQCSolver(dqc, boundary, residual_fn=residual_fn, true_solution_fn=true_solution_fn)

    print("=== Training DQC on du/dx + 2u = 0, u(0) = 1 ===")
    result = solver.solve(x_grid, n_iter=100, lr=0.05, log_every=20)

    # 5. Evaluate on a finer test grid (including points unseen during training!)
    x_test = torch.linspace(0.0, 0.8, 9, dtype=torch.float64)
    f_pred = result.predict(x_test)[0]
    u_true = true_solution_fn(x_test)[0]

    print("\n=== Predicted vs. Exact Solution u(x) = exp(-2x) ===")
    print(f"{'x':>6} | {'DQC f(x)':>10} | {'Exact u(x)':>10} | {'|error|':>10}")
    print("-" * 46)
    for xi, fi, ui in zip(x_test, f_pred, u_true):
        print(f"{xi.item():6.2f} | {fi.item():10.6f} | {ui.item():10.6f} | {abs(fi - ui).item():10.2e}")


if __name__ == "__main__":
    main()
