"""Reproduce Fig. 8 of arXiv:2011.10395v2 (Strongly Coupled ODEs, Eqs. 33–34).

Compares all three boundary handling strategies:
  - Pinned    (Eq. 24)
  - Floating  (Eqs. 25–26)
  - Optimized (Eq. 27)

Run with:
    python 2011.10395v2/experiments/reproduce_fig8.py
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from dqc.boundary import BoundaryHandler, BoundaryMode
from dqc.circuit import DifferentiableQuantumCircuit
from dqc.solver import DQCSolver
from problems.coupled_odes import CoupledODEProblem


BOUNDARY_SPECS: list[tuple[BoundaryMode, str, str]] = [
    ("pinned", "Pin", "tab:red"),
    ("floating", "Float", "tab:purple"),
    ("optimized", "Optim", "tab:cyan"),
]


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig8_coupled_odes.png"

    problem = CoupledODEProblem(lam1=5.0, lam2=3.0, u1_0=0.5, u2_0=0.0, x_min=0.0, x_max=0.9, n_points=20)
    x_train = problem.grid()
    x_plot = problem.grid(n_points=200)
    u1_true, u2_true = problem.true_solution(x_plot)

    print("=== Running Fig. 8: Coupled ODEs (N=6, d=5, ChebT, lr=0.02, 250 iters) ===", flush=True)
    results = {}
    for mode, label, _ in BOUNDARY_SPECS:
        c1 = DifferentiableQuantumCircuit(n_qubits=6, depth=5, feature_map="chebyshev_tower", seed=42)
        c2 = DifferentiableQuantumCircuit(n_qubits=6, depth=5, feature_map="chebyshev_tower", seed=43)
        b1 = BoundaryHandler(mode=mode, x0=0.0, u0=problem.u1_0, eta=1.0)
        b2 = BoundaryHandler(mode=mode, x0=0.0, u0=problem.u2_0, eta=1.0)

        solver = DQCSolver(
            circuits=[c1, c2],
            boundaries=[b1, b2],
            residual_fn=problem.residual,
            true_solution_fn=problem.true_solution,
        )
        res = solver.solve(x_train, n_iter=250, lr=0.02)
        f1_plot, f2_plot = res.predict(x_plot)
        results[mode] = (res, f1_plot, f2_plot)
        print(
            f"  {label:<6} | final L_F = {res.loss_full[-1]:.3e} | final L_Q = {res.loss_quality[-1]:.3e}",
            flush=True,
        )

    # Plot Fig. 8(a, b)
    fig, (ax_sol, ax_loss) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_sol.plot(x_plot.numpy(), u1_true.numpy(), color="black", linewidth=2.2, label="$u_1$ exact")
    ax_sol.plot(x_plot.numpy(), u2_true.numpy(), color="dimgray", linewidth=2.2, label="$u_2$ exact")
    for mode, label, color in BOUNDARY_SPECS:
        res, f1_plot, f2_plot = results[mode]
        ax_sol.plot(x_plot.numpy(), f1_plot.numpy(), color=color, linestyle="--", linewidth=1.6, label=f"$f_1$ {label}")
        ax_sol.plot(x_plot.numpy(), f2_plot.numpy(), color=color, linestyle=":", linewidth=1.8, label=f"$f_2$ {label}")

        ax_loss.semilogy(res.loss_full, color=color, linestyle="--", alpha=0.85, label=f"$L_F$ {label}")
        ax_loss.semilogy(res.loss_quality, color=color, linestyle="-", linewidth=1.6, label=f"$L_Q$ {label}")

    ax_sol.set_title("Fig. 8(a): Strongly Coupled ODEs ($\\lambda_1=5, \\lambda_2=3$)")
    ax_sol.set_xlabel("x")
    ax_sol.set_ylabel("f(x)")
    ax_sol.legend(fontsize=7, ncol=2)
    ax_sol.grid(alpha=0.3)

    ax_loss.set_title("Fig. 8(b): Boundary Handling Comparison")
    ax_loss.set_xlabel("Iteration $n_j$")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved figure to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
