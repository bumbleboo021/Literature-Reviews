"""Reproduce Fig. 6(a, b, d, e) of arXiv:2011.10395v2.

Compares three quantum feature maps (Product, Chebyshev, Chebyshev Tower)
on the damped oscillator ODE (Eq. 30) for lambda = 8 and lambda = 20.

Run with:
    python 2011.10395v2/experiments/reproduce_fig6.py
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import torch

from dqc.boundary import BoundaryHandler
from dqc.circuit import DifferentiableQuantumCircuit
from dqc.solver import DQCSolver
from problems.damped_oscillator import DampedOscillatorProblem


FEATURE_MAP_SPECS = [
    ("product", "Prod", "tab:red", "--"),
    ("chebyshev", "Cheb", "tab:green", "-."),
    ("chebyshev_tower", "ChebT", "tab:blue", ":"),
]


def run_case(
    lam: float,
    n_qubits: int = 6,
    depth: int = 5,
    n_iter: int = 250,
    lr: float = 0.02,
    seed: int = 42,
):
    problem = DampedOscillatorProblem(lam=lam, kappa=0.1, u0=1.0, x_min=0.0, x_max=0.9, n_points=20)
    x_train = problem.grid()
    x_plot = problem.grid(n_points=200)
    u_true_plot = problem.true_solution(x_plot)[0]

    print(f"\n=== Running Fig. 6 case: lambda = {lam} (N={n_qubits}, d={depth}, n_iter={n_iter}) ===", flush=True)
    results = {}
    for fmap_key, label, _, _ in FEATURE_MAP_SPECS:
        dqc = DifferentiableQuantumCircuit(
            n_qubits=n_qubits, depth=depth, feature_map=fmap_key, seed=seed
        )
        boundary = BoundaryHandler(mode="floating", x0=0.0, u0=problem.u0)
        solver = DQCSolver(
            dqc,
            boundary,
            residual_fn=problem.residual,
            true_solution_fn=problem.true_solution,
        )
        res = solver.solve(x_train, n_iter=n_iter, lr=lr)
        pred_plot = res.predict(x_plot)[0]
        results[fmap_key] = (res, pred_plot)
        print(
            f"  {label:<6} | final L_F = {res.loss_full[-1]:.3e} | final L_Q = {res.loss_quality[-1]:.3e}",
            flush=True,
        )

    return x_plot, u_true_plot, results


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig6_feature_maps.png"

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    for col, lam in enumerate([8.0, 20.0]):
        x_plot, u_true_plot, results = run_case(lam=lam, n_qubits=6, depth=5, n_iter=250, lr=0.05)

        ax_sol = axes[0, col]
        ax_loss = axes[1, col]

        ax_sol.plot(x_plot.numpy(), u_true_plot.numpy(), color="black", linewidth=2.2, label="u(x) exact")
        for fmap_key, label, color, ls in FEATURE_MAP_SPECS:
            res, pred_plot = results[fmap_key]
            ax_sol.plot(x_plot.numpy(), pred_plot.numpy(), color=color, linestyle=ls, linewidth=1.8, label=label)
            ax_loss.semilogy(res.loss_full, color=color, linestyle="--", alpha=0.8, label=f"$L_F$ {label}")
            ax_loss.semilogy(res.loss_quality, color=color, linestyle="-", linewidth=1.6, label=f"$L_Q$ {label}")

        ax_sol.set_title(f"DQC Solution ($\\lambda = {int(lam)}, \\kappa = 0.1$)")
        ax_sol.set_xlabel("x")
        ax_sol.set_ylabel("f(x)")
        ax_sol.legend(fontsize=8)
        ax_sol.grid(alpha=0.3)

        ax_loss.set_title(f"Training Loss & Quality ($\\lambda = {int(lam)}$)")
        ax_loss.set_xlabel("Iteration $n_j$")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=7, ncol=2)
        ax_loss.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved figure to: {out_path}")


if __name__ == "__main__":
    main()
