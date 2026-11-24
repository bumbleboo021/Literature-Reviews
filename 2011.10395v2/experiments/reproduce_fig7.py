"""Reproduce Fig. 7 of arXiv:2011.10395v2 (Highly Nontrivial Nonlinear ODE, Eq. 32).

Stage 1: Train 6-qubit, depth-5 DQC with Chebyshev Tower feature map for 200 iterations.
Stage 2: Transfer Stage-1 angles into an Evolution-Enhanced Chebyshev Tower DQC
         (tau = 2.0, nearest-neighbor Ising Hamiltonian) and train for 200 more iterations.

Run with:
    python 2011.10395v2/experiments/reproduce_fig7.py
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import torch

from dqc.boundary import BoundaryHandler
from dqc.circuit import DifferentiableQuantumCircuit
from dqc.feature_maps import build_ising_evolution_unitary
from dqc.solver import DQCSolver
from problems.nontrivial_ode import NontrivialODEProblem


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig7_nontrivial_ode.png"

    problem = NontrivialODEProblem(u0=0.75, x_min=0.0, x_max=0.9, n_points=100)
    x_train = problem.grid()
    x_plot = problem.grid(n_points=300)
    u_true_plot = problem.true_solution(x_plot)[0]

    # Stage 1: Chebyshev Tower feature map (200 iterations, lr = 0.01)
    print("=== Fig. 7 Stage 1: Chebyshev Tower (N=6, d=5, 100 pts, 200 iters) ===", flush=True)
    dqc_chebt = DifferentiableQuantumCircuit(
        n_qubits=6, depth=5, feature_map="chebyshev_tower", seed=42
    )
    b1 = BoundaryHandler(mode="floating", x0=0.0, u0=problem.u0)
    res_chebt = DQCSolver(
        dqc_chebt, b1, residual_fn=problem.residual, true_solution_fn=problem.true_solution
    ).solve(x_train, n_iter=200, lr=0.01, log_every=50)
    pred_chebt = res_chebt.predict(x_plot)[0]
    print(
        f"  ChebT final | L_F = {res_chebt.loss_full[-1]:.3e} | L_Q = {res_chebt.loss_quality[-1]:.3e}",
        flush=True,
    )

    # Stage 2: Evolution-Enhanced Chebyshev Tower initialized from Stage-1 angles
    print("\n=== Fig. 7 Stage 2: Evolution-Enhanced Chebyshev Tower (tau=2.0, 200 iters) ===", flush=True)
    u_evo = build_ising_evolution_unitary(n_qubits=6, tau=2.0, seed=42)
    dqc_even = DifferentiableQuantumCircuit(
        n_qubits=6, depth=5, feature_map="chebyshev_tower", evolution_unitary=u_evo, seed=42
    )
    dqc_even.params = dqc_chebt.params.detach().clone().requires_grad_(True)
    b2 = BoundaryHandler(mode="floating", x0=0.0, u0=problem.u0)
    res_even = DQCSolver(
        dqc_even, b2, residual_fn=problem.residual, true_solution_fn=problem.true_solution
    ).solve(x_train, n_iter=200, lr=0.01, log_every=50)
    pred_even = res_even.predict(x_plot)[0]
    print(
        f"  EvEn  final | L_F = {res_even.loss_full[-1]:.3e} | L_Q = {res_even.loss_quality[-1]:.3e}",
        flush=True,
    )

    # Plot Fig. 7(a, b)
    fig, (ax_sol, ax_loss) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_sol.plot(x_plot.numpy(), u_true_plot.numpy(), color="black", linewidth=2.0, label="True Solution")
    ax_sol.plot(x_plot.numpy(), pred_chebt.numpy(), color="tab:red", linestyle="--", linewidth=1.6, label="f(x) ChebT")
    ax_sol.plot(x_plot.numpy(), pred_even.numpy(), color="tab:blue", linestyle="-.", linewidth=1.6, label="f(x) EvEn")
    ax_sol.set_title("Fig. 7(a): Nontrivial Nonlinear ODE (Eq. 32)")
    ax_sol.set_xlabel("x")
    ax_sol.set_ylabel("f(x)")
    ax_sol.legend(fontsize=8)
    ax_sol.grid(alpha=0.3)

    ax_loss.semilogy(res_chebt.loss_full, color="tab:red", linestyle="--", label="$L_F$ ChebT")
    ax_loss.semilogy(res_chebt.loss_quality, color="tab:red", linestyle="-", label="$L_Q$ ChebT")
    ax_loss.semilogy(res_even.loss_full, color="tab:blue", linestyle="--", label="$L_F$ EvEn")
    ax_loss.semilogy(res_even.loss_quality, color="tab:blue", linestyle="-", label="$L_Q$ EvEn")
    ax_loss.set_title("Fig. 7(b): Loss & Solution Quality")
    ax_loss.set_xlabel("Iteration $n_j$")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved figure to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
