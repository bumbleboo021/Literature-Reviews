"""Extension Benchmark: Classical Chebyshev-PINN vs. Quantum DQC on Fig. 6 (lambda = 8 & 20).

Clean apples-to-apples comparison between two Chebyshev spectral solvers:
  1. Quantum DQC (N=6, d=5, Chebyshev Tower, 90 params, 250 Adam steps)
  2. Classical Cheb-PINN ([T_1..T_12] -> 32 -> 32 -> 1, 1,505 params, 500 Adam -> 200 L-BFGS-B)

Run with:
    python 2011.10395v2/experiments/ext_pinn_vs_dqc.py
"""

from __future__ import annotations

from pathlib import Path
import time
import matplotlib.pyplot as plt

from dqc.boundary import BoundaryHandler
from dqc.circuit import DifferentiableQuantumCircuit
from dqc.extensions.pinn_baseline import ChebyshevPINN, train_pinn_adam, train_pinn_lbfgs
from dqc.solver import DQCSolver
from problems.damped_oscillator import DampedOscillatorProblem


def run_comparison(lam: float, seed: int = 42):
    problem = DampedOscillatorProblem(lam=lam, kappa=0.1, u0=1.0, x_min=0.0, x_max=0.9, n_points=20)
    x_train_dqc = problem.grid()
    x_train_pinn = problem.grid(n_points=60)
    x_plot = problem.grid(n_points=200)
    u_true_plot = problem.true_solution(x_plot)[0]

    print(f"\n=== Cheb-PINN vs. Quantum DQC (lambda = {lam}) ===", flush=True)

    # 1. Quantum DQC (90 params, 250 Adam steps)
    t0 = time.perf_counter()
    dqc = DifferentiableQuantumCircuit(n_qubits=6, depth=5, feature_map="chebyshev_tower", seed=seed)
    b = BoundaryHandler(mode="floating", x0=0.0, u0=problem.u0)
    res_dqc = DQCSolver(dqc, b, problem.residual, problem.true_solution).solve(x_train_dqc, n_iter=250, lr=0.05)
    t_dqc = time.perf_counter() - t0
    pred_dqc = res_dqc.predict(x_plot)[0]
    print(
        f"  Quantum DQC (90 params)   | time = {t_dqc:6.2f}s | L_F = {res_dqc.loss_full[-1]:.3e} | L_Q = {res_dqc.loss_quality[-1]:.3e}",
        flush=True,
    )

    # 2. Classical Chebyshev-PINN (1,505 params, 500 Adam -> 200 L-BFGS-B)
    t0 = time.perf_counter()
    cpinn = ChebyshevPINN(degree=12, hidden_dims=(32, 32), seed=seed)
    res_adam = train_pinn_adam(
        cpinn, x_train_pinn, problem.residual, x0=0.0, u0=problem.u0,
        true_solution_fn=problem.true_solution, n_iter=500, lr=1e-2,
    )
    res_lbfgs = train_pinn_lbfgs(
        cpinn, x_train_pinn, problem.residual, x0=0.0, u0=problem.u0,
        true_solution_fn=problem.true_solution, n_iter=200,
    )
    t_pinn = time.perf_counter() - t0
    pred_pinn = res_lbfgs.predict(x_plot)
    lq_pinn = res_adam.loss_quality + res_lbfgs.loss_quality
    lf_pinn = res_adam.loss_full + res_lbfgs.loss_full
    print(
        f"  Cheb-PINN ({cpinn.param_count():4d} params)  | time = {t_pinn:6.2f}s | L_F = {lf_pinn[-1]:.3e} | L_Q = {lq_pinn[-1]:.3e}",
        flush=True,
    )

    return x_plot, u_true_plot, (res_dqc.loss_quality, pred_dqc, 90), (lq_pinn, pred_pinn, cpinn.param_count())


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ext_pinn_vs_dqc.png"

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))

    for col, lam in enumerate([8.0, 20.0]):
        x_plot, u_true, (lq_dqc, p_dqc, n_dqc), (lq_pinn, p_pinn, n_pinn) = run_comparison(lam=lam)

        ax_sol = axes[0, col]
        ax_loss = axes[1, col]

        ax_sol.plot(x_plot.numpy(), u_true.numpy(), color="black", linewidth=2.5, label="u(x) exact")
        ax_sol.plot(x_plot.numpy(), p_dqc.numpy(), color="tab:blue", linestyle="--", linewidth=1.9, label=f"Quantum DQC ({n_dqc}p)")
        ax_sol.plot(x_plot.numpy(), p_pinn.numpy(), color="tab:orange", linestyle="-.", linewidth=1.9, label=f"Cheb-PINN ({n_pinn}p)")
        ax_sol.set_title(f"Cheb-PINN vs. DQC ($\\lambda = {int(lam)}, \\kappa = 0.1$)")
        ax_sol.set_xlabel("x")
        ax_sol.set_ylabel("f(x)")
        ax_sol.legend(fontsize=8.5)
        ax_sol.grid(alpha=0.3)

        ax_loss.semilogy(lq_dqc, color="tab:blue", linewidth=2.0, label=f"$L_Q$ Quantum DQC ({n_dqc}p, 250 Adam)")
        ax_loss.semilogy(lq_pinn, color="tab:orange", linewidth=2.0, label=f"$L_Q$ Cheb-PINN ({n_pinn}p, Adam+LBFGS)")
        ax_loss.axvline(500, color="gray", linestyle=":", alpha=0.8, label="L-BFGS switch (iter 500)")
        ax_loss.set_title(f"Solution Quality $L_Q$ ($\\lambda = {int(lam)}$)")
        ax_loss.set_xlabel("Optimization Step")
        ax_loss.set_ylabel("$L_Q$ (MSE vs. Exact)")
        ax_loss.legend(fontsize=8)
        ax_loss.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved figure to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
