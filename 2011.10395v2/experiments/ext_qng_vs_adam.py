"""Extension Benchmark: 6-Way Optimizer Comparison on Fig. 6 (lambda = 8).

Compares six optimization strategies from the EXACT same initial circuit parameters
(N=4, d=4, Chebyshev Tower, 120 total steps):
  1. Pure Adam      (120 steps)
  2. Pure QNG       (120 steps, Fubini-Study metric + backtracking)
  3. Pure L-BFGS    (120 steps, SciPy L-BFGS-B)
  4. QNG  -> Adam   (40 QNG warm-start  + 80 Adam)
  5. Adam -> L-BFGS (40 Adam warm-start + 80 L-BFGS-B)
  6. QNG  -> L-BFGS (40 QNG warm-start  + 80 L-BFGS-B)

Run with:
    python 2011.10395v2/experiments/ext_qng_vs_adam.py
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import torch

from dqc.boundary import BoundaryHandler
from dqc.circuit import DifferentiableQuantumCircuit
from dqc.extensions.lagrange_interpolation.qng_solver import LBFGSSolver, QNGSolver
from dqc.solver import DQCSolver
from problems.damped_oscillator import DampedOscillatorProblem


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ext_qng_vs_adam.png"

    problem = DampedOscillatorProblem(lam=8.0, kappa=0.1, u0=1.0, x_min=0.0, x_max=0.9, n_points=20)
    x_train = problem.grid()
    x_plot = problem.grid(n_points=200)
    u_true_plot = problem.true_solution(x_plot)[0]

    n_qubits, depth, seed = 4, 4, 42
    n_warm, n_refine = 40, 80
    n_total = n_warm + n_refine

    def make_setup():
        dqc = DifferentiableQuantumCircuit(
            n_qubits=n_qubits, depth=depth, feature_map="chebyshev_tower", seed=seed
        )
        b = BoundaryHandler(mode="floating", x0=0.0, u0=problem.u0)
        return dqc, b

    qng_kwargs = dict(lr=0.05, diag_reg=2e-2, metric_every=4, metric_subsample=3, max_step_norm=0.3)

    # 1. Pure Adam (120 steps)
    print("=== 1/6: Pure Adam (120 steps) ===", flush=True)
    c1, b1 = make_setup()
    r1 = DQCSolver(c1, b1, problem.residual, problem.true_solution).solve(x_train, n_iter=n_total, lr=0.03)
    p1, lq1, lf1 = r1.predict(x_plot)[0], r1.loss_quality, r1.loss_full
    print(f"  Pure Adam     | L_F = {lf1[-1]:.3e} | L_Q = {lq1[-1]:.3e}", flush=True)

    # 2. Pure QNG (120 steps)
    print("=== 2/6: Pure QNG (120 steps) ===", flush=True)
    c2, b2 = make_setup()
    r2 = QNGSolver(c2, b2, problem.residual, problem.true_solution).solve(x_train, n_iter=n_total, **qng_kwargs)
    p2, lq2, lf2 = r2.predict(x_plot)[0], r2.loss_quality, r2.loss_full
    print(f"  Pure QNG      | L_F = {lf2[-1]:.3e} | L_Q = {lq2[-1]:.3e}", flush=True)

    # 3. Pure L-BFGS (120 steps)
    print("=== 3/6: Pure L-BFGS (120 steps) ===", flush=True)
    c3, b3 = make_setup()
    r3 = LBFGSSolver(c3, b3, problem.residual, problem.true_solution).solve(x_train, n_iter=n_total)
    p3, lq3, lf3 = r3.predict(x_plot)[0], r3.loss_quality, r3.loss_full
    print(f"  Pure L-BFGS   | L_F = {lf3[-1]:.3e} | L_Q = {lq3[-1]:.3e}", flush=True)

    # 4. QNG (40) -> Adam (80)
    print("=== 4/6: QNG (40) -> Adam (80) ===", flush=True)
    c4, b4 = make_setup()
    r4_a = QNGSolver(c4, b4, problem.residual, problem.true_solution).solve(x_train, n_iter=n_warm, **qng_kwargs)
    r4_b = DQCSolver(c4, b4, problem.residual, problem.true_solution).solve(x_train, n_iter=n_refine, lr=0.03)
    lf4, lq4 = r4_a.loss_full + r4_b.loss_full, r4_a.loss_quality + r4_b.loss_quality
    p4 = r4_b.predict(x_plot)[0]
    print(f"  QNG  -> Adam  | L_F = {lf4[-1]:.3e} | L_Q = {lq4[-1]:.3e}", flush=True)

    # 5. Adam (40) -> L-BFGS (80)
    print("=== 5/6: Adam (40) -> L-BFGS (80) ===", flush=True)
    c5, b5 = make_setup()
    r5_a = DQCSolver(c5, b5, problem.residual, problem.true_solution).solve(x_train, n_iter=n_warm, lr=0.03)
    r5_b = LBFGSSolver(c5, b5, problem.residual, problem.true_solution).solve(x_train, n_iter=n_refine)
    lf5, lq5 = r5_a.loss_full + r5_b.loss_full, r5_a.loss_quality + r5_b.loss_quality
    p5 = r5_b.predict(x_plot)[0]
    print(f"  Adam -> LBFGS | L_F = {lf5[-1]:.3e} | L_Q = {lq5[-1]:.3e}", flush=True)

    # 6. QNG (40) -> L-BFGS (80)
    print("=== 6/6: QNG (40) -> L-BFGS (80) ===", flush=True)
    c6, b6 = make_setup()
    r6_a = QNGSolver(c6, b6, problem.residual, problem.true_solution).solve(x_train, n_iter=n_warm, **qng_kwargs)
    r6_b = LBFGSSolver(c6, b6, problem.residual, problem.true_solution).solve(x_train, n_iter=n_refine)
    lf6, lq6 = r6_a.loss_full + r6_b.loss_full, r6_a.loss_quality + r6_b.loss_quality
    p6 = r6_b.predict(x_plot)[0]
    print(f"  QNG  -> LBFGS | L_F = {lf6[-1]:.3e} | L_Q = {lq6[-1]:.3e}", flush=True)

    # Plot 1x2 comparison
    fig, (ax_sol, ax_loss) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    curves = [
        ("Pure Adam", p1, lq1, "tab:blue", "--"),
        ("Pure QNG", p2, lq2, "tab:red", ":"),
        ("Pure L-BFGS", p3, lq3, "tab:orange", "-."),
        ("QNG -> Adam", p4, lq4, "tab:brown", "--"),
        ("Adam -> L-BFGS", p5, lq5, "tab:green", "-"),
        ("QNG -> L-BFGS", p6, lq6, "tab:purple", "-"),
    ]

    ax_sol.plot(x_plot.numpy(), u_true_plot.numpy(), color="black", linewidth=2.5, label="u(x) exact")
    for label, pred, lq, color, ls in curves:
        ax_sol.plot(x_plot.numpy(), pred.numpy(), color=color, linestyle=ls, linewidth=1.6, label=label)
        ax_loss.semilogy(lq, color=color, linestyle=ls, linewidth=1.8, label=label)

    ax_sol.set_title("Damped Oscillator ($\\lambda = 8$): 6-Way Optimizer Comparison")
    ax_sol.set_xlabel("x")
    ax_sol.set_ylabel("f(x)")
    ax_sol.legend(fontsize=8)
    ax_sol.grid(alpha=0.3)

    ax_loss.axvline(n_warm, color="gray", linestyle=":", alpha=0.8, label=f"switch (iter {n_warm})")
    ax_loss.set_title("Solution Quality $L_Q$ (MSE vs. Exact)")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("$L_Q$")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved figure to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
