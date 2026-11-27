"""Extension Benchmark: Baseline HEA vs. Controlled-Rotation (Lagrange GPSR) + 2nd-Order ODE.

Demonstrates two concrete upgrades enabled by `dqc.extensions`:
  1. A/B comparison on Fig. 6 (lambda = 20 damped oscillator):
     Baseline HEA (fixed CNOTs) vs. Controlled-Rotation HEA (trainable CRY entanglers,
     hardware-differentiable via the 4-term Lagrange parameter-shift rule).
  2. Solving a 2nd-order damped harmonic oscillator ODE using exact d^2f/dx^2.

Run with:
    python 2011.10395v2/experiments/ext_lagrange_comparison.py
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import torch

from dqc.boundary import BoundaryHandler
from dqc.circuit import DifferentiableQuantumCircuit
from dqc.extensions.lagrange_interpolation.cr_ansatz import ControlledRotationDQC
from dqc.losses import solution_quality
from dqc.solver import DQCSolver
from problems.damped_oscillator import DampedOscillatorProblem
from problems.second_order_ode import SecondOrderHarmonicProblem


def run_ab_damped_oscillator(n_iter: int = 250, lr: float = 0.05, seed: int = 42):
    """Compare baseline HEA vs CR-HEA on Eq. (30) with lambda = 20."""
    problem = DampedOscillatorProblem(lam=20.0, kappa=0.1, u0=1.0, x_min=0.0, x_max=0.9, n_points=20)
    x_train = problem.grid()
    x_plot = problem.grid(n_points=200)
    u_true_train = problem.true_solution(x_train)[0]
    u_true_plot = problem.true_solution(x_plot)[0]

    # 1. Baseline HEA (untouched dqc/ module)
    print("=== A/B Part 1: Baseline HEA (fixed CNOTs, N=6, d=5, lam=20) ===", flush=True)
    base_dqc = DifferentiableQuantumCircuit(n_qubits=6, depth=5, feature_map="chebyshev_tower", seed=seed)
    base_b = BoundaryHandler(mode="floating", x0=0.0, u0=problem.u0)
    base_res = DQCSolver(
        base_dqc, base_b, residual_fn=problem.residual, true_solution_fn=problem.true_solution
    ).solve(x_train, n_iter=n_iter, lr=lr)
    base_pred = base_res.predict(x_plot)[0]
    print(f"  Baseline HEA | L_F = {base_res.loss_full[-1]:.3e} | L_Q = {base_res.loss_quality[-1]:.3e}", flush=True)

    # 2. Extension CR-HEA (trainable CRY gates, 4-term Lagrange GPSR compatible)
    print("=== A/B Part 2: Extension CR-HEA (trainable CRY, N=6, d=5, lam=20) ===", flush=True)
    cr_dqc = ControlledRotationDQC(n_qubits=6, depth=5, feature_map="chebyshev_tower", seed=seed)
    optimizer = torch.optim.Adam(cr_dqc.trainable_params(), lr=lr)
    x0_t = torch.tensor(0.0, dtype=torch.float64)
    u0_t = torch.tensor(problem.u0, dtype=torch.float64)

    cr_lf: list[float] = []
    cr_lq: list[float] = []
    for _ in range(n_iter):
        optimizer.zero_grad()
        raw_f, df_dx = cr_dqc.eval_with_derivative(x_train)
        fb = u0_t - cr_dqc.raw_eval(x0_t)
        f = raw_f + fb
        res = problem.residual(x_train, [f], [df_dx])[0]
        loss = torch.mean(res**2)
        loss.backward()
        optimizer.step()
        cr_lf.append(float(loss.item()))
        with torch.no_grad():
            cr_lq.append(solution_quality(f.detach(), u_true_train))

    with torch.no_grad():
        cr_pred = cr_dqc.raw_eval(x_plot) + (u0_t - cr_dqc.raw_eval(x0_t))
    print(f"  CR-HEA (ext) | L_F = {cr_lf[-1]:.3e} | L_Q = {cr_lq[-1]:.3e}", flush=True)

    return x_plot, u_true_plot, base_res, base_pred, cr_lf, cr_lq, cr_pred


def run_second_order_ode(n_iter: int = 250, lr: float = 0.03, seed: int = 42):
    """Solve 2nd-order damped harmonic oscillator u'' + 2u' + 36u = 0, u(0)=1, u'(0)=0."""
    print("\n=== Extension Demo: 2nd-Order Damped Harmonic Oscillator (N=4, d=4) ===", flush=True)
    problem = SecondOrderHarmonicProblem(gamma=1.0, omega0=6.0, u0=1.0, du0=0.0, n_points=25)
    x_train = problem.grid()
    x_plot = problem.grid(n_points=200)
    u_true_train = problem.true_solution(x_train)
    u_true_plot = problem.true_solution(x_plot)

    dqc = ControlledRotationDQC(n_qubits=4, depth=4, feature_map="chebyshev_tower", seed=seed)
    optimizer = torch.optim.Adam(dqc.trainable_params(), lr=lr)
    x0_t = torch.tensor(0.0, dtype=torch.float64)
    u0_t = torch.tensor(problem.u0, dtype=torch.float64)
    du0_t = torch.tensor(problem.du0, dtype=torch.float64)

    lf_hist: list[float] = []
    lq_hist: list[float] = []
    for step in range(n_iter):
        optimizer.zero_grad()
        raw_f, df_dx, d2f_dx2 = dqc.eval_with_second_derivative(x_train)
        fb = u0_t - dqc.raw_eval(x0_t)
        f = raw_f + fb

        ode_res = problem.residual(x_train, f, df_dx, d2f_dx2)
        # Floating boundary enforces f(0) = u0 exactly; add slope boundary penalty for f'(0) = du0
        loss = torch.mean(ode_res**2) + 10.0 * (df_dx[0] - du0_t) ** 2
        loss.backward()
        optimizer.step()

        lf_hist.append(float(loss.item()))
        with torch.no_grad():
            lq_hist.append(solution_quality(f.detach(), u_true_train))
        if step % 50 == 0 or step == n_iter - 1:
            print(f"  iter {step + 1:4d}/{n_iter} | L_F = {lf_hist[-1]:.3e} | L_Q = {lq_hist[-1]:.3e}", flush=True)

    with torch.no_grad():
        pred_plot = dqc.raw_eval(x_plot) + (u0_t - dqc.raw_eval(x0_t))
    return x_plot, u_true_plot, pred_plot, lf_hist, lq_hist


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ext_lagrange_comparison.png"

    x1, u1_true, base_res, base_pred, cr_lf, cr_lq, cr_pred = run_ab_damped_oscillator()
    x2, u2_true, u2_pred, lf2, lq2 = run_second_order_ode()

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))

    # Top-left: A/B solution
    axes[0, 0].plot(x1.numpy(), u1_true.numpy(), color="black", linewidth=2.2, label="u(x) exact")
    axes[0, 0].plot(x1.numpy(), base_pred.numpy(), color="tab:blue", linestyle="--", linewidth=1.7, label="Baseline HEA (CNOT)")
    axes[0, 0].plot(x1.numpy(), cr_pred.numpy(), color="tab:orange", linestyle="-.", linewidth=1.7, label="Extension CR-HEA (CRY)")
    axes[0, 0].set_title("A/B: Fig. 6 ($\\lambda=20$) Baseline vs. CR-HEA")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("f(x)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    # Bottom-left: A/B loss & quality
    axes[1, 0].semilogy(base_res.loss_full, color="tab:blue", linestyle="--", label="$L_F$ Baseline")
    axes[1, 0].semilogy(base_res.loss_quality, color="tab:blue", linestyle="-", label="$L_Q$ Baseline")
    axes[1, 0].semilogy(cr_lf, color="tab:orange", linestyle="--", label="$L_F$ CR-HEA")
    axes[1, 0].semilogy(cr_lq, color="tab:orange", linestyle="-", label="$L_Q$ CR-HEA")
    axes[1, 0].set_title("A/B: Loss & Quality Comparison")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    # Top-right: 2nd-order ODE solution
    axes[0, 1].plot(x2.numpy(), u2_true.numpy(), color="black", linewidth=2.2, label="Exact $u(x)$")
    axes[0, 1].plot(x2.numpy(), u2_pred.numpy(), color="tab:green", linestyle="--", linewidth=1.8, label="CR-HEA DQC")
    axes[0, 1].set_title("Extension: 2nd-Order Harmonic Oscillator")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("f(x)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    # Bottom-right: 2nd-order ODE loss & quality
    axes[1, 1].semilogy(lf2, color="tab:green", linestyle="--", label="$L_F$ (2nd-order)")
    axes[1, 1].semilogy(lq2, color="tab:green", linestyle="-", label="$L_Q$ (Quality)")
    axes[1, 1].set_title("2nd-Order ODE Training Metrics")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved extension benchmark figure to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
