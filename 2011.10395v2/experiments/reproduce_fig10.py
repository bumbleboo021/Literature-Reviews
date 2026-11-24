"""Reproduce Fig. 10 of arXiv:2011.10395v2 (Quasi-1D Navier-Stokes Nozzle, Eqs. 40–50).

Two-stage training workflow (pp. 16–17):
  - 3 parallel 6-qubit circuits for (rho(x), T(x), V(x)), Chebyshev feature map,
    HEA depth d = 6, floating boundary handling at x = 0.
  - Stage 1: 20 points on [0.0, 0.4], no regularization, Adam lr = 0.01, 200 iters.
  - Stage 2: 40 points = 20 on [0.0, 0.4] + 20 on [0.6, 0.9] (excluding throat (0.4, 0.6)),
             initialized from Stage-1 angles, Adam lr = 0.005, 600 iters, with weak
             reverse-sigmoid regularization dropping smoothly around n_drop = 150.

Run with:
    python 2011.10395v2/experiments/reproduce_fig10.py
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import torch

from dqc.boundary import BoundaryHandler
from dqc.circuit import DifferentiableQuantumCircuit
from dqc.losses import regularization_loss, regularization_schedule
from dqc.solver import DQCSolver
from problems.navier_stokes import NavierStokesNozzleProblem


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig10_navier_stokes.png"

    problem = NavierStokesNozzleProblem()
    x_plot = torch.linspace(0.0, 0.9, 200, dtype=torch.float64)
    rho_true, T_true, V_true = problem.true_solution(x_plot)

    # Exact inlet boundary values at x = 0 from the isentropic reservoir solution
    # (rho(0) ~ 0.995, T(0) ~ 0.998, V(0) ~ 0.098, closely matching paper's (1, 1, 0.1))
    rho0, T0, V0 = [v[0].item() for v in problem.true_solution(torch.tensor([0.0], dtype=torch.float64))]

    circuits = [
        DifferentiableQuantumCircuit(n_qubits=6, depth=6, feature_map="chebyshev", seed=42 + k)
        for k in range(3)
    ]
    boundaries = [
        BoundaryHandler(mode="floating", x0=0.0, u0=rho0),
        BoundaryHandler(mode="floating", x0=0.0, u0=T0),
        BoundaryHandler(mode="floating", x0=0.0, u0=V0),
    ]

    # --- Stage 1: Subsonic region [0.0, 0.4] ---
    print("=== Fig. 10 Stage 1: Subsonic region [0.0, 0.4] (20 pts, lr=0.01, 200 iters) ===", flush=True)
    x_stage1 = torch.linspace(0.0, 0.4, 20, dtype=torch.float64)
    res_s1 = DQCSolver(
        circuits=circuits,
        boundaries=boundaries,
        residual_fn=problem.residual,
        true_solution_fn=problem.true_solution,
    ).solve(x_stage1, n_iter=200, lr=0.01, log_every=50)
    preds_s1 = res_s1.predict(x_plot)
    print(f"  Stage 1 final | L_F = {res_s1.loss_full[-1]:.3e} | L_Q = {res_s1.loss_quality[-1]:.3e}", flush=True)

    # --- Stage 2: Full nozzle [0.0, 0.4] U [0.6, 0.9] with reverse-sigmoid regularization ---
    print("\n=== Fig. 10 Stage 2: Full nozzle (40 pts, lr=0.005, 600 iters, n_drop=150) ===", flush=True)
    x_super = torch.linspace(0.6, 0.9, 20, dtype=torch.float64)
    x_stage2 = torch.cat([x_stage1, x_super])

    # Regularization points: 20 Stage-1 predictions in [0, 0.4] + 5 weak supersonic bias points in [0.6, 0.9]
    x_reg_sub = x_stage1
    u_reg_sub = [p.detach().clone() for p in res_s1.predict(x_reg_sub)]

    x_reg_sup = torch.linspace(0.62, 0.88, 5, dtype=torch.float64)
    u_reg_sup = problem.true_solution(x_reg_sup)

    x_reg = torch.cat([x_reg_sub, x_reg_sup])
    u_reg = [torch.cat([us, up]) for us, up in zip(u_reg_sub, u_reg_sup)]

    n_iter_s2 = 600

    def reg_loss_fn(step, circs, bnds):
        zeta = regularization_schedule(step=step, n_iter=n_iter_s2, kind="reverse_sigmoid", n_drop=150, delta=0.1)
        if zeta < 1e-6:
            return torch.tensor(0.0, dtype=torch.float64)
        total_reg = torch.tensor(0.0, dtype=torch.float64)
        for k, (c, b) in enumerate(zip(circs, bnds)):
            f_reg = b.eval_only(c, x_reg)
            total_reg = total_reg + regularization_loss(f_reg, u_reg[k], weight=0.1 * zeta)
        return total_reg

    res_s2 = DQCSolver(
        circuits=circuits,
        boundaries=boundaries,
        residual_fn=problem.residual,
        true_solution_fn=problem.true_solution,
        reg_loss_fn=reg_loss_fn,
    ).solve(x_stage2, n_iter=n_iter_s2, lr=0.005, log_every=100)
    preds_s2 = res_s2.predict(x_plot)
    print(f"  Stage 2 final | L_F = {res_s2.loss_full[-1]:.3e} | L_Q = {res_s2.loss_quality[-1]:.3e}", flush=True)

    # Plot Fig. 10(a, b, c)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = [r"\rho", "T", "V"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    trues = [rho_true, T_true, V_true]

    for ax, preds, title in [
        (axes[0], preds_s1, "Fig. 10(a): Stage 1 (Subsonic [0, 0.4])"),
        (axes[1], preds_s2, "Fig. 10(b): Stage 2 (Full Nozzle)"),
    ]:
        for tr, pr, lbl, col in zip(trues, preds, labels, colors):
            ax.plot(x_plot.numpy(), tr.numpy(), color=col, linewidth=2.0, label=f"${lbl}$ exact")
            ax.plot(x_plot.numpy(), pr.numpy(), color=col, linestyle="--", linewidth=1.7, label=f"$f_{{{lbl}}}$ DQC")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_ylim(0.0, 2.0)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    axes[2].semilogy(res_s2.loss_full, color="tab:red", label="$L_F$ (Full)")
    axes[2].semilogy(res_s2.loss_diff, color="tab:green", linestyle="--", label="$L_D$ (Diff)")
    axes[2].semilogy(res_s2.loss_quality, color="tab:purple", label="$L_Q$ (Quality)")
    axes[2].axvline(150, color="gray", linestyle=":", label="$n_{\\mathrm{drop}}=150$")
    axes[2].set_title("Fig. 10(c): Stage 2 Training Metrics")
    axes[2].set_xlabel("Iteration $n_j$")
    axes[2].set_ylabel("Loss")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved figure to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
