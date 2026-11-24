"""Example 02: Evaluating a Differentiable Quantum Circuit f(x) and df(x)/dx.

Run with:
    python 2011.10395v2/examples/02_differentiable_circuit.py
"""

import torch

from dqc.circuit import DifferentiableQuantumCircuit


def main() -> None:
    dqc = DifferentiableQuantumCircuit(
        n_qubits=4,
        depth=3,
        feature_map="chebyshev_tower",
        seed=42,
    )
    xs = torch.linspace(0.0, 0.8, 5, dtype=torch.float64)

    # 1. Exact value and derivative via PennyLane backprop + PyTorch autograd
    f_val, df_auto = dqc.eval_with_derivative(xs)

    # 2. Exact hardware parameter-shift rule (Eq. 2 / Fig. 3b)
    df_pshift = dqc.df_dx_parameter_shift(xs)

    # 3. Numerical finite difference for comparison
    eps = 1e-6
    with torch.no_grad():
        df_fd = (dqc.raw_eval(xs + eps) - dqc.raw_eval(xs - eps)) / (2.0 * eps)

    print("=== DQC Evaluation (N=4, depth=3, chebyshev_tower) ===")
    print(f"{'x':>6} | {'f(x)':>10} | {'df/dx (auto)':>13} | {'df/dx (pshift)':>14} | {'df/dx (FD)':>12}")
    print("-" * 68)
    for i in range(len(xs)):
        print(
            f"{xs[i].item():6.2f} | "
            f"{f_val[i].item():+10.6f} | "
            f"{df_auto[i].item():+13.6f} | "
            f"{df_pshift[i].item():+14.6f} | "
            f"{df_fd[i].item():+12.6f}"
        )

    max_diff = (df_auto - df_pshift).abs().max().item()
    print(f"\nMax |df_auto - df_pshift| = {max_diff:.2e}")

    # 4. Verify gradient flow into variational parameters theta through df/dx
    dummy_ode_loss = torch.mean((df_auto + f_val) ** 2)
    dummy_ode_loss.backward()
    print(f"Dummy ODE loss = {dummy_ode_loss.item():.6f}")
    print(f"||grad_theta L||_2 = {dqc.params.grad.norm().item():.6f} (shape {tuple(dqc.params.grad.shape)})")


if __name__ == "__main__":
    main()
