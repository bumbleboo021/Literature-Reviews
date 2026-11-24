"""Example 01: Inspecting Feature Maps, HEA Ansatz, and Cost Observable.

Run with:
    python 2011.10395v2/examples/01_feature_maps_and_ansatz.py
"""

import pennylane as qml
import torch

from dqc.ansatz import hardware_efficient_ansatz, init_hea_params
from dqc.cost import total_magnetization
from dqc.feature_maps import apply_feature_map


def main() -> None:
    n_qubits = 3
    depth = 2
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)
    cost_op = total_magnetization(wires)

    # Random variational angles of shape (depth, n_qubits, 3)
    params = init_hea_params(n_qubits=n_qubits, depth=depth, seed=42)
    print(f"Ansatz parameter shape: {tuple(params.shape)} ({params.numel()} trainable angles)")
    print(f"Cost operator: {cost_op}\n")

    # 1. Draw the circuit for Chebyshev Tower + HEA (like Fig. 2a + Fig. 5a)
    @qml.qnode(dev, interface="torch")
    def demo_circuit(x, theta):
        apply_feature_map("chebyshev_tower", x, wires)
        hardware_efficient_ansatz(theta, wires)
        return qml.expval(cost_op)

    x_sample = torch.tensor(0.5, dtype=torch.float64)
    print("=== Circuit Diagram (N=3 qubits, depth=2, Chebyshev Tower) ===")
    print(qml.draw(demo_circuit)(x_sample, params))
    print()

    # 2. Compare raw expectation values f(x) = <C> across all 3 feature maps
    xs = torch.tensor([0.0, 0.3, 0.6, 0.9], dtype=torch.float64)
    print("=== Expectation values f(x) = <sum_j Z_j> for each feature map ===")
    for fmap_name in ["product", "chebyshev", "chebyshev_tower"]:
        @qml.qnode(dev, interface="torch")
        def circuit(x, theta):
            apply_feature_map(fmap_name, x, wires)
            hardware_efficient_ansatz(theta, wires)
            return qml.expval(cost_op)

        values = [circuit(xi, params).item() for xi in xs]
        formatted = ", ".join(f"{v:+.4f}" for v in values)
        print(f"  {fmap_name:<16} -> x={xs.tolist()} -> f(x)=[{formatted}]")


if __name__ == "__main__":
    main()
