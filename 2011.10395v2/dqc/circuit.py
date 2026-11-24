"""Differentiable Quantum Circuit (DQC) — Section II & III.A of arXiv:2011.10395v2.

Combines:
  1. Quantum feature map U_phi(x)
  2. Variational ansatz U_theta
  3. Cost observable C

to represent a trial function f(x) = <0| U_phi(x)^dag U_theta^dag C U_theta U_phi(x) |0>
and its exact analytical derivative df(x)/dx.
"""

from __future__ import annotations

import math
import pennylane as qml
import torch

from dqc.ansatz import hardware_efficient_ansatz, init_hea_params
from dqc.cost import total_magnetization
from dqc.feature_maps import apply_feature_map


def _feature_map_angles_and_derivs(
    name: str, x: torch.Tensor, n_qubits: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return per-qubit rotation angles phi_j(x) and d(phi_j)/dx for parameter-shift."""
    denom = torch.sqrt(1.0 - x**2)
    if name == "product":
        angle = torch.arcsin(x)
        dangle = 1.0 / denom
        return [angle] * n_qubits, [dangle] * n_qubits
    if name == "chebyshev":
        angle = 2.0 * torch.arccos(x)
        dangle = -2.0 / denom
        return [angle] * n_qubits, [dangle] * n_qubits
    if name == "chebyshev_tower":
        base = 2.0 * torch.arccos(x)
        dbase = -2.0 / denom
        angles = [j * base for j in range(1, n_qubits + 1)]
        dangles = [j * dbase for j in range(1, n_qubits + 1)]
        return angles, dangles
    raise ValueError(f"Unknown feature map '{name}'")


class DifferentiableQuantumCircuit:
    """Differentiable Quantum Circuit (DQC) for representing u(x) and du/dx.

    Uses PennyLane's `default.qubit` with `diff_method="backprop"` and PyTorch
    autograd for fast, exact analytical differentiation during simulation, and
    also exposes `df_dx_parameter_shift` to demonstrate the exact hardware
    parameter-shift rule (Eq. 2 / Fig. 3b).
    """

    def __init__(
        self,
        n_qubits: int = 6,
        depth: int = 5,
        feature_map: str = "chebyshev_tower",
        cost_op: qml.operation.Operator | None = None,
        evolution_unitary: torch.Tensor | None = None,
        seed: int | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.feature_map = feature_map
        self.wires = list(range(n_qubits))
        self.dtype = dtype
        self.cost_op = total_magnetization(self.wires) if cost_op is None else cost_op
        self.evolution_unitary = evolution_unitary

        self.params = init_hea_params(n_qubits, depth, seed=seed, dtype=dtype)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _qnode(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
            apply_feature_map(self.feature_map, x, self.wires)
            if self.evolution_unitary is not None:
                qml.QubitUnitary(self.evolution_unitary, wires=self.wires)
            hardware_efficient_ansatz(theta, self.wires)
            return qml.expval(self.cost_op)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _shifted_qnode(angles: list[torch.Tensor], theta: torch.Tensor) -> torch.Tensor:
            for w, ang in zip(self.wires, angles):
                qml.RY(ang, wires=w)
            if self.evolution_unitary is not None:
                qml.QubitUnitary(self.evolution_unitary, wires=self.wires)
            hardware_efficient_ansatz(theta, self.wires)
            return qml.expval(self.cost_op)

        self._qnode = _qnode
        self._shifted_qnode = _shifted_qnode

    def raw_eval(self, x: torch.Tensor, params: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate the raw expectation value <f_{phi,theta}(x) | C | f_{phi,theta}(x)> (Eq. 1).

        Args:
            x: Scalar or 1D tensor of grid points in (-1, 1).
            params: Optional override for variational parameters.
        """
        theta = self.params if params is None else params
        return self._qnode(x, theta)

    def eval_with_derivative(
        self, x: torch.Tensor, params: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (f(x), df(x)/dx) with `create_graph=True` so Adam can backprop through df/dx."""
        x_req = x.detach().clone().to(dtype=self.dtype).requires_grad_(True)
        f_val = self.raw_eval(x_req, params=params)
        df_dx = torch.autograd.grad(
            outputs=f_val.sum(),
            inputs=x_req,
            create_graph=True,
        )[0]
        return f_val, df_dx

    def df_dx_parameter_shift(
        self, x: torch.Tensor, params: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute exact df/dx via the hardware parameter-shift rule (Eq. 2 / Fig. 3b).

        df/dx = (1/2) * sum_{j=1..N} (d phi_j / dx) * (<C>_{phi_j + pi/2} - <C>_{phi_j - pi/2})
        """
        theta = self.params if params is None else params
        angles, dangles = _feature_map_angles_and_derivs(self.feature_map, x, self.n_qubits)
        shift = math.pi / 2.0

        total = torch.zeros_like(x, dtype=self.dtype)
        for j in range(self.n_qubits):
            angles_plus = list(angles)
            angles_minus = list(angles)
            angles_plus[j] = angles[j] + shift
            angles_minus[j] = angles[j] - shift

            c_plus = self._shifted_qnode(angles_plus, theta)
            c_minus = self._shifted_qnode(angles_minus, theta)
            total = total + 0.5 * dangles[j] * (c_plus - c_minus)
        return total
