"""Controlled-Rotation Ansatz (n=3 generator -> 4-term Lagrange GPSR) & 2nd-order DQC.

Extends baseline `dqc/` without modifying it:
  - Replaces fixed nearest-neighbor CNOTs with trainable CRY(phi) gates, whose
    generator |1><1| x (Y/2) has 3 distinct eigenvalues {-1/2, 0, +1/2} and is
    hardware-differentiable via the 4-term Lagrange parameter-shift rule.
  - Exposes `eval_with_second_derivative(x)` returning (f, df/dx, d2f/dx2) for
    solving 2nd-order ODEs.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import pennylane as qml
import torch

from dqc.cost import total_magnetization
from dqc.feature_maps import apply_feature_map


def init_cr_hea_params(
    n_qubits: int,
    depth: int,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize (single_qubit_params, entangling_cry_params) uniformly in [0, 2*pi].

    Shapes:
      - rot_params: (depth, n_qubits, 3) for Rz-Rx-Rz
      - ent_params: (depth, n_qubits - 1) for trainable CRY(phi) gates
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    rot_params = (torch.rand((depth, n_qubits, 3), generator=gen, dtype=dtype) * (2.0 * math.pi)).requires_grad_(True)
    ent_params = (torch.rand((depth, n_qubits - 1), generator=gen, dtype=dtype) * (2.0 * math.pi)).requires_grad_(True)
    return rot_params, ent_params


def controlled_rotation_ansatz(
    rot_params: torch.Tensor,
    ent_params: torch.Tensor,
    wires: Sequence[int],
) -> None:
    """Controlled-Rotation HEA: `depth` layers of Rz-Rx-Rz + trainable nearest-neighbor CRY."""
    n_qubits = len(wires)
    depth = rot_params.shape[0]
    for layer in range(depth):
        for idx, w in enumerate(wires):
            qml.RZ(rot_params[layer, idx, 0], wires=w)
            qml.RX(rot_params[layer, idx, 1], wires=w)
            qml.RZ(rot_params[layer, idx, 2], wires=w)
        for idx in range(n_qubits - 1):
            qml.CRY(ent_params[layer, idx], wires=[wires[idx], wires[idx + 1]])


class ControlledRotationDQC:
    """DQC using trainable CRY entanglers (4-term Lagrange GPSR compatible) and 2nd-order autodiff."""

    def __init__(
        self,
        n_qubits: int = 6,
        depth: int = 5,
        feature_map: str = "chebyshev_tower",
        cost_op: qml.operation.Operator | None = None,
        seed: int | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.feature_map = feature_map
        self.wires = list(range(n_qubits))
        self.dtype = dtype
        self.cost_op = total_magnetization(self.wires) if cost_op is None else cost_op

        self.rot_params, self.ent_params = init_cr_hea_params(n_qubits, depth, seed=seed, dtype=dtype)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _qnode(x: torch.Tensor, rot: torch.Tensor, ent: torch.Tensor) -> torch.Tensor:
            apply_feature_map(self.feature_map, x, self.wires)
            controlled_rotation_ansatz(rot, ent, self.wires)
            return qml.expval(self.cost_op)

        self._qnode = _qnode

    def trainable_params(self) -> list[torch.Tensor]:
        return [self.rot_params, self.ent_params]

    def raw_eval(self, x: torch.Tensor) -> torch.Tensor:
        return self._qnode(x, self.rot_params, self.ent_params)

    def eval_with_derivative(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_req = x.detach().clone().to(dtype=self.dtype).requires_grad_(True)
        f_val = self.raw_eval(x_req)
        df_dx = torch.autograd.grad(f_val.sum(), x_req, create_graph=True)[0]
        return f_val, df_dx

    def eval_with_second_derivative(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (f(x), df/dx, d^2f/dx^2) with full autograd graph for 2nd-order ODEs."""
        x_req = x.detach().clone().to(dtype=self.dtype).requires_grad_(True)
        f_val = self.raw_eval(x_req)
        df_dx = torch.autograd.grad(f_val.sum(), x_req, create_graph=True)[0]
        d2f_dx2 = torch.autograd.grad(df_dx.sum(), x_req, create_graph=True)[0]
        return f_val, df_dx, d2f_dx2
