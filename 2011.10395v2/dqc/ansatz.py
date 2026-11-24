"""Variational quantum circuits (ansatze) from Section III.B of arXiv:2011.10395v2."""

from __future__ import annotations

from collections.abc import Sequence
import math
import pennylane as qml
import torch


def hea_param_shape(n_qubits: int, depth: int) -> tuple[int, int, int]:
    """Return the parameter tensor shape (depth, n_qubits, 3) for the HEA."""
    return (depth, n_qubits, 3)


def init_hea_params(
    n_qubits: int,
    depth: int,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Initialize HEA angles uniformly in [0, 2*pi] (Sec. III.E / III.H)."""
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    shape = hea_param_shape(n_qubits, depth)
    params = torch.rand(shape, generator=gen, dtype=dtype) * (2.0 * math.pi)
    return params.requires_grad_(True)


def hardware_efficient_ansatz(params: torch.Tensor, wires: Sequence[int]) -> None:
    """Hardware-Efficient Ansatz (HEA) — Section III.B, Fig. 5(a).

    Consists of `depth` repeated blocks. Each block applies:
      1. Single-qubit Rz(theta_0) -> Rx(theta_1) -> Rz(theta_2) on every qubit
         (an arbitrary SU(2) rotation per qubit).
      2. Linear nearest-neighbor CNOT entangling layer: CNOT(w_i, w_{i+1}).

    Args:
        params: Tensor of shape (depth, len(wires), 3).
        wires: Ordered sequence of qubit indices.
    """
    n_qubits = len(wires)
    if params.ndim != 3 or params.shape[1] != n_qubits or params.shape[2] != 3:
        raise ValueError(
            f"Expected params shape (depth, {n_qubits}, 3), got {tuple(params.shape)}"
        )

    depth = params.shape[0]
    for layer in range(depth):
        for idx, w in enumerate(wires):
            qml.RZ(params[layer, idx, 0], wires=w)
            qml.RX(params[layer, idx, 1], wires=w)
            qml.RZ(params[layer, idx, 2], wires=w)
        for idx in range(n_qubits - 1):
            qml.CNOT(wires=[wires[idx], wires[idx + 1]])
