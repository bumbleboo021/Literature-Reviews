"""Quantum feature maps from Section III.A of arXiv:2011.10395v2.

Each feature map encodes a scalar (or 1D batched) variable x in (-1, 1)
into quantum state amplitudes using single-qubit R_y rotations:

    R_y(phi) = exp(-i * (phi / 2) * Y)

Because PennyLane circuits are built by queueing operations inside a QNode,
these functions do not create a device or return a value — they simply apply
gates onto the provided wires.
"""

from __future__ import annotations

from collections.abc import Sequence
import pennylane as qml
import torch


def product_feature_map(x: torch.Tensor, wires: Sequence[int]) -> None:
    """Product feature map — Eq. (6).

    U_phi(x) = tensor_prod_{j=1..N} R_{y, j}(arcsin(x))
    """
    angle = torch.arcsin(x)
    for w in wires:
        qml.RY(angle, wires=w)


def chebyshev_feature_map(x: torch.Tensor, wires: Sequence[int]) -> None:
    """Sparse Chebyshev feature map — Eq. (14).

    U_phi(x) = tensor_prod_{j=1..N} R_{y, j}(2 * arccos(x))

    The factor of 2 cancels the 1/2 inside R_y(phi) = exp(-i * phi/2 * Y),
    producing degree-1 Chebyshev polynomial amplitudes T_1(x) = cos(arccos(x)) = x
    on each qubit (Eq. 12).
    """
    angle = 2.0 * torch.arccos(x)
    for w in wires:
        qml.RY(angle, wires=w)


def chebyshev_tower_feature_map(x: torch.Tensor, wires: Sequence[int]) -> None:
    """Chebyshev tower feature map — Eq. (15).

    U_phi(x) = tensor_prod_{j=1..N} R_{y, j}(2 * j * arccos(x))

    Note that j is 1-indexed (j = 1, 2, ..., N) so qubit 0 gets degree 1,
    qubit 1 gets degree 2, ..., creating a tower of Chebyshev polynomials
    T_1(x), T_2(x), ..., T_N(x).
    """
    base_angle = 2.0 * torch.arccos(x)
    for j, w in enumerate(wires, start=1):
        qml.RY(j * base_angle, wires=w)


FEATURE_MAPS = {
    "product": product_feature_map,
    "chebyshev": chebyshev_feature_map,
    "chebyshev_tower": chebyshev_tower_feature_map,
}


def apply_feature_map(name: str, x: torch.Tensor, wires: Sequence[int]) -> None:
    """Dispatch a feature map by name."""
    if name not in FEATURE_MAPS:
        valid = ", ".join(sorted(FEATURE_MAPS))
        raise ValueError(f"Unknown feature map '{name}'. Choose from: {valid}")
    FEATURE_MAPS[name](x, wires)
