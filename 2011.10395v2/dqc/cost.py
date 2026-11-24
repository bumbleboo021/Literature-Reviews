"""Cost function observables from Section III.C of arXiv:2011.10395v2.

The trial function is read out as the expectation value of a Hermitian
cost operator C:

    f(x) = <f_{phi, theta}(x) | C | f_{phi, theta}(x)>   (Eq. 1)
"""

from __future__ import annotations

from collections.abc import Sequence
import pennylane as qml


def total_magnetization(wires: Sequence[int]) -> qml.operation.Operator:
    """Total magnetization in the Z direction: C = sum_j Z_j.

    This is the primary cost operator used in all Section IV benchmarks
    (Figs. 6, 7, 8, and 10). Its eigenvalues span [-N, +N] for N qubits.
    """
    if len(wires) == 0:
        raise ValueError("wires must be non-empty")
    if len(wires) == 1:
        return qml.PauliZ(wires[0])
    return qml.sum(*[qml.PauliZ(w) for w in wires])


def single_qubit_magnetization(wire: int = 0) -> qml.operation.Operator:
    """Single-qubit Z magnetization: C = Z_j (eigenvalues in [-1, 1])."""
    return qml.PauliZ(wire)


def ising_cost_hamiltonian(
    wires: Sequence[int],
    couplings: Sequence[float] | None = None,
    hz: Sequence[float] | None = None,
    hx: Sequence[float] | None = None,
) -> qml.operation.Operator:
    """Ising Hamiltonian with transverse & longitudinal fields — Eq. (16).

    C = sum_j J_{j,j+1} Z_j Z_{j+1} + sum_j h^z_j Z_j + sum_j h^x_j X_j
    """
    n = len(wires)
    if n < 2:
        raise ValueError("Ising cost Hamiltonian requires at least 2 wires")

    couplings = [1.0] * (n - 1) if couplings is None else list(couplings)
    hz = [0.0] * n if hz is None else list(hz)
    hx = [0.0] * n if hx is None else list(hx)

    coeffs: list[float] = []
    ops: list[qml.operation.Operator] = []

    for i in range(n - 1):
        if couplings[i] != 0.0:
            coeffs.append(float(couplings[i]))
            ops.append(qml.PauliZ(wires[i]) @ qml.PauliZ(wires[i + 1]))
    for i in range(n):
        if hz[i] != 0.0:
            coeffs.append(float(hz[i]))
            ops.append(qml.PauliZ(wires[i]))
        if hx[i] != 0.0:
            coeffs.append(float(hx[i]))
            ops.append(qml.PauliX(wires[i]))

    return qml.Hamiltonian(coeffs, ops)
