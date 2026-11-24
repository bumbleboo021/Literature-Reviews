"""Batched statevector gates. Qubit 0 is the leftmost (most significant) bit."""

import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1., -1.]).astype(complex)
PAULI = {"I": I, "X": X, "Y": Y, "Z": Z}


def rotation(axis, angle):
    """R_axis(angle) = exp(-i angle P_axis / 2)."""
    if axis not in ("X", "Y", "Z"):
        raise ValueError("Rotation axis must be X, Y, or Z.")
    return np.cos(angle/2)*I - 1j*np.sin(angle/2)*PAULI[axis]


def rx(angle):
    return rotation("X", angle)


def ry(angle):
    return rotation("Y", angle)


def rz(angle):
    return rotation("Z", angle)


def apply_gate(states, gate, qubit, n_qubits):
    if not 0 <= qubit < n_qubits:
        raise ValueError("Qubit index out of range.")
    a = np.asarray(states, dtype=complex).reshape((-1,) + (2,)*n_qubits)
    a = np.moveaxis(a, qubit+1, -1) @ np.asarray(gate).T
    return np.moveaxis(a, -1, qubit+1).reshape(-1, 2**n_qubits)


def apply_cnot(states, control, target, n_qubits):
    if control == target or not (0 <= control < n_qubits and 0 <= target < n_qubits):
        raise ValueError("CNOT needs distinct valid control and target qubits.")
    indices = np.arange(2**n_qubits)
    active = (indices >> (n_qubits-1-control)) & 1
    permutation = indices ^ (active << (n_qubits-1-target))
    return states[:, permutation]


def pauli_string(label):
    if not label or any(p not in PAULI for p in label):
        raise ValueError("Use a nonempty Pauli string, e.g. 'ZXI'.")
    result = np.ones((1, 1), dtype=complex)
    for p in label:
        result = np.kron(result, PAULI[p])
    return result

