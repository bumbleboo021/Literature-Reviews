"""Hermitian readouts and Hamiltonians: paper Eqs. (16)-(18)."""

from dataclasses import dataclass
import numpy as np
from .gates import pauli_string


def single_pauli(n_qubits, qubit, axis="Z"):
    if n_qubits < 1 or not 0 <= qubit < n_qubits:
        raise ValueError("Invalid number of qubits or qubit index.")
    label = ["I"]*n_qubits
    label[qubit] = axis
    return pauli_string("".join(label))


def z_observables(n_qubits):
    return np.stack([single_pauli(n_qubits, j) for j in range(n_qubits)])


def total_magnetization(n_qubits, weights=None):
    w = np.ones(n_qubits) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n_qubits,):
        raise ValueError("Magnetization needs one weight per qubit.")
    return np.einsum("k,kij->ij", w, z_observables(n_qubits))


def ising_hamiltonian(n_qubits, couplings=1.0, transverse=0.0, longitudinal=0.0):
    """Open chain sum J_j Z_j Z_(j+1) + hx_j X_j + hz_j Z_j (Eq. 16).

    For the Fig. 7 convention -J ZZ + h X, pass negative couplings.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive.")
    js = np.broadcast_to(np.asarray(couplings, float), (n_qubits-1,))
    hx = np.broadcast_to(np.asarray(transverse, float), (n_qubits,))
    hz = np.broadcast_to(np.asarray(longitudinal, float), (n_qubits,))
    result = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for j in range(n_qubits):
        result += hx[j]*single_pauli(n_qubits, j, "X") + hz[j]*single_pauli(n_qubits, j)
    for j, coupling in enumerate(js):
        label = ["I"]*n_qubits
        label[j] = label[j+1] = "Z"
        result += coupling*pauli_string("".join(label))
    return result


def spin_glass(couplings, longitudinal=0.0):
    """Eq. (17). A symmetric zero-diagonal J matrix; count each pair once."""
    js = np.asarray(couplings, float)
    if js.ndim != 2 or js.shape[0] != js.shape[1] or js.shape[0] < 1:
        raise ValueError("couplings must be a nonempty square matrix.")
    if not np.allclose(js, js.T) or not np.allclose(np.diag(js), 0):
        raise ValueError("Use symmetric J with zero diagonal.")
    n = len(js)
    result = total_magnetization(n, np.broadcast_to(longitudinal, (n,)))
    for i in range(n):
        for j in range(i+1, n):
            label = ["I"]*n
            label[i] = label[j] = "Z"
            result += js[i, j]*pauli_string("".join(label))
    return result


@dataclass
class Readout:
    """O outputs as W_(o,k) <C_k>. W may be fixed or trained (Eq. 18)."""

    matrices: np.ndarray
    weights: np.ndarray
    trainable: bool = False

    def __post_init__(self):
        self.matrices = np.asarray(self.matrices, complex).copy()
        if self.matrices.ndim == 2:
            self.matrices = self.matrices[None]
        if (self.matrices.ndim != 3 or self.matrices.shape[1] != self.matrices.shape[2]
                or self.matrices.shape[0] == 0):
            raise ValueError("Readout matrices must have shape (K, dim, dim).")
        if not np.allclose(self.matrices, self.matrices.conj().transpose(0, 2, 1)):
            raise ValueError("Every observable must be Hermitian.")
        self.weights = np.atleast_2d(np.asarray(self.weights, float)).copy()
        if self.weights.shape[0] == 0 or self.weights.shape[1] != len(self.matrices):
            raise ValueError("Readout weights must have shape (outputs, observables).")
        if not np.all(np.isfinite(self.weights)) or not np.all(np.isfinite(self.matrices)):
            raise ValueError("Readout must be finite.")
        self._diagonals = np.diagonal(self.matrices, axis1=1, axis2=2).real.copy()
        diagonal_matrices = np.zeros_like(self.matrices)
        indices = np.arange(self.matrices.shape[1])
        diagonal_matrices[:, indices, indices] = self._diagonals
        self._all_diagonal = np.array_equal(self.matrices, diagonal_matrices)

    def expectations(self, states):
        """Read all observables; exploit diagonal Z/ZZ costs without dense products."""
        if self._all_diagonal:
            return (abs(states)**2) @ self._diagonals.T
        return np.einsum("bi,kij,bj->bk", states.conj(), self.matrices, states, optimize=True).real

    @property
    def n_outputs(self):
        return self.weights.shape[0]

    @classmethod
    def magnetization(cls, n_qubits):
        return cls(total_magnetization(n_qubits), [[1.]])

    @classmethod
    def weighted_z(cls, n_qubits, n_outputs=1, trainable=True, seed=0):
        weights = np.random.default_rng(seed).normal(0, 0.5, (n_outputs, n_qubits))
        return cls(z_observables(n_qubits), weights, trainable)

