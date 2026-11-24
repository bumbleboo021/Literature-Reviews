"""HEA and checkerboard alternating blocks, Sec. III B / Fig. 5.

Every angle belongs to exactly ONE rotation gate, allowing simple parameter shifts.
"""

import numpy as np
from .gates import apply_gate, apply_cnot, rotation


class _Ansatz:
    @property
    def n_parameters(self):
        return sum(gate[0] != "CNOT" for gate in self.gates)

    def apply(self, theta, states):
        theta = np.asarray(theta, float)
        if theta.shape != (self.n_parameters,) or not np.all(np.isfinite(theta)):
            raise ValueError("Invalid ansatz parameters.")
        result = np.asarray(states, complex).copy()
        p = 0
        for axis, q, target in self.gates:
            if axis == "CNOT":
                result = apply_cnot(result, q, target, self.n_qubits)
            else:
                result = apply_gate(result, rotation(axis, theta[p]), q, self.n_qubits)
                p += 1
        return result

    def _block(self, qubits, depth, axes):
        for _ in range(depth):
            # Execute Rz, Rx, Rz in this time order for the paper's HEA.
            for q in qubits:
                self.gates.extend((axis, q, None) for axis in axes)
            self.gates.extend(("CNOT", a, b) for a, b in zip(qubits[:-1], qubits[1:]))

    def _validate(self, n_qubits, depth, axes):
        if not isinstance(n_qubits, int) or n_qubits < 1 or not isinstance(depth, int) or depth < 0:
            raise ValueError("Need positive integer n_qubits and nonnegative integer depth.")
        if not axes or any(a not in "XYZ" for a in axes):
            raise ValueError("axes must contain X/Y/Z rotation labels.")
        self.n_qubits, self.gates = n_qubits, []


class HardwareEfficientAnsatz(_Ansatz):
    def __init__(self, n_qubits, depth=5, axes="ZXZ"):
        self._validate(n_qubits, depth, axes)
        self._block(list(range(n_qubits)), depth, axes)


class AlternatingBlocksAnsatz(_Ansatz):
    def __init__(self, n_qubits, layers=3, block_width=2, block_depth=1, axes="ZXZ"):
        self._validate(n_qubits, layers, axes)
        if (not isinstance(block_width, int) or block_width < 2
                or not isinstance(block_depth, int) or block_depth < 1):
            raise ValueError("Need integer block_width >= 2 and block_depth >= 1.")
        for layer in range(layers):
            # Shift every second partition; retain shorter blocks at both boundaries.
            offset = 0 if layer % 2 == 0 else block_width//2
            starts = list(range(offset, n_qubits, block_width))
            if offset:
                self._block(list(range(min(offset, n_qubits))), block_depth, axes)
            for start in starts:
                self._block(list(range(start, min(start+block_width, n_qubits))), block_depth, axes)

