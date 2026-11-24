"""Unit tests for variational ansatz (Sec. III.B) and cost operators (Sec. III.C)."""

import math
import pennylane as qml
import pytest
import torch

from dqc.ansatz import hardware_efficient_ansatz, hea_param_shape, init_hea_params
from dqc.cost import ising_cost_hamiltonian, total_magnetization


def test_hea_zero_params_leaves_zero_state_unchanged():
    """With all angles = 0, Rz(0) = Rx(0) = I and CNOT|00..0> = |00..0>,
    so <sum_j Z_j> must equal N.
    """
    n_qubits, depth = 4, 3
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)
    cost_op = total_magnetization(wires)

    @qml.qnode(dev, interface="torch")
    def circuit(params):
        hardware_efficient_ansatz(params, wires)
        return qml.expval(cost_op)

    zeros = torch.zeros(hea_param_shape(n_qubits, depth), dtype=torch.float64)
    assert torch.allclose(circuit(zeros), torch.tensor(float(n_qubits), dtype=torch.float64), atol=1e-7)


def test_hea_single_qubit_rx_pi_flips_magnetization():
    """On 1 qubit with Rz=0, Rx=pi, Rz=0, |0> flips to |1> so <Z> = -1."""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface="torch")
    def circuit(params):
        hardware_efficient_ansatz(params, wires=[0])
        return qml.expval(total_magnetization([0]))

    params = torch.tensor([[[0.0, math.pi, 0.0]]], dtype=torch.float64)
    assert torch.allclose(circuit(params), torch.tensor(-1.0, dtype=torch.float64), atol=1e-7)


def test_init_hea_params_shape_range_and_reproducibility():
    p1 = init_hea_params(n_qubits=6, depth=5, seed=42)
    p2 = init_hea_params(n_qubits=6, depth=5, seed=42)
    assert p1.shape == (5, 6, 3)
    assert p1.requires_grad
    assert torch.equal(p1, p2)
    assert (p1 >= 0.0).all() and (p1 <= 2.0 * math.pi).all()


def test_hea_invalid_param_shape_raises():
    with pytest.raises(ValueError, match="Expected params shape"):
        hardware_efficient_ansatz(torch.zeros(2, 3, 2), wires=[0, 1, 2])


def test_ising_cost_hamiltonian_on_zero_state():
    """On |000>, <Z_i Z_{i+1}> = +1, <Z_i> = +1, <X_i> = 0."""
    wires = [0, 1, 2]
    dev = qml.device("default.qubit", wires=3)
    ham = ising_cost_hamiltonian(wires, couplings=[1.0, 2.0], hz=[0.5, 0.5, 0.5], hx=[1.0, 1.0, 1.0])

    @qml.qnode(dev, interface="torch")
    def circuit():
        return qml.expval(ham)

    # Expected: (1.0 + 2.0) + (0.5 + 0.5 + 0.5) + 0 = 4.5
    assert torch.allclose(circuit(), torch.tensor(4.5, dtype=torch.float64), atol=1e-7)
