"""Unit tests verifying Lagrange GPSR for n=2, n=3, n=4 generators and 2nd-order derivatives."""

import math
import numpy as np
import pennylane as qml
import pytest

from dqc.extensions.lagrange_gpsr import (
    apply_second_order_shift_rule,
    apply_shift_rule,
    derive_lagrange_shift_rule,
)


def test_two_term_rule_pauli_ry():
    """n = 2: PennyLane RY(x) = exp(-i * x * Y/2) has eigenvalues {-0.5, +0.5} -> 2-term rule."""
    rule = derive_lagrange_shift_rule([-0.5, 0.5])
    assert rule.n_terms == 2
    assert math.isclose(rule.shifts[0], math.pi / 2.0, abs_tol=1e-12)
    assert math.isclose(rule.coeffs[0], 0.5, abs_tol=1e-12)

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(0.7, wires=0)
        qml.RY(x, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RZ(1.1, wires=1)
        return qml.expval(qml.PauliZ(0) + qml.PauliZ(1))

    x0 = 0.42
    eps = 1e-6
    df_gpsr = apply_shift_rule(lambda x: float(circuit(x)), x0, rule)
    df_fd = (float(circuit(x0 + eps)) - float(circuit(x0 - eps))) / (2.0 * eps)
    assert math.isclose(df_gpsr, df_fd, abs_tol=1e-7)


def test_four_term_rule_controlled_rotation_cry():
    """n = 3: PennyLane CRY(x) has generator |1><1| x (Y/2) with eigenvalues {-0.5, 0, +0.5} -> 4-term rule."""
    rule = derive_lagrange_shift_rule([-0.5, 0.0, 0.5])
    assert rule.n_terms == 4

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.RY(0.9, wires=0)
        qml.RX(0.4, wires=1)
        qml.CRY(x, wires=[0, 1])
        qml.RZ(0.6, wires=1)
        return qml.expval(qml.PauliZ(0) + qml.PauliZ(1))

    x0 = 1.23
    eps = 1e-6
    df_gpsr = apply_shift_rule(lambda x: float(circuit(x)), x0, rule)
    df_fd = (float(circuit(x0 + eps)) - float(circuit(x0 - eps))) / (2.0 * eps)
    assert math.isclose(df_gpsr, df_fd, abs_tol=1e-7)


def test_multi_term_rule_global_3qubit_rotation():
    """n = 4: Global 3-qubit rotation exp(-i * x * sum_j Y_j / 2) has eigenvalues {-1.5, -0.5, 0.5, 1.5} -> 6-term rule."""
    rule = derive_lagrange_shift_rule([-1.5, -0.5, 0.5, 1.5])
    assert rule.n_terms == 6

    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(0.3, wires=0)
        qml.RX(0.8, wires=1)
        qml.RX(1.2, wires=2)
        # Global rotation U(x) = exp(-i * x * (Y0 + Y1 + Y2) / 2) = RY(x) x RY(x) x RY(x)
        for w in range(3):
            qml.RY(x, wires=w)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.RZ(0.5, wires=2)
        return qml.expval(qml.PauliZ(0) + qml.PauliZ(1) + qml.PauliZ(2))

    x0 = 0.65
    eps = 1e-6
    df_gpsr = apply_shift_rule(lambda x: float(circuit(x)), x0, rule)
    df_fd = (float(circuit(x0 + eps)) - float(circuit(x0 - eps))) / (2.0 * eps)
    assert math.isclose(df_gpsr, df_fd, abs_tol=1e-6)


def test_second_order_mixed_and_diagonal_derivatives():
    """Section 4.1, Eq. (24): verify d^2 C / (dx1 dx2) for mixed RY (n=2) and CRY (n=3) gates."""
    r_ry = derive_lagrange_shift_rule([-0.5, 0.5])
    r_cry = derive_lagrange_shift_rule([-0.5, 0.0, 0.5])

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x1, x2):
        qml.RY(x1, wires=0)
        qml.RX(0.5, wires=1)
        qml.CRY(x2, wires=[0, 1])
        qml.RY(0.7, wires=1)
        return qml.expval(qml.PauliZ(0) + qml.PauliZ(1))

    x1, x2 = 0.8, 1.1
    eps = 1e-4

    d2_gpsr = apply_second_order_shift_rule(
        lambda a, b: float(circuit(a, b)), x1, x2, r_ry, r_cry
    )
    d2_fd = (
        float(circuit(x1 + eps, x2 + eps))
        - float(circuit(x1 + eps, x2 - eps))
        - float(circuit(x1 - eps, x2 + eps))
        + float(circuit(x1 - eps, x2 - eps))
    ) / (4.0 * eps**2)
    assert math.isclose(d2_gpsr, d2_fd, abs_tol=1e-5)


def test_invalid_eigenvalues_raises():
    with pytest.raises(ValueError, match="at least 2 distinct eigenvalues"):
        derive_lagrange_shift_rule([1.0, 1.0])
