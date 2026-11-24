"""Unit tests verifying that feature maps match Eqs. (6), (12), (14), (15)."""

import pennylane as qml
import pytest
import torch

from dqc.feature_maps import (
    apply_feature_map,
    chebyshev_feature_map,
    chebyshev_tower_feature_map,
    product_feature_map,
)


def test_product_feature_map_expectation():
    """For 1 qubit: <0| R_y(arcsin x)^dag Z R_y(arcsin x) |0> = cos(arcsin x) = sqrt(1 - x^2)."""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface="torch")
    def circuit(x):
        product_feature_map(x, wires=[0])
        return qml.expval(qml.PauliZ(0))

    x = torch.tensor(0.6, dtype=torch.float64)
    expected = torch.sqrt(1.0 - x**2)
    assert torch.allclose(circuit(x), expected, atol=1e-7)


def test_chebyshev_feature_map_amplitudes_and_expval():
    """Eq. (12): R_y(2 arccos x)|0> = T_1(x)|0> + sqrt(1 - x^2) U_0(x)|1>,
    and <Z> = cos(2 arccos x) = 2x^2 - 1 = T_2(x).
    """
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface="torch")
    def state_circuit(x):
        chebyshev_feature_map(x, wires=[0])
        return qml.state()

    @qml.qnode(dev, interface="torch")
    def expval_circuit(x):
        chebyshev_feature_map(x, wires=[0])
        return qml.expval(qml.PauliZ(0))

    x = torch.tensor(0.4, dtype=torch.float64)
    state = state_circuit(x)
    assert torch.allclose(state[0].real, x, atol=1e-7)
    assert torch.allclose(state[1].real, torch.sqrt(1.0 - x**2), atol=1e-7)

    t2_x = 2.0 * x**2 - 1.0
    assert torch.allclose(expval_circuit(x), t2_x, atol=1e-7)


def test_chebyshev_tower_feature_map_degrees():
    """Eq. (15): qubit j (1-indexed) encodes degree j -> <Z_j> = cos(2j arccos x) = T_{2j}(x)."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def circuit(x):
        chebyshev_tower_feature_map(x, wires=[0, 1])
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    x = torch.tensor(0.3, dtype=torch.float64)
    z0, z1 = circuit(x)

    t2_x = 2.0 * x**2 - 1.0
    t4_x = 8.0 * x**4 - 8.0 * x**2 + 1.0
    assert torch.allclose(z0, t2_x, atol=1e-7)
    assert torch.allclose(z1, t4_x, atol=1e-7)


def test_apply_feature_map_invalid_name():
    with pytest.raises(ValueError, match="Unknown feature map"):
        apply_feature_map("nonexistent", torch.tensor(0.2), wires=[0])
