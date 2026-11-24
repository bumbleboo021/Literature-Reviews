import numpy as np
import pytest
from numpy.polynomial import chebyshev as cheb

from dqc.gates import rx, ry, rz, apply_cnot, pauli_string
from dqc.ansatz import HardwareEfficientAnsatz, AlternatingBlocksAnsatz
from dqc.feature_maps import FeatureMap, AffineDomain
from dqc.observables import Readout, ising_hamiltonian, spin_glass, z_observables
from dqc.model import QuantumModel, ParallelModel
from dqc.boundaries import Boundary


def test_gate_conventions_and_bit_order():
    for gate in (rx, ry, rz):
        u = gate(0.73)
        np.testing.assert_allclose(u.conj().T @ u, np.eye(2), atol=1e-14)
        np.testing.assert_allclose(gate(-0.73) @ u, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(abs(ry(np.pi/3) @ [1, 0])**2, [0.75, 0.25])
    basis = np.eye(4, dtype=complex)
    np.testing.assert_array_equal(apply_cnot(basis, 0, 1, 2), basis[[0, 1, 3, 2]])
    np.testing.assert_array_equal(apply_cnot(basis, 1, 0, 2), basis[[0, 3, 2, 1]])


@pytest.mark.parametrize("kind,n", [("prod", 1), ("cheb", 1), ("chebt", 3)])
def test_feature_derivatives_against_known_polynomials(kind, n):
    x = np.array([-0.63, -0.17, 0.28, 0.77])
    fmap = FeatureMap(n, kind)
    # Read only the last qubit, so ChebT gives T_(2n), not T_n.
    model = QuantumModel(fmap, HardwareEfficientAnsatz(n, 0),
                         Readout(pauli_string("I"*(n-1)+"Z"), [[1.]]),
                         Boundary("pinned"))
    result = model.predict(np.empty(0), x, order=2)
    if kind == "prod":
        expected = (np.sqrt(1-x*x), -x/np.sqrt(1-x*x), -(1-x*x)**(-1.5))
    else:
        degree = 2*n if kind == "chebt" else 2
        coeff = [0]*degree+[1]
        expected = tuple(cheb.chebval(x, cheb.chebder(coeff, m=d)) for d in range(3))
    for actual, wanted in zip(result.derivatives, expected):
        np.testing.assert_allclose(actual[:, 0], wanted, atol=3e-12)


@pytest.mark.parametrize("kind", ["prod", "cheb", "chebt"])
def test_second_derivative_cross_terms_and_physical_chain_rule(kind):
    domain = AffineDomain.interval(2, 6, interior=0.7)
    hamiltonian = ising_hamiltonian(2, -0.8, 0.4, 0.2)
    fmap = FeatureMap(2, kind, domain, hamiltonian, time=0.7)
    model = QuantumModel(fmap, HardwareEfficientAnsatz(2, 2, "YZ"),
                         Readout.weighted_z(2, trainable=True), Boundary("floating", 2, [1.]))
    p = model.initial_parameters(10)
    x, step = np.array([2.2, 3.7, 5.6]), 1e-4
    e = model.predict(p, x, 2)
    plus, minus = model.predict(p, x+step, 1), model.predict(p, x-step, 1)
    np.testing.assert_allclose(e.dx, (plus.value-minus.value)/(2*step), atol=1e-7)
    np.testing.assert_allclose(e.dxx, (plus.dx-minus.dx)/(2*step), atol=2e-6)
    # A direct second difference provides an additional independent check.
    np.testing.assert_allclose(e.dxx, (plus.value-2*e.value+minus.value)/step**2, atol=2e-6)


@pytest.mark.parametrize("kind", ["pinned", "floating", "optimized"])
def test_parameter_and_mixed_jacobians_including_anchor(kind):
    model = QuantumModel(FeatureMap(2, "chebt"), HardwareEfficientAnsatz(2, 1, "YZ"),
                         Readout.weighted_z(2, n_outputs=2), Boundary(kind, 0, [0.7, -0.2]))
    p = model.initial_parameters(19)
    x, h = np.array([0.04, 0.41, 0.78]), 1e-6
    batch = model.prepare(x, 2)
    e = model.evaluate(p, batch, jacobian=True)
    for k in range(model.n_parameters):
        pp, pm = p.copy(), p.copy()
        pp[k] += h
        pm[k] -= h
        ep, em = model.evaluate(pp, batch), model.evaluate(pm, batch)
        for derivative in range(3):
            numerical = (ep.derivatives[derivative]-em.derivatives[derivative])/(2*h)
            np.testing.assert_allclose(e.jacobians[derivative][..., k], numerical,
                                       rtol=2e-6, atol=2e-7)
    if kind == "floating":
        anchor = model.predict(p, [0], 0, True)
        np.testing.assert_allclose(anchor.value, [[0.7, -0.2]], atol=1e-15)
        np.testing.assert_allclose(anchor.jacobians[0], 0, atol=2e-15)


def test_parallel_block_jacobian_and_shared_register():
    models = [QuantumModel(FeatureMap(1, "cheb"), HardwareEfficientAnsatz(1, 1, "Y"),
                           Readout.magnetization(1), Boundary("optimized", 0, [u]))
              for u in [0.5, 0.0]]
    model = ParallelModel(models)
    p = model.initial_parameters(2)
    x = np.array([0.1, 0.3])
    e = model.predict(p, x, 1, True)
    assert e.value.shape == (2, 2)
    np.testing.assert_allclose(e.jacobians[0][:, 0, 2:], 0)
    np.testing.assert_allclose(e.jacobians[0][:, 1, :2], 0)
    r, j = model.boundary_residual(p, True)
    assert r.shape == (2,) and j.shape == (2, 4)
    np.testing.assert_allclose(r, 0, atol=1e-15)


def test_ansatz_and_hamiltonians_preserve_norm():
    for ansatz in (HardwareEfficientAnsatz(5, 2), AlternatingBlocksAnsatz(5, 3, 4, 2)):
        initial = FeatureMap(5).prepare([0.2, 0.7], 0).states
        theta = np.random.default_rng(1).normal(size=ansatz.n_parameters)
        states = ansatz.apply(theta, initial)
        np.testing.assert_allclose(np.sum(abs(states)**2, axis=1), 1, atol=1e-13)
    h = ising_hamiltonian(2, -1, transverse=0.3, longitudinal=0.2)
    np.testing.assert_allclose(h, h.conj().T)
    np.testing.assert_allclose(spin_glass([[0, 0.4], [0.4, 0]], [0.1, 0.2]),
                               0.4*pauli_string("ZZ")+0.1*pauli_string("ZI")+0.2*pauli_string("IZ"))
    np.testing.assert_allclose(z_observables(2)[0], pauli_string("ZI"))


def test_invalid_domains_and_prepared_batch_ownership():
    fmap = FeatureMap(1)
    with pytest.raises(ValueError):
        fmap.prepare([1.], 1)
    fmap.prepare([1.], 0)  # Endpoint values themselves are allowed.
    with pytest.raises(ValueError):
        fmap.prepare([1.1], 0)
    model = QuantumModel(fmap, HardwareEfficientAnsatz(1, 0), Readout.magnetization(1))
    with pytest.raises(ValueError):
        model.evaluate(np.empty(0), FeatureMap(1).prepare([0.1]))


def test_noncommuting_readout_matches_direct_matrix_expectations():
    observables = np.stack([pauli_string("XI"), pauli_string("YZ"), pauli_string("ZZ")])
    readout = Readout(observables, [[0.2, 0.3, -0.5]])
    rng = np.random.default_rng(3)
    states = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    expected = np.array([[np.vdot(s, c @ s).real for c in observables] for s in states])
    np.testing.assert_allclose(readout.expectations(states), expected)
    diagonal = Readout(z_observables(2), [[1, 1]])
    expected = np.array([[np.vdot(s, c @ s).real for c in diagonal.matrices] for s in states])
    np.testing.assert_allclose(diagonal.expectations(states), expected)

