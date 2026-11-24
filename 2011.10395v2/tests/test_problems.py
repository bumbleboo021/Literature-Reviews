import numpy as np
import pytest
from dqc.model import Evaluation
from dqc.problems import NonlinearDecay, NonlinearOscillator, DampedOscillator, CoupledSystem, NozzleSteady
from dqc.problems.nozzle import area, log_area_derivative, transient_initial_conditions, transient_rhs


@pytest.mark.parametrize("problem", [NonlinearDecay(), NonlinearOscillator(), DampedOscillator(),
    DampedOscillator(residual_form="cosine_weighted"), CoupledSystem(), NozzleSteady(),
    NozzleSteady(residual_form="divided")])
def test_problem_partials_against_independent_differences(problem):
    x, h, m, o = np.array([0.03, 0.12, 0.23]), 1e-6, 3, problem.n_outputs
    rng = np.random.default_rng(123)
    values = rng.uniform(0.4, 1.0, (m, o))
    derivatives = rng.uniform(-0.3, 0.4, (m, o))
    if o == 3:
        values[:, 2] = 0.15  # Far from T = V².
    jet = [values, derivatives]
    partials = problem.partials(x, Evaluation(tuple(jet)))
    for order in range(2):
        for col in range(o):
            plus, minus = [a.copy() for a in jet], [a.copy() for a in jet]
            plus[order][:, col] += h
            minus[order][:, col] -= h
            fd = (problem.residual(x, Evaluation(tuple(plus)))
                  - problem.residual(x, Evaluation(tuple(minus))))/(2*h)
            np.testing.assert_allclose(partials[order][:, :, col], fd, atol=2e-8)


def test_reference_solutions_satisfy_their_equations():
    x, h = np.linspace(0.01, 0.8, 20), 1e-6
    for problem in (NonlinearDecay(), DampedOscillator(), CoupledSystem(), NonlinearOscillator()):
        y = problem.reference(x)
        dy = (problem.reference(x+h)-problem.reference(x-h))/(2*h)
        residual = problem.residual(x, Evaluation((y, dy)))
        np.testing.assert_allclose(residual, 0, atol=3e-7)


def test_nozzle_conservation_explicit_equivalence_and_geometry():
    x = np.array([0.1, 0.3, 0.7])
    y = np.array([[1., 1., 0.1], [0.9, 0.95, 0.2], [0.5, 0.7, 1.1]])
    problem = NozzleSteady()
    dy = problem.explicit_rhs(x, y)
    np.testing.assert_allclose(problem.residual(x, Evaluation((y, dy))), 0, atol=1e-13)
    np.testing.assert_allclose(transient_rhs(x, y, dy), 0, atol=1e-13)
    np.testing.assert_allclose(area([0, 0.5, 1]), [5.95, 1, 5.95])
    h = 1e-6
    np.testing.assert_allclose(log_area_derivative(x), (np.log(area(x+h))-np.log(area(x-h)))/(2*h))
    init = transient_initial_conditions(np.array([0., 0.5]))
    np.testing.assert_allclose(init[0], [1, 1, 0.1])
    with pytest.raises(ValueError, match="singular"):
        problem.explicit_rhs([0.5], [[1., 1., 1.]])


def test_damped_original_pole_is_not_silently_regularized():
    x = np.array([np.pi/(2*8)])
    jet = Evaluation((np.ones((1, 1)), np.zeros((1, 1))))
    with pytest.raises(ValueError, match="pole"):
        DampedOscillator().residual(x, jet)
    assert np.all(np.isfinite(DampedOscillator(residual_form="cosine_weighted").residual(x, jet)))

