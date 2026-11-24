import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from dqc.experiments import ExperimentConfig, make_experiment, make_model
from dqc.objective import Objective, Regularization
from dqc.optimizers import fit_adam, fit_least_squares
from dqc.schedules import Constant, LinearDecay, TanhDecay
from dqc.losses import kl_divergence, js_divergence, mse, mae
from dqc.reporting import evaluate_metrics
from dqc.cli import main
from dqc.model import Evaluation
from dqc.problems.base import Problem


@pytest.mark.parametrize("boundary", ["floating", "pinned", "optimized"])
@pytest.mark.parametrize("distance", ["mse", "mae"])
def test_full_loss_gradient_including_regularization_and_boundaries(boundary, distance):
    config = ExperimentConfig(n_qubits=2, depth=1, boundary=boundary)
    model, problem, _ = make_experiment(config)
    x = np.array([0.07, 0.3, 0.6])
    reg = Regularization([0.2, 0.7], [0.9, 0.5], LinearDecay(10))
    objective = Objective(model, problem, x, boundary_weight=1.7, regularization=reg, distance=distance)
    p = model.initial_parameters(12)+0.04
    terms, gradient = objective.loss_and_gradient(p, step=2)
    assert np.isfinite(terms["total"])
    fd = np.empty_like(p)
    for k in range(len(p)):
        pp, pm = p.copy(), p.copy()
        pp[k] += 1e-6
        pm[k] -= 1e-6
        fp = objective.loss_and_gradient(pp, step=2, gradient=False)[0]["total"]
        fm = objective.loss_and_gradient(pm, step=2, gradient=False)[0]["total"]
        fd[k] = (fp-fm)/2e-6
    np.testing.assert_allclose(gradient, fd, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("encoding", ["shared", "parallel"])
def test_coupled_objective_cross_output_gradient(encoding):
    config = ExperimentConfig(problem="coupled", encoding=encoding, n_qubits=2, depth=1)
    model, problem, _ = make_experiment(config)
    objective = Objective(model, problem, [0.03, 0.31, 0.74])
    p = model.initial_parameters(2)
    _, gradient = objective.loss_and_gradient(p)
    fd = np.zeros_like(p)
    for k in range(len(p)):
        pp, pm = p.copy(), p.copy()
        pp[k] += 1e-6
        pm[k] -= 1e-6
        fd[k] = (objective.loss_and_gradient(pp, gradient=False)[0]["total"]
                 - objective.loss_and_gradient(pm, gradient=False)[0]["total"])/2e-6
    np.testing.assert_allclose(gradient, fd, atol=2e-6, rtol=2e-6)


def test_least_squares_vector_matches_objective_and_gradient():
    model, problem, _ = make_experiment(ExperimentConfig(n_qubits=2, depth=1, boundary="optimized"))
    objective = Objective(model, problem, [0.1, 0.3, 0.7], boundary_weight=3,
                          regularization=Regularization([0.2], [0.7], Constant(0.4)))
    p = model.initial_parameters(4)
    terms, gradient = objective.loss_and_gradient(p)
    r, j = objective.least_squares_system(p, True)
    np.testing.assert_allclose(terms["total"], r @ r)
    np.testing.assert_allclose(gradient, 2*r @ j)
    fading = Objective(model, problem, [0.1, 0.3],
                        regularization=Regularization([0.2], [0.7], LinearDecay(20)))
    with pytest.raises(ValueError, match="Adam"):
        fading.least_squares_system(p)


def test_schedules_and_probability_distances():
    assert Constant(0.5)(100) == 0.5
    np.testing.assert_allclose([LinearDecay(10)(i) for i in [0, 5, 20]], [1, 0.5, 0])
    schedule = TanhDecay(600, 150)
    assert schedule(0) > schedule(150) > schedule(600)
    assert schedule(150) == 1
    assert mse([1, 2], [0, 0]) == 2.5 and mae([1, 2]) == 1.5
    assert kl_divergence([0, 1], [0.5, 0.5]) == pytest.approx(np.log(2))
    assert np.isinf(kl_divergence([1, 0], [0, 1]))
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(np.log(2))
    with pytest.raises(ValueError):
        kl_divergence([-1, 2], [0.5, 0.5])
    with pytest.raises(ValueError):
        js_divergence([1, 1], [0.5, 0.5])


def test_decay_end_to_end_generalizes_between_collocation_points():
    model, problem, x = make_experiment(ExperimentConfig())
    objective = Objective(model, problem, x)
    fit = fit_least_squares(objective, seed=7, max_nfev=140)
    metrics, _, _, _ = evaluate_metrics(model, fit.parameters, problem, np.linspace(0, 0.9, 201))
    assert metrics["solution_rmse_per_output"][0] < 0.002
    assert metrics["residual_mse_sum_equations"] < 1e-4


def test_adam_reduces_loss_and_records_final_iterate():
    model, problem, x = make_experiment(ExperimentConfig(n_qubits=2, depth=1, points=12))
    objective = Objective(model, problem, x)
    p = model.initial_parameters(5)
    before = objective.loss_and_gradient(p, gradient=False)[0]["total"]
    fit = fit_adam(objective, initial=p, steps=60, learning_rate=0.02)
    after = objective.loss_and_gradient(fit.parameters, step=60, gradient=False)[0]["total"]
    assert after < before
    assert fit.history[-1]["total"] == pytest.approx(after)
    assert len(fit.history) == 61


def test_cli_saved_parameters_round_trip(tmp_path):
    main(["--problem", "decay", "--optimizer", "adam", "--steps", "2", "--qubits", "1",
          "--depth", "1", "--points", "5", "--no-plot", "--output", str(tmp_path)])
    run = json.loads((tmp_path/"run.json").read_text())
    config = ExperimentConfig(**run["config"])
    model, problem, _ = make_experiment(config)
    p = np.load(tmp_path/"parameters.npz")["parameters"]
    data = np.loadtxt(tmp_path/"predictions.csv", delimiter=",", skiprows=1)
    np.testing.assert_allclose(model.predict(p, data[:, 0], 0).value[:, 0], data[:, 1], atol=1e-14)


def test_second_order_ode_objective_gradient():
    class Harmonic(Problem):
        order = 2

        def residual(self, x, e):
            return e.dxx+3*e.value

        def partials(self, x, e):
            return (np.full((len(x), 1, 1), 3.), np.zeros((len(x), 1, 1)), np.ones((len(x), 1, 1)))

    model, _, _ = make_experiment(ExperimentConfig(n_qubits=2, depth=1))
    objective = Objective(model, Harmonic(), [0.04, 0.37, 0.71])
    p = model.initial_parameters(3)
    _, gradient = objective.loss_and_gradient(p)
    fd = np.zeros_like(p)
    for k in range(len(p)):
        pp, pm = p.copy(), p.copy()
        pp[k] += 1e-6
        pm[k] -= 1e-6
        fd[k] = (objective.loss_and_gradient(pp, gradient=False)[0]["total"]
                 - objective.loss_and_gradient(pm, gradient=False)[0]["total"])/2e-6
    np.testing.assert_allclose(gradient, fd, atol=3e-5, rtol=2e-6)


def test_nozzle_staged_script_plumbing_with_synthetic_guidance(tmp_path):
    # Synthetic labels exercise I/O and stages only; never used as a physical reference.
    guidance = tmp_path/"synthetic_guidance.csv"
    np.savetxt(guidance, np.column_stack([np.linspace(0.61, 0.89, 5),
                                         np.ones(5), np.ones(5), np.full(5, 0.2)]),
               delimiter=",", header="x,density,temperature,velocity", comments="")
    script = Path(__file__).resolve().parents[1]/"examples/nozzle_staged.py"
    result = subprocess.run([sys.executable, str(script), "--steps", "1", "--guidance", str(guidance)],
                             cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
    folder = tmp_path/"outputs/nozzle_staged/stage2"
    assert (folder/"regularization.csv").exists()
    assert (folder/"schedule.json").exists()
    record = json.loads((folder/"run.json").read_text())
    assert record["config"]["stop"] == 0.9
    assert np.isfinite(record["dense_validation"]["residual_mse_sum_equations"])
