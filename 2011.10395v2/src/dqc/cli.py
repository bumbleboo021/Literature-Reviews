"""python -m dqc / dqc-solve entry point."""

import argparse
import json
from pathlib import Path
from .experiments import ExperimentConfig, make_experiment
from .objective import Objective
from .optimizers import fit_adam, fit_least_squares
from .reporting import save_run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a noiseless CPU DQC solver.")
    parser.add_argument("--config", type=Path, help="JSON ExperimentConfig; CLI options override it.")
    parser.add_argument("--problem", choices=["decay", "damped", "nonlinear", "coupled", "nozzle"])
    parser.add_argument("--optimizer", choices=["adam", "least_squares"])
    parser.add_argument("--qubits", dest="n_qubits", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--points", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--max-nfev", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--feature-map", choices=["prod", "cheb", "chebt"])
    parser.add_argument("--boundary", choices=["pinned", "floating", "optimized"])
    parser.add_argument("--evolution-time", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    data = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    for key, value in vars(args).items():
        if key not in ("config", "output", "no_plot") and value is not None:
            data[key] = value
    try:
        config = ExperimentConfig(**data)
        model, problem, grid = make_experiment(config)
        objective = Objective(model, problem, grid, config.boundary_weight, distance=config.distance)
        print(f"{config.problem}: {config.n_qubits} qubits/register, {model.n_parameters} parameters; "
              f"{len(grid)} collocation points, {config.optimizer}", flush=True)
        if config.optimizer == "adam":
            def progress(step, terms):
                if step % max(1, config.steps//10) == 0:
                    print(f"step {step:4d}  loss={terms['total']:.6g}", flush=True)
            fit = fit_adam(objective, seed=config.seed, steps=config.steps,
                           learning_rate=config.learning_rate, callback=progress)
        else:
            fit = fit_least_squares(objective, seed=config.seed, max_nfev=config.max_nfev)
        output = args.output or Path("outputs")/config.problem
        result = save_run(output, config, model, problem, grid, fit, plot=not args.no_plot)
    except (ValueError, TypeError, FloatingPointError) as exc:
        parser.error(str(exc))
    print(fit.message)
    print(json.dumps(result["dense_validation"], indent=2))
    print(f"Saved to {output.resolve()}")

