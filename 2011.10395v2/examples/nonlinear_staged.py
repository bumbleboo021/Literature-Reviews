"""Fig. 7-style training: ChebT then fixed Ising evolution-enhanced ChebT.

Small CPU defaults, not identical six-qubit reproduction. Both stages save outputs.
"""

import argparse
from dataclasses import replace
from dqc.experiments import ExperimentConfig, make_experiment
from dqc.objective import Objective
from dqc.optimizers import fit_adam
from dqc.reporting import save_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100, help="Adam updates per stage")
    args = parser.parse_args()
    first = ExperimentConfig(problem="nonlinear", depth=3, points=100,
                              optimizer="adam", steps=args.steps, seed=11)
    model1, problem, x = make_experiment(first)
    fit1 = fit_adam(Objective(model1, problem, x), seed=first.seed,
                    steps=first.steps, learning_rate=first.learning_rate)
    save_run("outputs/nonlinear_staged/stage1", first, model1, problem, x, fit1)
    second = replace(first, evolution_time=2.0)
    model2, problem, x = make_experiment(second)
    fit2 = fit_adam(Objective(model2, problem, x), initial=fit1.parameters,
                    steps=second.steps, learning_rate=second.learning_rate)
    result = save_run("outputs/nonlinear_staged/stage2", second, model2, problem, x, fit2)
    print(result["dense_validation"])


if __name__ == "__main__":
    main()

