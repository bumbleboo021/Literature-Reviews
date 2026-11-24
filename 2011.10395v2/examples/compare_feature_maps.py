"""Compare encodings under the same small CPU budget: python examples/compare_feature_maps.py."""

import argparse
from dataclasses import replace
from dqc.experiments import ExperimentConfig, make_experiment
from dqc.objective import Objective
from dqc.optimizers import fit_adam
from dqc.reporting import save_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    base = ExperimentConfig(problem="damped", optimizer="adam", steps=args.steps, points=20)
    for kind in ("prod", "cheb", "chebt"):
        config = replace(base, feature_map=kind)
        model, problem, x = make_experiment(config)
        fit = fit_adam(Objective(model, problem, x), seed=config.seed, steps=config.steps,
                       learning_rate=config.learning_rate)
        result = save_run(f"outputs/feature_maps/{kind}", config, model, problem, x, fit)
        print(kind, result["dense_validation"])


if __name__ == "__main__":
    main()

