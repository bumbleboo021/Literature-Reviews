"""Upstream training, then OPTIONAL full split-domain training with supplied guidance.

Usage: python examples/nozzle_staged.py --steps 100
       python examples/nozzle_staged.py --guidance path/to/points.csv

CSV header: x,density,temperature,velocity ; exactly 5 downstream rows in [0.6,0.9].
The paper does not tabulate its 5 downstream labels. We do not invent those values.
"""

import argparse
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from dqc.experiments import ExperimentConfig, make_experiment, make_model
from dqc.grids import nozzle_stage_grid
from dqc.objective import Objective, Regularization
from dqc.optimizers import fit_adam
from dqc.reporting import save_run
from dqc.schedules import TanhDecay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100, help="Upstream updates; stage 2 uses 3x this.")
    parser.add_argument("--guidance", type=Path)
    args = parser.parse_args()
    guidance_x = guidance_values = None
    if args.guidance is not None:
        data = np.atleast_1d(np.genfromtxt(args.guidance, delimiter=",", names=True))
        expected = ("x", "density", "temperature", "velocity")
        if data.dtype.names != expected or len(data) != 5:
            parser.error("Guidance CSV must have header x,density,temperature,velocity and exactly five rows.")
        guidance_x = np.asarray(data["x"])
        guidance_values = np.column_stack([data[key] for key in expected[1:]])
        if (not np.all(np.isfinite(guidance_values)) or not np.all(np.isfinite(guidance_x))
                or np.any(guidance_x < 0.6) or np.any(guidance_x > 0.9)
                or np.any(guidance_values <= 0) or len(np.unique(guidance_x)) != 5):
            parser.error("Guidance must contain 5 unique finite positions in [0.6,0.9] and positive fields.")
    config = ExperimentConfig(problem="nozzle", optimizer="adam", points=20,
                              steps=args.steps, learning_rate=0.01)
    model, problem, x1 = make_experiment(config)
    first = fit_adam(Objective(model, problem, x1), seed=config.seed,
                     steps=config.steps, learning_rate=config.learning_rate)
    save_run("outputs/nozzle_staged/stage1", config, model, problem, x1, first)
    if args.guidance is None:
        print("Upstream run saved. Supply --guidance with five downstream labels to run stage 2.")
        return
    # Keep the feature coordinates unchanged between stages (identity mapping).
    second_config = replace(config, stop=0.9, steps=3*args.steps, learning_rate=0.005)
    second_model = make_model(second_config, problem)
    upstream_x = nozzle_stage_grid(1)
    upstream_values = model.predict(first.parameters, upstream_x, order=0).value
    regularization = Regularization(
        np.r_[upstream_x, guidance_x], np.vstack([upstream_values, guidance_values]),
        TanhDecay(second_config.steps, drop_step=second_config.steps/4))
    x2 = nozzle_stage_grid(2)
    objective = Objective(second_model, problem, x2, regularization=regularization)
    second = fit_adam(objective, initial=first.parameters, steps=second_config.steps,
                      learning_rate=second_config.learning_rate)
    output = Path("outputs/nozzle_staged/stage2")
    save_run(output, second_config, second_model, problem, x2, second)
    # Save ALL actual regularization values and the schedule, not only the model seed.
    np.savetxt(output/"regularization.csv", np.column_stack([regularization.x, regularization.values]),
               delimiter=",", header="x,density,temperature,velocity", comments="")
    (output/"schedule.json").write_text(json.dumps({"kind": "tanh", "total_steps": second_config.steps,
                                                   "drop_step": second_config.steps/4, "width": 0.05}), encoding="utf-8")
    print("Stage 2 saved, including residuals inside the excluded throat region for inspection.")


if __name__ == "__main__":
    main()

