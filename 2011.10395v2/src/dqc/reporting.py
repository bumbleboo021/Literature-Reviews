"""Independent dense-grid metrics and portable output files."""

from dataclasses import asdict
from pathlib import Path
import json
import platform
import numpy as np
import scipy


def evaluate_metrics(model, parameters, problem, x):
    e = model.predict(parameters, x, order=problem.order)
    residual = problem.residual(x, e)
    reference = problem.reference(x)
    metrics = {
        "residual_mse_sum_equations": float(np.mean(np.sum(residual**2, axis=1))),
        "residual_max_abs": float(np.max(abs(residual))),
    }
    if reference is not None:
        metrics["solution_rmse_per_output"] = np.sqrt(np.mean((e.value-reference)**2, axis=0)).tolist()
        metrics["solution_max_abs_per_output"] = np.max(abs(e.value-reference), axis=0).tolist()
    return metrics, e, residual, reference


def save_run(output, config, model, problem, x_train, fit, plot=True):
    """Save exact config + parameters; rebuild model from config to restore a run."""
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    xv = np.linspace(float(np.min(x_train)), float(np.max(x_train)), 401)
    metrics, e, residual, reference = evaluate_metrics(model, fit.parameters, problem, xv)
    training_metrics, _, _, _ = evaluate_metrics(model, fit.parameters, problem, x_train)
    metadata = {
        "config": asdict(config), "n_parameters": model.n_parameters,
        "optimizer_message": fit.message, "optimizer_converged": fit.success,
        "training_seconds": fit.seconds, "dense_validation": metrics,
        "training_grid": training_metrics,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "implementation": "Noiseless classical statevector simulation; not a hardware experiment.",
    }
    if config.problem == "nozzle":
        metadata["physical_diagnostics"] = {
            "density_min": float(e.value[:, 0].min()), "temperature_min": float(e.value[:, 1].min()),
            "all_density_temperature_positive": bool(np.all(e.value[:, :2] > 0)),
        }
    (path/"run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    (path/"history.json").write_text(json.dumps(fit.history, indent=2, allow_nan=False), encoding="utf-8")
    np.savez(path/"parameters.npz", parameters=fit.parameters, x_train=x_train)
    data = [xv[:, None], e.value, residual]
    columns = ["x"] + [f"prediction_{s}" for s in problem.labels] + [f"residual_{s}" for s in problem.labels]
    if reference is not None:
        data.append(reference)
        columns += [f"reference_{s}" for s in problem.labels]
    np.savetxt(path/"predictions.csv", np.column_stack(data), delimiter=",",
               header=",".join(columns), comments="")
    if plot:
        # Select backend before pyplot import: CLI never opens GUI windows.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.7))
        for j, label in enumerate(problem.labels):
            line, = axes[0].plot(xv, e.value[:, j], label=f"DQC {label}")
            if reference is not None:
                axes[0].plot(xv, reference[:, j], "--", color=line.get_color(), alpha=0.65)
            axes[2].plot(xv, residual[:, j], label=label)
        axes[0].set(title="Solution (dashed: reference if available)", xlabel="x", ylabel="value")
        axes[0].legend(fontsize=8)
        history = np.array([r["total"] for r in fit.history])
        axes[1].semilogy(np.maximum(history, 1e-20))
        axes[1].set(title="Recorded objective", xlabel="iteration / residual evaluation", ylabel="loss")
        axes[2].set(title="Residual on independent dense grid", xlabel="x", ylabel="residual")
        axes[2].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(path/"solution.png", dpi=150)
        plt.close(fig)
    return metadata

