"""Classical optimizers. Gradients come from shifts + classical chain rules."""

from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import least_squares


@dataclass
class FitResult:
    parameters: np.ndarray
    history: list
    seconds: float
    message: str
    success: bool


def _initial(objective, initial, seed):
    p = objective.model.initial_parameters(seed) if initial is None else np.asarray(initial, float).copy()
    if p.shape != (objective.model.n_parameters,) or not np.all(np.isfinite(p)):
        raise ValueError("Invalid initial parameter vector.")
    return p


def fit_adam(objective, initial=None, seed=0, steps=250, learning_rate=0.01,
             beta1=0.9, beta2=0.999, epsilon=1e-8, callback=None):
    """Paper-style Adam. Returns FINAL iterate, including the last update.

    Scheduled objectives at different steps are not ranked as a 'best loss'.
    history records the objective at the current schedule step before each update,
    and one final evaluation. callback(step, terms) can report progress.
    """
    if not isinstance(steps, int) or steps < 1 or learning_rate <= 0 or epsilon <= 0:
        raise ValueError("Need positive integer steps, learning rate, and epsilon.")
    if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
        raise ValueError("Adam beta values must lie in [0, 1).")
    p = _initial(objective, initial, seed)
    m, v, history = np.zeros_like(p), np.zeros_like(p), []
    start = time.perf_counter()
    for step in range(steps):
        terms, gradient = objective.loss_and_gradient(p, step)
        history.append({"step": step, **terms})
        if callback:
            callback(step, terms)
        m = beta1*m+(1-beta1)*gradient
        v = beta2*v+(1-beta2)*gradient**2
        t = step+1
        p -= learning_rate*(m/(1-beta1**t))/(np.sqrt(v/(1-beta2**t))+epsilon)
    terms, _ = objective.loss_and_gradient(p, steps, gradient=False)
    history.append({"step": steps, **terms})
    return FitResult(p, history, time.perf_counter()-start,
                     "Fixed Adam iteration budget completed; inspect independent errors.", False)


def fit_least_squares(objective, initial=None, seed=0, max_nfev=180):
    """Convenience optimizer for noiseless CPU examples; paper uses Adam."""
    if not isinstance(max_nfev, int) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer.")
    p = _initial(objective, initial, seed)
    history = []

    def residual(params):
        r, _ = objective.least_squares_system(params)
        history.append({"evaluation": len(history), "total": float(np.dot(r, r))})
        return r

    def jac(params):
        return objective.least_squares_system(params, jacobian=True)[1]

    start = time.perf_counter()
    result = least_squares(residual, p, jac=jac, max_nfev=max_nfev,
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    return FitResult(result.x, history, time.perf_counter()-start, result.message, bool(result.success))

