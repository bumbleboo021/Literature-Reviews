"""DE, boundary and scheduled regularization losses, Eqs. (19)-(29)."""

from dataclasses import dataclass, field
import numpy as np
from .schedules import Constant


@dataclass
class Regularization:
    x: np.ndarray
    values: np.ndarray
    schedule: object = field(default_factory=Constant)

    def __post_init__(self):
        self.x = np.atleast_1d(np.asarray(self.x, float)).copy()
        self.values = np.asarray(self.values, float).copy()
        if self.values.ndim == 1:
            self.values = self.values[:, None]
        if (self.x.ndim != 1 or not len(self.x) or self.values.ndim != 2
                or len(self.x) != len(self.values) or not np.all(np.isfinite(self.x))
                or not np.all(np.isfinite(self.values))):
            raise ValueError("Regularization needs finite x:(R,) and values:(R, outputs).")


class Objective:
    def __init__(self, model, problem, x, boundary_weight=1.0, regularization=None, distance="mse"):
        self.model, self.problem = model, problem
        if model.n_outputs != problem.n_outputs:
            raise ValueError("Model output count differs from problem output count.")
        self.x = np.atleast_1d(np.asarray(x, float)).copy()
        if self.x.ndim != 1 or not len(self.x) or not np.all(np.isfinite(self.x)):
            raise ValueError("Collocation points must be a nonempty finite 1-D array.")
        if not np.isfinite(boundary_weight) or boundary_weight < 0:
            raise ValueError("boundary_weight must be finite and nonnegative.")
        if distance not in ("mse", "mae"):
            raise ValueError("Residual objectives support mse or mae. KL/JS need probabilities.")
        self.distance, self.boundary_weight = distance, boundary_weight
        self.batch = model.prepare(self.x, problem.order)
        self.regularization = regularization
        self.reg_batch = None
        if regularization is not None:
            if regularization.values.shape[1] != model.n_outputs:
                raise ValueError("Regularization output count differs from model.")
            self.reg_batch = model.prepare(regularization.x, 0)

    def _blocks(self, params, step, jacobian):
        e = self.model.evaluate(params, self.batch, jacobian)
        residual = self.problem.residual(self.x, e)
        if residual.shape != (len(self.x), self.problem.n_outputs):
            raise ValueError("Problem residual has incorrect shape.")
        j = self.problem.residual_jacobian(self.x, e) if jacobian else None
        blocks = [("differential", residual.ravel(),
                   j.reshape(-1, self.model.n_parameters) if jacobian else None, 1/len(self.x))]
        bc, bj = self.model.boundary_residual(params, jacobian)
        if len(bc) and self.boundary_weight:
            blocks.append(("boundary", bc, bj, self.boundary_weight))
        if self.regularization is not None:
            weight = float(self.regularization.schedule(step))
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("Regularization schedule must produce nonnegative finite weights.")
            if weight:
                e = self.model.evaluate(params, self.reg_batch, jacobian)
                r = (e.value-self.regularization.values).ravel()
                j = e.jacobians[0].reshape(-1, self.model.n_parameters) if jacobian else None
                blocks.append(("regularization", r, j, weight))
        if any(not np.all(np.isfinite(r)) for _, r, _, _ in blocks):
            raise FloatingPointError("Nonfinite residuals; inspect inputs and physical singularities.")
        return blocks

    def loss_and_gradient(self, params, step=0, gradient=True):
        terms = {"differential": 0., "boundary": 0., "regularization": 0.}
        grad = np.zeros(self.model.n_parameters) if gradient else None
        for name, residual, jac, weight in self._blocks(params, step, gradient):
            if self.distance == "mse":
                terms[name] = float(weight*np.dot(residual, residual))
                if gradient:
                    grad += 2*weight*(residual @ jac)
            else:
                terms[name] = float(weight*np.sum(abs(residual)))
                if gradient:
                    grad += weight*(np.sign(residual) @ jac)
        terms["total"] = sum(terms.values())
        if gradient and not np.all(np.isfinite(grad)):
            raise FloatingPointError("Nonfinite objective gradient.")
        return terms, grad

    def least_squares_system(self, params, jacobian=False):
        if self.distance != "mse":
            raise ValueError("Least-squares requires MSE.")
        # One optimizer run has a fixed objective; scheduled fading is for Adam.
        if self.regularization is not None and not isinstance(self.regularization.schedule, Constant):
            raise ValueError("Use Adam for time-varying regularization schedules.")
        blocks = self._blocks(params, 0, jacobian)
        residual = np.concatenate([np.sqrt(w)*r for _, r, _, w in blocks])
        jac = np.concatenate([np.sqrt(w)*j for _, _, j, w in blocks]) if jacobian else None
        return residual, jac

