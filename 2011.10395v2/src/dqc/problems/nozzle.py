"""Quasi-1D inviscid compressible nozzle: Eqs. (40)-(50).

Includes transient PDE RHS evaluation, steady conservation residuals, and the
explicit divided form. This module is not a time-dependent PDE solver.
"""

from dataclasses import dataclass
import numpy as np
from .base import Problem


def area(x):
    return 1+4.95*(2*np.asarray(x)-1)**2


def log_area_derivative(x):
    return 19.8*(2*np.asarray(x)-1)/area(x)


def transient_initial_conditions(x):
    """Eqs. (45)-(47), in density/temperature/velocity column order."""
    x = np.atleast_1d(np.asarray(x, float))
    rho, temperature = 1-0.944*x, 1-0.694*x
    if np.any(temperature <= 0):
        raise ValueError("Initial temperature must be positive.")
    return np.column_stack([rho, temperature, (0.1+3.27*x)*np.sqrt(temperature)])


def transient_rhs(x, values, spatial_derivatives, gamma=1.4):
    """Time derivatives from Eqs. (40)-(42), given values and x-derivatives."""
    values, spatial_derivatives = np.asarray(values), np.asarray(spatial_derivatives)
    rho, temp, vel = values.T
    dr, dt, dv = spatial_derivatives.T
    if np.any(abs(rho) < 1e-12):
        raise ValueError("Density is too close to zero.")
    a = log_area_derivative(x)
    return np.column_stack([-rho*dv-rho*vel*a-vel*dr,
                            -vel*dt-(gamma-1)*temp*(dv+vel*a),
                            -vel*dv-(dt+temp*dr/rho)/gamma])


@dataclass
class NozzleSteady(Problem):
    """Default conservation residual avoids the explicit sonic denominator.

    Choosing conservation vs divided changes optimization weighting, even where
    the physical roots agree. Negative outputs are NOT automatically prevented.
    """

    gamma: float = 1.4
    residual_form: str = "conservation"
    n_outputs = 3
    domain = (0.0, 0.4)
    initial_values = (1.0, 1.0, 0.1)
    labels = ("density", "temperature", "velocity")

    def __post_init__(self):
        if self.gamma <= 1 or not np.isfinite(self.gamma):
            raise ValueError("gamma must be finite and greater than one.")
        if self.residual_form not in ("conservation", "divided"):
            raise ValueError("Nozzle residual must be conservation or divided.")

    def explicit_rhs(self, x, y):
        rho, temp, vel = np.asarray(y).T
        a, d = log_area_derivative(x), temp-vel**2
        if np.any(abs(d) < 1e-10):
            raise ValueError("Explicit steady equations are singular near T = V².")
        return np.column_stack([rho*vel**2*a/d,
                                temp*vel**2*(self.gamma-1)*a/d, -temp*vel*a/d])

    def residual(self, x, e):
        if self.residual_form == "divided":
            return e.dx-self.explicit_rhs(x, e.value)
        return -transient_rhs(x, e.value, e.dx, self.gamma)

    def partials(self, x, e):
        rho, temp, vel = e.value.T
        dr, dt, dv = e.dx.T
        a, c, m = log_area_derivative(x), self.gamma-1, len(x)
        dy, dd = np.zeros((m, 3, 3)), np.zeros((m, 3, 3))
        if self.residual_form == "divided":
            self.explicit_rhs(x, e.value)  # singularity validation
            d = temp-vel**2
            # dy is minus the Jacobian of the RHS in Eqs. (48)-(50).
            dy[:, 0, 0] = -vel**2*a/d
            dy[:, 0, 1] = rho*vel**2*a/d**2
            dy[:, 0, 2] = -2*rho*a*vel*temp/d**2
            dy[:, 1, 1] = c*vel**4*a/d**2
            dy[:, 1, 2] = -2*c*temp**2*vel*a/d**2
            dy[:, 2, 1] = -vel**3*a/d**2
            dy[:, 2, 2] = temp*a*(temp+vel**2)/d**2
            dd[:] = np.eye(3)
            return dy, dd
        if np.any(abs(rho) < 1e-12):
            raise ValueError("Density is too close to zero.")
        dy[:, 0, 0], dy[:, 0, 2] = dv+vel*a, rho*a+dr
        dy[:, 1, 1], dy[:, 1, 2] = c*(dv+vel*a), dt+c*temp*a
        dy[:, 2, 0] = -temp*dr/(self.gamma*rho**2)
        dy[:, 2, 1], dy[:, 2, 2] = dr/(self.gamma*rho), dv
        dd[:, 0, 0], dd[:, 0, 2] = vel, rho
        dd[:, 1, 1], dd[:, 1, 2] = vel, c*temp
        dd[:, 2, 0], dd[:, 2, 1], dd[:, 2, 2] = temp/(self.gamma*rho), 1/self.gamma, vel
        return dy, dd

