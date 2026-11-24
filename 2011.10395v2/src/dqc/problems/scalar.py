"""Scalar benchmarks: paper Eq. (30), Eq. (32), plus a teaching decay ODE."""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from .base import Problem


@dataclass
class NonlinearDecay(Problem):
    """Teaching example u' + u² = 0; not a numbered paper experiment."""

    u0: float = 1.0

    @property
    def initial_values(self):
        return (self.u0,)

    def residual(self, x, e):
        return e.dx + e.value**2

    def partials(self, x, e):
        return (2*e.value[:, :, None], np.ones((len(x), 1, 1)))

    def reference(self, x):
        x = np.asarray(x)
        return (self.u0/(1+self.u0*x))[:, None]


@dataclass
class DampedOscillator(Problem):
    """Eq. (30). Original and explicitly cosine-weighted residuals are separate."""

    frequency: float = 8.0
    damping: float = 0.1
    u0: float = 1.0
    residual_form: str = "original"

    def __post_init__(self):
        if self.residual_form not in ("original", "cosine_weighted"):
            raise ValueError("residual_form must be original or cosine_weighted.")
        if not np.all(np.isfinite([self.frequency, self.damping, self.u0])):
            raise ValueError("ODE parameters must be finite.")

    @property
    def initial_values(self):
        return (self.u0,)

    def _coefficients(self, x):
        phase = self.frequency*np.asarray(x)
        if self.residual_form == "cosine_weighted":
            return (self.frequency*(self.damping*np.cos(phase)+np.sin(phase)), np.cos(phase))
        if np.any(abs(np.cos(phase)) < 1e-10):
            raise ValueError("Eq. (30) is undefined at a tan pole; change the grid or residual form explicitly.")
        return self.frequency*(self.damping+np.tan(phase)), np.ones_like(phase)

    def residual(self, x, e):
        a, b = self._coefficients(x)
        return b[:, None]*e.dx+a[:, None]*e.value

    def partials(self, x, e):
        a, b = self._coefficients(x)
        return a[:, None, None], b[:, None, None]

    def reference(self, x):
        # Correct multiplicative integration constant, unlike printed Eq. (31).
        x = np.asarray(x)
        return (self.u0*np.exp(-self.frequency*self.damping*x)*np.cos(self.frequency*x))[:, None]


@dataclass
class NonlinearOscillator(Problem):
    """Eq. (32): u' - 4u + 6u² - sin(50x) - u cos(25x) + 1/2 = 0."""

    u0: float = 0.75

    @property
    def initial_values(self):
        return (self.u0,)

    def residual(self, x, e):
        x = np.asarray(x)[:, None]
        u = e.value
        return e.dx-4*u+6*u**2-np.sin(50*x)-u*np.cos(25*x)+0.5

    def partials(self, x, e):
        a = -4+12*e.value-np.cos(25*np.asarray(x))[:, None]
        return a[:, :, None], np.ones((len(x), 1, 1))

    def reference(self, x):
        x = np.asarray(x, float)
        if np.any(x < 0):
            raise ValueError("Reference integration is implemented for x >= 0.")
        if np.max(x) == 0:
            return np.full((len(x), 1), self.u0)

        def rhs(t, u):
            return 4*u-6*u**2+np.sin(50*t)+u*np.cos(25*t)-0.5

        solution = solve_ivp(rhs, (0, float(np.max(x))), [self.u0], method="DOP853",
                             rtol=1e-11, atol=1e-13, dense_output=True)
        if not solution.success:
            raise RuntimeError(solution.message)
        return solution.sol(x).T

