"""Product, Cheb, ChebT and fixed evolution-enhanced encodings (Eqs. 6, 11-15).

Input derivatives use gate-wise shifts and the FULL second-order chain rule.
The exponential is a fixed x-independent mixing unitary, not an ODE step.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class AffineDomain:
    """z = scale*x + offset. Derivatives returned by FeatureMap are w.r.t. x."""

    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.scale) or self.scale == 0 or not np.isfinite(self.offset):
            raise ValueError("Domain scale must be finite/nonzero and offset finite.")

    @classmethod
    def interval(cls, start, stop, interior=0.8):
        if not np.isfinite(start+stop) or stop <= start or not 0 < interior < 1:
            raise ValueError("Need start < stop and 0 < interior < 1.")
        scale = 2*interior/(stop-start)
        return cls(scale, -interior-scale*start)

    def encode(self, x):
        return self.scale*np.asarray(x, float) + self.offset


@dataclass
class FeatureBatch:
    x: np.ndarray
    states: np.ndarray
    coefficients: np.ndarray
    owner: object = field(repr=False)

    @property
    def order(self):
        return len(self.coefficients)-1


def product_states(angles):
    states = np.ones((len(angles), 1), dtype=complex)
    for j in range(angles.shape[1]):
        one = np.stack([np.cos(angles[:, j]/2), np.sin(angles[:, j]/2)], axis=1)
        states = (states[:, :, None]*one[:, None, :]).reshape(len(angles), -1)
    return states


class FeatureMap:
    def __init__(self, n_qubits, kind="chebt", domain=None, hamiltonian=None, time=0.0):
        if not isinstance(n_qubits, int) or n_qubits < 1:
            raise ValueError("n_qubits must be a positive integer.")
        self.n_qubits = n_qubits
        self.kind = kind.lower()
        if self.kind not in ("prod", "cheb", "chebt"):
            raise ValueError("Feature map must be prod, cheb, or chebt.")
        self.domain = AffineDomain() if domain is None else domain
        self.unitary = None
        if not np.isfinite(time):
            raise ValueError("Evolution time must be finite.")
        if hamiltonian is None and time != 0:
            raise ValueError("Nonzero evolution time requires a Hamiltonian.")
        if hamiltonian is not None:
            h = np.asarray(hamiltonian, complex)
            if h.shape != (2**n_qubits, 2**n_qubits) or not np.allclose(h, h.conj().T):
                raise ValueError("Evolution Hamiltonian must be Hermitian with matching dimension.")
            if not np.all(np.isfinite(h)):
                raise ValueError("Hamiltonian must be finite.")
            self.unitary = expm(-1j*h*time)

    def angles(self, x, order=2):
        x = np.atleast_1d(np.asarray(x, float))
        if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
            raise ValueError("Inputs must be a nonempty finite 1-D array.")
        z = self.domain.encode(x)
        if np.any(abs(z) > 1) or (order > 0 and np.any(abs(z) >= 1)):
            raise ValueError("Encoded inputs need |z| <= 1; derivatives require |z| < 1.")
        scale = self.domain.scale
        if self.kind == "prod":
            phi = np.arcsin(z)[:, None]*np.ones((1, self.n_qubits))
            if order == 0:
                return phi, None, None
            first = np.ones_like(phi)*scale/np.sqrt(1-z[:, None]**2)
            second = np.ones_like(phi)*scale**2*z[:, None]/(1-z[:, None]**2)**1.5
        else:
            ns = np.arange(1, self.n_qubits+1) if self.kind == "chebt" else np.ones(self.n_qubits)
            phi = 2*np.arccos(z)[:, None]*ns
            if order == 0:
                return phi, None, None
            first = -2*ns*scale/np.sqrt(1-z[:, None]**2)
            second = -2*ns*scale**2*z[:, None]/(1-z[:, None]**2)**1.5
        return phi, first, second

    def prepare(self, x, order=1):
        if order not in (0, 1, 2):
            raise ValueError("Derivative order must be 0, 1, or 2.")
        x = np.atleast_1d(np.asarray(x, float)).copy()
        phi, dphi, ddphi = self.angles(x, order)
        n, m = self.n_qubits, len(x)
        shifts = [np.zeros(n)]
        coefficients = [np.vstack([np.ones(m)] + [np.zeros(m)]*order)]

        def add(shift, derivative, weight):
            shifts.append(shift)
            c = np.zeros((order+1, m))
            c[derivative] = weight
            coefficients.append(c)

        if order >= 1:
            for j in range(n):
                for sign in (1, -1):
                    shift = np.zeros(n)
                    shift[j] = sign*np.pi/2
                    add(shift, 1, sign*dphi[:, j]/2)
                    if order == 2:
                        coefficients[-1][2] = sign*ddphi[:, j]/2
        if order == 2:
            # Diagonal angle Hessian: [q(phi+pi)-2q(phi)+q(phi-pi)]/4.
            for j in range(n):
                coefficients[0][2] -= dphi[:, j]**2/2
                for sign in (1, -1):
                    shift = np.zeros(n)
                    shift[j] = sign*np.pi
                    add(shift, 2, dphi[:, j]**2/4)
            # Cross Hessian counted twice in sum_(i,j) phi_i' phi_j' q_(i,j).
            for i in range(n):
                for j in range(i+1, n):
                    for si in (1, -1):
                        for sj in (1, -1):
                            shift = np.zeros(n)
                            shift[i], shift[j] = si*np.pi/2, sj*np.pi/2
                            add(shift, 2, si*sj*dphi[:, i]*dphi[:, j]/2)
        angles = phi[None, :, :] + np.asarray(shifts)[:, None, :]
        states = product_states(angles.reshape(-1, n))
        if self.unitary is not None:
            states = states @ self.unitary.T
        return FeatureBatch(x, states, np.stack(coefficients, axis=1), self)

