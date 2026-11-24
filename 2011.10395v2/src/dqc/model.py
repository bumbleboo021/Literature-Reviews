"""Expectation-valued functions, analytic input/parameter derivatives, and systems."""

from dataclasses import dataclass
import numpy as np
from .boundaries import Boundary


@dataclass
class Evaluation:
    """derivatives[k] has shape (M, outputs); jacobians[k] -> (M, outputs, P)."""

    derivatives: tuple
    jacobians: tuple | None = None

    @property
    def value(self):
        return self.derivatives[0]

    @property
    def dx(self):
        return self.derivatives[1]

    @property
    def dxx(self):
        return self.derivatives[2]


class QuantumModel:
    """One register, one or more outputs. All derivatives use exact shifts."""

    def __init__(self, feature_map, ansatz, readout, boundary=None):
        if feature_map.n_qubits != ansatz.n_qubits:
            raise ValueError("Feature map and ansatz qubit counts differ.")
        if readout.matrices.shape[1] != 2**feature_map.n_qubits:
            raise ValueError("Readout dimension does not match register.")
        self.feature_map, self.ansatz, self.readout = feature_map, ansatz, readout
        self.n_outputs = readout.n_outputs
        self.boundary = Boundary(values=np.ones(self.n_outputs)) if boundary is None else boundary
        if len(self.boundary.values) != self.n_outputs:
            raise ValueError("One boundary value is needed per output.")
        self.n_angles = ansatz.n_parameters
        self.n_weights = readout.weights.size if readout.trainable else 0
        self.n_offsets = self.n_outputs if self.boundary.kind == "optimized" else 0
        self.n_parameters = self.n_angles+self.n_weights+self.n_offsets
        self._anchor = feature_map.prepare([self.boundary.x0], order=0)

    def initial_parameters(self, seed=0):
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-0.8, 0.8, self.n_angles)
        weights = self.readout.weights.ravel() if self.n_weights else np.empty(0)
        p = np.r_[theta, weights, np.zeros(self.n_offsets)]
        if self.n_offsets:
            q0 = self._raw(p, self._anchor)[0][0][0]
            p[-self.n_offsets:] = self.boundary.values-q0
        return p

    def prepare(self, x, order=1):
        return self.feature_map.prepare(x, order)

    def _validate(self, params, batch):
        p = np.asarray(params, float)
        if p.shape != (self.n_parameters,) or not np.all(np.isfinite(p)):
            raise ValueError("Invalid model parameter shape or nonfinite parameters.")
        if batch.owner is not self.feature_map:
            raise ValueError("Prepared inputs belong to a different feature map.")
        return p

    def _raw(self, params, batch):
        p = self._validate(params, batch)
        states = self.ansatz.apply(p[:self.n_angles], batch.states)
        measured = self.readout.expectations(states)
        measured = measured.reshape(batch.coefficients.shape[1], len(batch.x), -1)
        features = np.einsum("dbm,bmk->dmk", batch.coefficients, measured)
        weights = (p[self.n_angles:self.n_angles+self.n_weights].reshape(self.readout.weights.shape)
                   if self.n_weights else self.readout.weights)
        outputs = np.einsum("dmk,ok->dmo", features, weights)
        return outputs, features

    def evaluate(self, params, batch, jacobian=False):
        p = self._validate(params, batch)
        outputs, features = self._raw(p, batch)
        anchor_features = None
        if self.boundary.kind == "floating":
            anchor, anchor_features = self._raw(p, self._anchor)
            outputs[0] += self.boundary.values-anchor[0, 0]
        elif self.n_offsets:
            outputs[0] += p[-self.n_offsets:]
        if not jacobian:
            return Evaluation(tuple(outputs))
        jac = np.zeros(outputs.shape + (self.n_parameters,))
        # Shift circuit outputs, not the nonlinear loss. Anchor dependence is included.
        for k in range(self.n_angles):
            plus, minus = p.copy(), p.copy()
            plus[k] += np.pi/2
            minus[k] -= np.pi/2
            ep = np.asarray(self.evaluate(plus, batch).derivatives)
            em = np.asarray(self.evaluate(minus, batch).derivatives)
            jac[..., k] = (ep-em)/2
        if self.n_weights:
            fs = features.copy()
            if anchor_features is not None:
                fs[0] -= anchor_features[0, 0]
            k = len(self.readout.matrices)
            for o in range(self.n_outputs):
                jac[:, :, o, self.n_angles+o*k:self.n_angles+(o+1)*k] = fs
        if self.n_offsets:
            jac[0, :, :, -self.n_offsets:] = np.eye(self.n_outputs)[None]
        return Evaluation(tuple(outputs), tuple(jac))

    def predict(self, params, x, order=1, jacobian=False):
        return self.evaluate(params, self.prepare(x, order), jacobian)

    def boundary_residual(self, params, jacobian=False):
        if self.boundary.kind == "floating":
            return np.empty(0), np.empty((0, self.n_parameters)) if jacobian else None
        e = self.evaluate(params, self._anchor, jacobian)
        residual = (e.value-self.boundary.values).ravel()
        jac = e.jacobians[0].reshape(-1, self.n_parameters) if jacobian else None
        return residual, jac


class ParallelModel:
    """Separate registers/parameters with one jointly optimized vector of outputs."""

    def __init__(self, models):
        self.models = tuple(models)
        if not self.models:
            raise ValueError("At least one model is required.")
        self.n_parameters = sum(m.n_parameters for m in self.models)
        self.n_outputs = sum(m.n_outputs for m in self.models)
        self._cuts = np.cumsum([0]+[m.n_parameters for m in self.models])

    def initial_parameters(self, seed=0):
        return np.concatenate([m.initial_parameters(seed+i) for i, m in enumerate(self.models)])

    def prepare(self, x, order=1):
        return tuple(m.prepare(x, order) for m in self.models)

    def evaluate(self, params, batches, jacobian=False):
        p = np.asarray(params, float)
        if p.shape != (self.n_parameters,) or len(batches) != len(self.models):
            raise ValueError("Invalid parallel model parameters/prepared inputs.")
        results = [m.evaluate(p[a:b], batch, jacobian)
                   for m, batch, a, b in zip(self.models, batches, self._cuts[:-1], self._cuts[1:])]
        derivatives = tuple(np.concatenate([e.derivatives[k] for e in results], axis=1)
                            for k in range(len(results[0].derivatives)))
        if not jacobian:
            return Evaluation(derivatives)
        jac = [np.zeros(v.shape+(self.n_parameters,)) for v in derivatives]
        output_start = 0
        for m, e, a, b in zip(self.models, results, self._cuts[:-1], self._cuts[1:]):
            for k in range(len(jac)):
                jac[k][:, output_start:output_start+m.n_outputs, a:b] = e.jacobians[k]
            output_start += m.n_outputs
        return Evaluation(derivatives, tuple(jac))

    def predict(self, params, x, order=1, jacobian=False):
        return self.evaluate(params, self.prepare(x, order), jacobian)

    def boundary_residual(self, params, jacobian=False):
        p = np.asarray(params, float)
        pieces, rows = [], []
        for m, a, b in zip(self.models, self._cuts[:-1], self._cuts[1:]):
            r, j = m.boundary_residual(p[a:b], jacobian)
            pieces.append(r)
            if jacobian:
                full = np.zeros((len(r), self.n_parameters))
                full[:, a:b] = j
                rows.append(full)
        return np.concatenate(pieces), np.concatenate(rows) if jacobian else None
