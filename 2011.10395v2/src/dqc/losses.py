"""Distance functions, Sec. III D. KL/JS require probability distributions."""

import numpy as np
from scipy.special import rel_entr


def mse(a, b=0.0):
    return float(np.mean((np.asarray(a)-np.asarray(b))**2))


def mae(a, b=0.0):
    return float(np.mean(abs(np.asarray(a)-np.asarray(b))))


def _probabilities(p):
    p = np.asarray(p, float)
    if (p.ndim != 1 or not len(p) or not np.all(np.isfinite(p))
            or np.any(p < 0) or not np.isclose(p.sum(), 1, atol=1e-10, rtol=1e-10)):
        raise ValueError("KL/JS inputs must be normalized, nonnegative finite probability vectors.")
    return p


def kl_divergence(p, q):
    p, q = _probabilities(p), _probabilities(q)
    if p.shape != q.shape:
        raise ValueError("Probability vectors must have equal shape.")
    return float(np.sum(rel_entr(p, q)))


def js_divergence(p, q):
    p, q = _probabilities(p), _probabilities(q)
    if p.shape != q.shape:
        raise ValueError("Probability vectors must have equal shape.")
    middle = (p+q)/2
    return (kl_divergence(p, middle)+kl_divergence(q, middle))/2

