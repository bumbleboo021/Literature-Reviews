"""Lagrange Interpolation General Parameter-Shift Rule (GPSR).

Implements Algorithm 1, Eqs. (4), (8), (10)–(20), and Section 4.1 (Eqs. 24–25) from:
    Vu Tuan Hai & Le Bin Ho,
    "Lagrange interpolation approach for parameter-shift rule of general quantum gates".

Given any parameterized gate U(x) = exp(-i * x * G) whose Hermitian generator G
has `n` distinct eigenvalues, this module computes exact real shift coefficients
{c_k} for shifts {alpha_k} such that:

    dC(x)/dx = sum_{k=1..m} c_k * [C(x + alpha_k) - C(x - alpha_k)]

Works for:
  - n = 2 (single-qubit Pauli rotations -> 2-term rule)
  - n = 3 (controlled rotations CRX/CRY/CRZ -> 4-term rule)
  - n >= 4 (multi-qubit / global rotations -> 2m-term rule)
  - 2nd-order derivatives d^2 C / (dx_i dx_j) via recursive application (Sec. 4.1)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ShiftRule:
    """Computed general parameter-shift rule: dC/dx = sum_k c_k [C(x + a_k) - C(x - a_k)]."""

    shifts: tuple[float, ...]
    coeffs: tuple[float, ...]
    eigenvalues: tuple[float, ...]

    @property
    def n_terms(self) -> int:
        """Total number of shifted circuit evaluations (2m)."""
        return 2 * len(self.shifts)


def lagrange_polynomial_coeffs(eigenvalues: Sequence[float]) -> np.ndarray:
    """Return matrix P of shape (n, n) where P[k, r] is the G^r coefficient of L_k(G).

    From Eq. (4):
        L_k(G) = prod_{l != k} (G - lam_l I) / (lam_k - lam_l) = sum_{r=0..n-1} P[k, r] * G^r
    """
    lams = np.asarray(eigenvalues, dtype=np.float64)
    n = len(lams)
    poly_matrix = np.zeros((n, n), dtype=np.float64)

    for k in range(n):
        others = np.delete(lams, k)
        # np.poly returns coefficients from highest degree to lowest; reverse to get 0..n-1
        num_coeffs = np.poly(others)[::-1]
        denom = float(np.prod(lams[k] - others))
        poly_matrix[k, :] = num_coeffs / denom

    return poly_matrix


def delta_m_upper_imag(
    eigenvalues: Sequence[float], alpha: float, poly_matrix: np.ndarray | None = None
) -> np.ndarray:
    """Compute Im(Delta M(alpha)) for all upper-triangular pairs (i < j) — Eqs. (15)–(17).

    Because Lambda_r(-alpha) = conj(Lambda_r(alpha)), Delta M(alpha) = M(alpha) - M(-alpha)
    is purely imaginary and skew-symmetric with zero diagonal.
    """
    lams = np.asarray(eigenvalues, dtype=np.float64)
    n = len(lams)
    if poly_matrix is None:
        poly_matrix = lagrange_polynomial_coeffs(lams)

    # Lambda_r(alpha) = sum_k exp(-i * alpha * lam_k) * P[k, r]
    phases = np.exp(-1j * alpha * lams)
    lam_vec = phases @ poly_matrix  # shape (n,)

    m_pos = np.outer(np.conj(lam_vec), lam_vec)
    delta_m = m_pos - np.conj(m_pos)  # purely imaginary: 2i * Im(m_pos)

    iu, ju = np.triu_indices(n, k=1)
    return delta_m[iu, ju].imag


def derive_lagrange_shift_rule(
    eigenvalues: Sequence[float],
    shifts: Sequence[float] | None = None,
    tol: float = 1e-10,
) -> ShiftRule:
    """Derive exact general parameter-shift coefficients via Lagrange interpolation (Algorithm 1).

    Args:
        eigenvalues: Distinct eigenvalues of the Hermitian generator G in U(x) = exp(-i * x * G).
            (Example: PennyLane's RY(x) = exp(-i * x * Y/2) has eigenvalues [-0.5, 0.5];
            CRY(x) = exp(-i * x * |1><1| x Y/2) has eigenvalues [-0.5, 0.0, 0.5].)
        shifts: Optional custom shift angles {alpha_k}. If None, picks well-conditioned shifts.
        tol: Numerical tolerance for verifying the linear system solution.
    """
    unique_lams = np.unique(np.round(np.asarray(eigenvalues, dtype=np.float64), decimals=12))
    n = len(unique_lams)
    if n < 2:
        raise ValueError("Generator must have at least 2 distinct eigenvalues.")

    poly_matrix = lagrange_polynomial_coeffs(unique_lams)

    # Determine the effective rank (number of independent non-zero rows in Im(Delta M))
    probe_angles = np.linspace(0.17, 2.71, max(20, 2 * n * n))
    probe_mat = np.column_stack([delta_m_upper_imag(unique_lams, a, poly_matrix) for a in probe_angles])
    nonzero_mask = np.max(np.abs(probe_mat), axis=1) > 1e-12
    # Row 0 corresponds to (i=0, j=1), i.e., the [B, G] commutator term!
    if not nonzero_mask[0]:
        raise RuntimeError("Unexpected: (0, 1) entry of Delta M is identically zero.")

    rank = int(np.linalg.matrix_rank(probe_mat[nonzero_mask], tol=1e-10))

    if shifts is None:
        # Choose `rank` well-spaced shifts in (0, pi) scaled by the spectral gap
        max_gap = float(unique_lams[-1] - unique_lams[0])
        if n == 2:
            chosen_shifts = [math.pi / (2.0 * max_gap)]
        else:
            # Equidistant Chebyshev-like shifts in (0, pi / max_gap) avoid sin(k*alpha) = 0
            chosen_shifts = [
                (2 * k - 1) * math.pi / (2.0 * rank * max_gap) for k in range(1, rank + 1)
            ]
    else:
        chosen_shifts = [float(s) for s in shifts]
        if len(chosen_shifts) < rank:
            raise ValueError(
                f"Generator with {n} distinct eigenvalues needs at least {rank} shifts, got {len(chosen_shifts)}."
            )

    # Build linear system A @ c = e_0, where e_0 = [-1, 0, ..., 0]^T
    # Because Delta M_{01} = i * Im(Delta M_{01}), requiring sum_k c_k * Delta M_{01}(a_k) = -i
    # is equivalent to sum_k c_k * Im(Delta M_{01}(a_k)) = -1.
    full_cols = np.column_stack([delta_m_upper_imag(unique_lams, a, poly_matrix) for a in chosen_shifts])
    a_mat = full_cols[nonzero_mask]
    rhs = np.zeros(a_mat.shape[0], dtype=np.float64)
    rhs[0] = -1.0

    coeffs, _, _, _ = np.linalg.lstsq(a_mat, rhs, rcond=None)
    residual = float(np.linalg.norm(a_mat @ coeffs - rhs))
    if residual > tol:
        raise ValueError(
            f"Shift angles {chosen_shifts} are ill-conditioned for eigenvalues {unique_lams.tolist()} "
            f"(residual={residual:.2e}). Try different shifts."
        )

    return ShiftRule(
        shifts=tuple(chosen_shifts),
        coeffs=tuple(float(c) for c in coeffs),
        eigenvalues=tuple(float(l) for l in unique_lams),
    )


def apply_shift_rule(eval_fn: Callable[[float], float], x: float, rule: ShiftRule) -> float:
    """Evaluate exact 1st-order derivative d(eval_fn)/dx at x using `rule` (Eq. 20)."""
    return sum(
        c * (eval_fn(x + a) - eval_fn(x - a)) for a, c in zip(rule.shifts, rule.coeffs)
    )


def apply_second_order_shift_rule(
    eval_fn: Callable[[float, float], float],
    x1: float,
    x2: float,
    rule1: ShiftRule,
    rule2: ShiftRule | None = None,
) -> float:
    """Evaluate exact 2nd-order derivative d^2(eval_fn)/(dx1 dx2) — Section 4.1, Eq. (24)."""
    r2 = rule1 if rule2 is None else rule2
    total = 0.0
    for a1, c1 in zip(rule1.shifts, rule1.coeffs):
        for a2, c2 in zip(r2.shifts, r2.coeffs):
            total += c1 * c2 * (
                eval_fn(x1 + a1, x2 + a2)
                - eval_fn(x1 + a1, x2 - a2)
                - eval_fn(x1 - a1, x2 + a2)
                + eval_fn(x1 - a1, x2 - a2)
            )
    return total
