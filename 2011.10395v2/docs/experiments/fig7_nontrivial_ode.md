# Experiment: Reproducing Fig. 7 (Highly Nontrivial Nonlinear ODE)

**Problem:** [`problems/nontrivial_ode.py`](../../problems/nontrivial_ode.py)  
**Script:** [`experiments/reproduce_fig7.py`](../../experiments/reproduce_fig7.py)  
**Paper reference:** Section IV, Eq. (32), Fig. 7 (p. 13) of `arXiv:2011.10395v2`.

## Equation

$$\frac{du}{dx} - 4u + 6u^2 - \sin(50x) - u\cos(25x) + \frac{1}{2} = 0, \qquad u(0) = 0.75$$

## Two-Stage Training Workflow

1. **Stage 1 (`ChebT`)**: Train a 6-qubit, depth-5 DQC with the Chebyshev Tower feature map on 100 grid points in $[0, 0.9]$ for 200 Adam iterations (`lr = 0.01`).
2. **Stage 2 (`EvEn`)**: Transfer the Stage-1 variational angles into an **Evolution-Enhanced** Chebyshev Tower DQC — appending $\exp(-i \hat{H} \tau)$ ($\tau = 2.0$, nearest-neighbor Ising Hamiltonian $\hat{H} = -J \sum_j \hat{Z}_j \hat{Z}_{j+1} + h \sum_j \hat{X}_j$) — and train for 200 more iterations.

## How to Run

```bash
python 2011.10395v2/experiments/reproduce_fig7.py
```

Saves [`results/fig7_nontrivial_ode.png`](../../results/fig7_nontrivial_ode.png).

---

## Reproduced Results & Conclusions

| Stage | Feature Map | Iterations | Final $L_F$ | Final $L_Q$ (MSE vs. DOP853) |
| :--- | :--- | :---: | :---: | :---: |
| Stage 1 | `chebyshev_tower` (`ChebT`) | 200 | `2.03e-01` | **`6.72e-05`** |
| Stage 2 | Evolution-Enhanced (`EvEn`, $\tau = 2.0$) | 200 (warm-started) | `2.61e-01` | **`8.44e-05`** |

**What Fig. 7 brings to us:**
- Demonstrates solving a **nonlinear, non-periodic, high-frequency ODE** ($6u^2$, $\sin(50x)$, $u\cos(25x)$) where no closed-form solution exists, using a denser grid (100 points) and `ChebT` / `EvEn`.
- Introduces **two-stage warm-start training** (train easy-to-optimize `ChebT` first, then transfer angles $\boldsymbol{\theta}$ into the entangled `EvEn` circuit $e^{-i\hat{H}\tau}$) to balance expressivity vs. trainability.
