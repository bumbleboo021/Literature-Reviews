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

Saves `2011.10395v2/results/fig7_nontrivial_ode.png`.
