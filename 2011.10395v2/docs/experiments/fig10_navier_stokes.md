# Experiment: Reproducing Fig. 10 (Quasi-1D Navier–Stokes Nozzle)

**Problem:** [`problems/navier_stokes.py`](../../problems/navier_stokes.py)  
**Script:** [`experiments/reproduce_fig10.py`](../../experiments/reproduce_fig10.py)  
**Paper reference:** Section IV, Eqs. (40)–(50), Fig. 10 (pp. 15–17) of `arXiv:2011.10395v2`.

## Setup

- Nozzle area $A(x) = 1 + 4.95(2x - 1)^2$ with $\gamma = 1.4$.
- Three parallel 6-qubit, depth-6 circuits for density $\rho(x)$, temperature $T(x)$, and velocity $V(x)$, using the Chebyshev feature map and `floating` boundary handling.
- **Stage 1**: 20 points in subsonic region $[0, 0.4]$, 200 Adam steps (`lr = 0.01`), no regularization.
- **Stage 2**: 40 points in $[0, 0.4] \cup [0.6, 0.9]$ (excluding the sonic throat $(0.4, 0.6)$), 600 Adam steps (`lr = 0.005`), with reverse-sigmoid regularization switching off around $n_{\text{drop}} = 150$.

## How to Run

```bash
python 2011.10395v2/experiments/reproduce_fig10.py
```

Saves `2011.10395v2/results/fig10_navier_stokes.png`.
