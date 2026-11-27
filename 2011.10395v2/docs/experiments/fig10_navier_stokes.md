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

Saves [`results/fig10_navier_stokes.png`](../../results/fig10_navier_stokes.png).

---

## Reproduced Results & Conclusions

| Stage | Training Domain | Iterations | Final $L_F$ | Final $L_Q$ (MSE vs. Exact Isentropic) |
| :--- | :--- | :---: | :---: | :---: |
| Stage 1 (Subsonic) | $[0.0, 0.4]$ (20 pts) | 200 (`lr=0.01`) | `7.13e-04` | **`2.35e-04`** |
| Stage 2 (Full Nozzle) | $[0.0, 0.4] \cup [0.6, 0.9]$ (40 pts) | 600 (`lr=0.005`) | `6.53e-03` | **`3.00e-04`** |

**What Fig. 10 brings to us:**
- Scales DQCs to a **stiff, nonlinear, 3-variable fluid dynamics PDE system** ($\rho(x), T(x), V(x)$) whose stationary RHS diverges at the sonic throat ($T = V^2$ at $x = 0.5$), breaking standard classical ODE solvers like Mathematica's `NDSolve`.
- Shows how **domain exclusion** (omitting $(0.4, 0.6)$ around the singularity) combined with **staged warm-starting** and a **reverse-sigmoid regularization schedule** $\zeta(n_j)$ (Eq. 29, dropping at $n_{\text{drop}} = 150$) lets a global spectral DQC bridge seamlessly across the sonic throat onto the physical supersonic branch.
