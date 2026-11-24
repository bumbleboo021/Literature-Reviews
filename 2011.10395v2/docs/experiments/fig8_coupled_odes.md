# Experiment: Reproducing Fig. 8 (Strongly Coupled ODEs & Boundary Handling)

**Problem:** [`problems/coupled_odes.py`](../../problems/coupled_odes.py)  
**Script:** [`experiments/reproduce_fig8.py`](../../experiments/reproduce_fig8.py)  
**Paper reference:** Section IV, Eqs. (33)–(36), Fig. 8 (pp. 14–15) of `arXiv:2011.10395v2`.

## Equations

$$
\begin{aligned}
\frac{du_1}{dx} - \lambda_1 u_2 - \lambda_2 u_1 &= 0, \qquad u_1(0) = 0.5 \\
\frac{du_2}{dx} + \lambda_2 u_2 + \lambda_1 u_1 &= 0, \qquad u_2(0) = 0.0
\end{aligned}
$$

with $\lambda_1 = 5$, $\lambda_2 = 3$ ($\omega = \sqrt{\lambda_1^2 - \lambda_2^2} = 4$).

## Setup

- Two parallel 6-qubit, depth-5 circuits (`chebyshev_tower`), trained jointly with Adam (`lr = 0.02`, 250 iterations, 20 points on $[0, 0.9]$).
- Compares **Pinned** (Eq. 24), **Floating** (Eqs. 25–26), and **Optimized** (Eq. 27) boundary handling.

## How to Run

```bash
python 2011.10395v2/experiments/reproduce_fig8.py
```

Saves `2011.10395v2/results/fig8_coupled_odes.png`.
