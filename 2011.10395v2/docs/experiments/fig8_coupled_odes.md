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

Saves [`results/fig8_coupled_odes.png`](../../results/fig8_coupled_odes.png).

---

## Reproduced Results & Conclusions

| Boundary Handler | Formula | Final $L_F$ | Final $L_Q$ (MSE vs. Exact) | Gain vs. Pinned |
| :--- | :--- | :---: | :---: | :---: |
| **Pinned** (`Pin`, Eq. 24) | $f(x) = \langle \hat{C}\rangle(x)$ + boundary loss | `2.95e-02` | `6.18e-02` | 1x |
| **Optimized** (`Optim`, Eq. 27) | $f(x) = f_c + \langle \hat{C}\rangle(x)$ + boundary loss | `2.07e-02` | `3.64e-02` | 1.7x |
| **Floating** (`Float`, Eqs. 25–26) | $f(x) = u_0 + \langle \hat{C}\rangle(x) - \langle \hat{C}\rangle(x_0)$ | **`1.58e-03`** | **`2.00e-04`** | **309x better $L_Q$!** |

**What Fig. 8 brings to us:**
- Demonstrates **parallel multi-circuit encoding** (two independent 6-qubit registers $\boldsymbol{\theta}_1, \boldsymbol{\theta}_2$ trained jointly) for **coupled systems of ODEs**.
- Proves that **Floating boundary handling** dramatically outperforms Pinned and Optimized (>180–300x lower MSE) because enforcing $f(x_0) = u_0$ analytically removes the competing boundary penalty from the loss function.
