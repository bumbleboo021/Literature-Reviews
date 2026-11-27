# Experiment: Reproducing Fig. 6 (Single Damped Oscillator ODE)

**Problem:** [`problems/damped_oscillator.py`](../../problems/damped_oscillator.py)  
**Script:** [`experiments/reproduce_fig6.py`](../../experiments/reproduce_fig6.py)  
**Paper reference:** Section IV, Eqs. (30)–(31), Fig. 6(a, b, d, e) of `arXiv:2011.10395v2`.

## Equation

$$\frac{du}{dx} + \lambda u\big(\kappa + \tan(\lambda x)\big) = 0, \qquad u(0) = 1$$

with analytical solution:

$$u(x) = e^{-\kappa \lambda x}\cos(\lambda x)$$

**Numerical note (pole-free form):** Because $\tan(\lambda x)$ has poles wherever $\cos(\lambda x) = 0$, evaluating the raw residual on an equidistant grid creates huge MSE spikes ($\sim 10^3$–$10^5$) that pin the trial function to $0$. Multiplying Eq. (30) by $\cos(\lambda x)$ gives the algebraically equivalent, pole-free residual:

$$
\cos(\lambda x)\frac{du}{dx} + \lambda u\big(\kappa\cos(\lambda x) + \sin(\lambda x)\big) = 0
$$

which has the exact same solution and matches the paper's initial loss scale ($10^1$–$10^2$) and convergence in Fig. 6.

## Setup (matching p. 12 of the paper)

- **Qubits / Ansatz**: $N = 6$ qubits, Hardware-Efficient Ansatz with depth $d = 5$ (90 trainable angles).
- **Cost operator**: Total magnetization $\hat{C} = \sum_{j=1}^6 \hat{Z}_j$.
- **Boundary handling**: `floating` ($f(0) = 1$ guaranteed exactly).
- **Training grid**: 20 equidistant points on $x \in [0, 0.9]$.
- **Optimizer**: Adam (`lr = 0.02`, `n_iter = 250`).
- **Comparison**: `product` (Prod) vs. `chebyshev` (Cheb) vs. `chebyshev_tower` (ChebT) for $\lambda = 8$ and $\lambda = 20$.

## How to Run

```bash
python 2011.10395v2/experiments/reproduce_fig6.py
```

## Expected Behavior

- For **$\lambda = 8$**: Both `Cheb` and `ChebT` converge closely to the exact damped cosine, while `Prod` plateaus early because its basis is too weak.
- For **$\lambda = 20$** (rapidly oscillating): Both `Prod` and sparse `Cheb` struggle to fit the oscillations, while **`ChebT`** (Chebyshev Tower) still captures the solution accurately — demonstrating why feature-map expressivity is crucial.
- Output plot is saved to [`results/fig6_feature_maps.png`](../../results/fig6_feature_maps.png).

---

## Reproduced Results & Conclusions

| Case | Feature Map | Final $L_F$ | Final $L_Q$ (MSE vs. Exact) | Outcome |
| :--- | :--- | :---: | :---: | :--- |
| $\lambda = 8$ | `product` (Prod) | `1.45e+00` | `9.85e-02` | Plateaus early — basis too weak |
| $\lambda = 8$ | `chebyshev` (Cheb) | `2.96e-02` | **`3.03e-04`** | Converges accurately |
| $\lambda = 8$ | `chebyshev_tower` (ChebT) | `3.81e-03` | `3.50e-02` | Slower early convergence (as noted on p. 13) |
| $\lambda = 20$ | `product` (Prod) | `1.54e+01` | `1.77e-01` | Fails to oscillate |
| $\lambda = 20$ | `chebyshev` (Cheb) | `7.82e+00` | `1.16e-01` | Fails to oscillate |
| $\lambda = 20$ | `chebyshev_tower` (ChebT) | **`6.51e-03`** | **`8.89e-04`** | **~200x lower $L_Q$; captures every oscillation** |

**What Fig. 6 brings to us:**
- Proves that **feature-map expressivity is the primary bottleneck** for oscillatory ODEs — swapping `product`/`chebyshev` for `chebyshev_tower` improves accuracy by ~200x at $\lambda = 20$ with the exact same ansatz depth ($d = 5$).
- Practical numerical lesson: clearing algebraic poles ($\tan(\lambda x) \to$ multiply by $\cos(\lambda x)$) is critical for stable MSE residuals.
