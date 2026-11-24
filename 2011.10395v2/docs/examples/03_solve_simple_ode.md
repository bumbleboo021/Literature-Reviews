# Example 03: End-to-End Solution of a Simple ODE

**Script:** [`examples/03_solve_simple_ode.py`](../../examples/03_solve_simple_ode.py)  
**Paper reference:** Section III.D–III.H and Fig. 1 of `arXiv:2011.10395v2`.

## Purpose

This example wires together the complete DQC workflow (Fig. 1) on a clean toy ODE before we tackle the paper's benchmark equations in Section IV:

$$\frac{du}{dx} + 2u = 0, \qquad u(0) = 1 \qquad \Longrightarrow \qquad u(x) = e^{-2x}$$

It demonstrates:
1. **`BoundaryHandler(mode="floating", x0=0.0, u0=1.0)`** (`dqc/boundary.py`): shifts the circuit output by $f_b = u_0 - \langle \hat{C} \rangle(x_0)$ (Eqs. 25–26) so the initial condition is satisfied *exactly* without needing a competing boundary loss term.
2. **`DQCSolver`** (`dqc/solver.py`): runs Adam over 100 iterations on 15 grid points in $[0, 0.8]$, logging both the **Full Loss** $L_F$ and **Quality of Solution** $L_Q = \frac{1}{M}\sum_i (f(x_i) - u(x_i))^2$.
3. **Interpolation (`result.predict(x_test)`)**: evaluates the trained quantum circuit at arbitrary points $x$ (including points not in the training grid).

## How to Run

```bash
python 2011.10395v2/examples/03_solve_simple_ode.py
```

## What It Returns

- Training log every 20 iterations showing $L_F$ and $L_Q$ dropping rapidly toward $10^{-5}$–$10^{-6}$.
- A table comparing the DQC prediction $f(x)$ against the analytical solution $u(x) = e^{-2x}$ across $x \in [0.0, 0.8]$, with `|error| = 0.00e+00` at $x = 0$ (exact floating boundary) and $\sim 10^{-3}$ or better everywhere else.
