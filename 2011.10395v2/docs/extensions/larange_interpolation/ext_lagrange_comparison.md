# Extension Benchmark: Baseline HEA vs. Controlled-Rotation (Lagrange GPSR) & 2nd-Order ODE

**Modules:** [`dqc/extensions/lagrange_gpsr.py`](../../dqc/extensions/lagrange_gpsr.py), [`dqc/extensions/cr_ansatz.py`](../../dqc/extensions/cr_ansatz.py), [`problems/second_order_ode.py`](../../problems/second_order_ode.py)  
**Script:** [`experiments/ext_lagrange_comparison.py`](../../experiments/ext_lagrange_comparison.py)

## What This Extension Tests

1. **A/B Benchmark on Fig. 6 ($\lambda = 20$)**:
   Compares our untouched baseline `DifferentiableQuantumCircuit` (fixed nearest-neighbor `CNOT`s) against `ControlledRotationDQC` (trainable `CRY` entanglers, whose $n = 3$ generator eigenvalues $\{-1/2, 0, +1/2\}$ are hardware-differentiable via the 4-term Lagrange parameter-shift rule).
2. **New Capability — 2nd-Order ODEs**:
   Solves the underdamped harmonic oscillator using exact second derivatives $d^2 f / dx^2$:

$$
\frac{d^2 u}{dx^2} + 2\gamma \frac{du}{dx} + \omega_0^2 u = 0, \qquad u(0) = 1, \quad u'(0) = 0
$$

with $\gamma = 1.0, \omega_0 = 6.0$ on $x \in [0, 0.9]$.

## How to Run

```bash
python 2011.10395v2/experiments/ext_lagrange_comparison.py
```

Saves `2011.10395v2/results/ext_lagrange_comparison.png`.
