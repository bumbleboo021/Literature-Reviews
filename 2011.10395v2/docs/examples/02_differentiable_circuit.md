# Example 02: Differentiable Quantum Circuit `f(x)` and `df(x)/dx`

**Script:** [`examples/02_differentiable_circuit.py`](../../examples/02_differentiable_circuit.py)  
**Paper reference:** Section II (Eqs. 1–2, Fig. 2) and Section III.A (Fig. 3b) of `arXiv:2011.10395v2`.

## Purpose

This example shows how `DifferentiableQuantumCircuit` (`dqc/circuit.py`) wraps the feature map, ansatz, and cost observable into one object that returns both:

- **Function values**: $f_\theta(x) = \langle \emptyset | \hat{\mathcal{U}}_\varphi^\dagger(x)\hat{\mathcal{U}}_\theta^\dagger \hat{C} \hat{\mathcal{U}}_\theta \hat{\mathcal{U}}_\varphi(x) | \emptyset \rangle$
- **Exact analytical derivatives**: $df_\theta(x)/dx$, computed in two equivalent ways:
  1. **PyTorch autograd (`eval_with_derivative`)** — fast on a simulator, and uses `create_graph=True` so we can subsequently compute $\nabla_\theta \mathcal{L}[df/dx, f, x]$ for Adam.
  2. **Hardware parameter-shift rule (`df_dx_parameter_shift`)** — shifts each feature-map rotation angle $\phi_j(x) \to \phi_j(x) \pm \frac{\pi}{2}$ one qubit at a time and applies the chain rule:

     ```math
     \frac{df}{dx} = \frac{1}{2} \sum_{j=1}^{N} \frac{d\phi_j(x)}{dx} \Big(\langle \hat{C} \rangle_{\phi_j + \pi/2} - \langle \hat{C} \rangle_{\phi_j - \pi/2}\Big)
     ```

## How to Run

```bash
python 2011.10395v2/examples/02_differentiable_circuit.py
```

## What It Returns

1. A table comparing `f(x)` and `df/dx` across 5 grid points $x \in [0.0, 0.8]$ using autograd, parameter-shift, and central finite differences (`eps = 1e-6`).
2. `Max |df_auto - df_pshift|` on the order of `1e-15` (machine precision), proving the simulator's autograd matches the exact quantum hardware derivative rule.
3. A check that calling `.backward()` on a dummy ODE loss $\frac{1}{M}\sum_i (df/dx_i + f(x_i))^2$ populates `dqc.params.grad` with the exact shape `(depth, n_qubits, 3)` needed by Adam.
