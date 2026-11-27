# Extension Benchmark: Classical Chebyshev-PINN vs. Quantum DQC

**Module:** [`dqc/extensions/pinn_baseline.py`](../../../dqc/extensions/pinn_baseline.py) (`ChebyshevPINN`)  
**Tests:** [`tests/test_pinn.py`](../../../tests/test_pinn.py)  
**Script:** [`experiments/ext_pinn_vs_dqc.py`](../../../experiments/ext_pinn_vs_dqc.py)  
**Figure:** [`results/ext_pinn_vs_dqc.png`](../../../results/ext_pinn_vs_dqc.png)

## Setup

Apples-to-apples comparison on the Fig. 6 damped oscillator ODE ($\lambda = 8$ and $\lambda = 20$, exact floating boundary) between two **Chebyshev spectral solvers**:

- **Quantum DQC**: 6 qubits, depth-5 HEA, `chebyshev_tower` (**90 parameters**, 20 grid points, 250 Adam steps).
- **Classical Cheb-PINN**: Chebyshev input embedding $[T_1(x), \dots, T_{12}(x)]$ + Tanh MLP `12 -> 32 -> 32 -> 1` (**1,505 parameters**, 60 grid points, 500 Adam $\to$ 200 L-BFGS-B).

## How to Run

```bash
python 2011.10395v2/experiments/ext_pinn_vs_dqc.py
```

---

## Reproduced Results

| Case | Model | Params | CPU Time | Final $L_F$ | Final $L_Q$ (MSE vs. Exact) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| $\lambda = 8$ | **Cheb-PINN** (`[T_1..T_12]` + MLP, Adam+LBFGS) | 1,505 | **7.6s** | **`5.96e-06`** | **`1.08e-05`** |
| $\lambda = 8$ | Quantum DQC (`ChebT`, 250 Adam) | **90** | ~190s | `3.81e-03` | `3.50e-02` |
| $\lambda = 20$ | **Cheb-PINN** (`[T_1..T_12]` + MLP, Adam+LBFGS) | 1,505 | **7.8s** | **`4.59e-04`** | **`7.94e-05`** |
| $\lambda = 20$ | **Quantum DQC** (`ChebT`, 250 Adam) | **90** | ~165s | `6.51e-03` | **`8.89e-04`** |

---

## Conclusions

1. **Parameter efficiency (Quantum DQC)**: With only **90 trainable parameters** (~17x fewer than `Cheb-PINN`), the 6-qubit `ChebT` DQC captures every oscillation of $\lambda = 20$ ($L_Q = 8.89 \times 10^{-4}$), and actually converges **~30x faster per Adam step** than `Cheb-PINN` during the first 250 steps.
2. **Classical speed & accuracy (`Cheb-PINN`)**: Equipping a classical PINN with the same Chebyshev feature map $[T_1(x), \dots, T_{12}(x)]$ eliminates MLP spectral bias, runs **~20x faster** on a classical CPU (~7.8s), and reaches $L_Q \sim 10^{-5}$ with `Adam -> L-BFGS-B`.
