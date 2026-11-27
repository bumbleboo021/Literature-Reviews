# Paper Experiments Overview (`arXiv:2011.10395v2`, Section IV)

Each benchmark figure in Section IV isolates and proves a distinct building block of the Differentiable Quantum Circuit (DQC) framework:

| Paper Figure | Problem & Difficulty | New Concept Tested | Best Reproduced $L_Q$ | Main Takeaway |
| :--- | :--- | :--- | :---: | :--- |
| **[Fig. 6](./fig6_damped_oscillator.md)** | Single linear ODE with rapid damping/oscillation ($\lambda = 8, 20$) | **Feature-map expressivity** (`Prod` vs `Cheb` vs `ChebT`) | `8.89e-04` ($\lambda=20$) | Feature map choice dominates accuracy (~200x gain for `ChebT` over `Prod`/`Cheb`). |
| **[Fig. 7](./fig7_nontrivial_ode.md)** | Nonlinear, non-periodic, high-frequency ODE ($6u^2, \sin(50x)$) | **Evolution-enhanced map** ($e^{-i\hat{H}\tau}$) + 2-stage warm-start | `6.72e-05` (`ChebT`) / `8.44e-05` (`EvEn`) | Warm-starting from `ChebT` $\to$ `EvEn` captures intricate nonlinear dynamics without getting stuck early. |
| **[Fig. 8](./fig8_coupled_odes.md)** | Strongly coupled 2-ODE system ($\lambda_1 = 5, \lambda_2 = 3$) | **Multi-circuit encoding** + **Boundary handling** (`Pin`, `Float`, `Optim`) | `2.00e-04` (`Float`) | **Floating boundary** beats Pinned & Optimized by 180–300x by eliminating boundary-vs-ODE loss competition. |
| **[Fig. 10](./fig10_navier_stokes.md)** | Quasi-1D stationary Navier–Stokes nozzle ($\rho, T, V$, sonic singularity) | **3-circuit system** + **singularity exclusion** + **regularization schedule $\zeta(n_j)$** | `3.00e-04` (full nozzle) | Excluding the sonic throat $(0.4, 0.6)$ + reverse-sigmoid regularization bridges a singularity where classical solvers fail. |

## How the 4 Figures Build on Each Other

1. **Fig. 6** establishes *which basis* (`chebyshev_tower`) makes a single DQC expressive enough for oscillatory ODEs.
2. **Fig. 7** pushes single-ODE complexity further with *nonlinearity + high frequency + Hamiltonian evolution*.
3. **Fig. 8** scales from 1 function to *coupled systems* and proves why *floating boundary handling* is the default choice.
4. **Fig. 10** combines everything — *multi-circuit parallel encoding + Chebyshev basis + floating boundaries + staged warm-start + decaying regularization* — to solve a real-world stiff aerospace fluid dynamics problem.
