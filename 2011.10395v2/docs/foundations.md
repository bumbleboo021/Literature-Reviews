# Foundations & Key Takeaways: Differentiable Quantum Circuits (arXiv:2011.10395v2)

This guide summarizes the core concepts behind **Kyriienko, Paine, & Elfving (2021)**, *"Solving nonlinear differential equations with differentiable quantum circuits"*, and links every concept to our PennyLane + PyTorch implementation.

---

## 1. The Big Picture: What Is a DQC?

A **Differentiable Quantum Circuit (DQC)** solves differential equations of the form

$$F\big[\{d^m f_n / dx^m\}, \{f_n(x)\}, x\big] = 0$$

by representing the trial solution $f_\theta(x)$ as the expectation value of a parametrized quantum circuit:

$$f_\theta(x) = \langle \emptyset | \hat{\mathcal{U}}_\varphi^\dagger(x)\,\hat{\mathcal{U}}_\theta^\dagger\,\hat{C}\,\hat{\mathcal{U}}_\theta\,\hat{\mathcal{U}}_\varphi(x) | \emptyset \rangle$$

Think of it as a **quantum spectral method** (or the quantum cousin of a Physics-Informed Neural Network, PINN):
- Instead of discretizing $u(x)$ onto a huge grid (finite differences) or amplitude-encoding a grid vector into a quantum state (HHL, which suffers from QRAM input and tomography output bottlenecks), a DQC maps $x$ into a **$2^N$-dimensional latent Hilbert space** of basis functions and learns variational angles $\theta$ via classical Adam optimization.
- Once trained, the circuit can be evaluated at **any** continuous point $x$ — including points between or beyond the training grid.

---

## 2. Quantum Feature Maps $\hat{\mathcal{U}}_\varphi(x)$ — `dqc/feature_maps.py`

A feature map encodes the scalar variable $x \in (-1, 1)$ into qubit rotation angles $\varphi_j(x)$. The choice of $\varphi_j(x)$ determines the **basis set of functions** your circuit can express:

| Feature Map | Gate per Qubit $j \in \{1, \dots, N\}$ | Basis Functions Generated | Expressivity |
| :--- | :--- | :--- | :--- |
| **Product** (Eq. 6) | $\hat{R}_y(\arcsin x)$ | Products of $\{1, x, \sqrt{1-x^2}\}$ | Low — struggles on oscillatory ODEs |
| **Chebyshev** (Eq. 14) | $\hat{R}_y(2\arccos x)$ | Degree-1 Chebyshev $T_1(x)$ chained across $N$ qubits | Medium — solves $\lambda=8$, struggles at $\lambda=20$ |
| **Chebyshev Tower** (Eq. 15) | $\hat{R}_y(2j\arccos x)$ | Tower of degrees $T_1(x), T_2(x), \dots, T_N(x)$ | High — captures rapid oscillations ($\lambda=20$) |
| **Evolution-Enhanced** (Fig. 3c) | Chebyshev Tower $+ e^{-i\hat{H}\tau}$ | Entangled Chebyshev + Fourier basis in full $2^N$ space | Very high — handles nontrivial dynamics (Fig. 7) |

### Things to Know & Takeaways
- **Why the factor of `2` in `2 * arccos(x)`?** Because $\hat{R}_y(\phi) = \exp(-i \frac{\phi}{2}\hat{Y})$ divides the angle by $2$. Setting $\phi = 2j\arccos(x)$ cancels the $\frac{1}{2}$, making the $|0\rangle$ amplitude literally $\cos(j\arccos x) = T_j(x)$ — the degree-$j$ Chebyshev polynomial of the first kind (Eq. 12).
- **1-based indexing matters**: In `chebyshev_tower`, qubit $j$ must use $j = 1, 2, \dots, N$ (`enumerate(wires, start=1)`). Starting at $0$ leaves qubit 0 unrotated!
- **Stay inside $x \in (-1, 1)$**: The derivatives $\frac{d}{dx}\arcsin(x) = \frac{1}{\sqrt{1-x^2}}$ and $\frac{d}{dx}\arccos(x) = -\frac{1}{\sqrt{1-x^2}}$ diverge at $x = \pm 1$. That's why the paper trains on $x \in [0, 0.9]$ (any physical domain can be rescaled into $(-1, 1)$).
- **Experimental takeaway (Fig. 6)**: Feature-map expressivity is often more important than circuit depth — on $\lambda=20$, `chebyshev_tower` achieved **$200\times$ lower MSE** ($L_Q = 8.89\times 10^{-4}$) than `product` or sparse `chebyshev` with the exact same ansatz!

---

## 3. Variational Ansatz $\hat{\mathcal{U}}_\theta$ — `dqc/ansatz.py`

If the feature map supplies the *basis functions*, the variational ansatz supplies the **trainable weights** that mix and entangle those basis functions into the ODE solution.

- **Hardware-Efficient Ansatz (HEA, Fig. 5a)**: Repeats `depth` blocks of:
  1. Single-qubit Euler rotations $\hat{R}_z(\theta_0)\,\hat{R}_x(\theta_1)\,\hat{R}_z(\theta_2)$ on every qubit (able to implement any single-qubit unitary in $SU(2)$).
  2. Nearest-neighbor linear `CNOT` chain `CNOT(0,1), ..., CNOT(N-2, N-1)`.
- **Parameter count**: $\text{depth} \times N \times 3$ angles, initialized uniformly in $[0, 2\pi]$.
- **Expressivity vs. Trainability**: Deeper circuits can represent more complex functions, but overly deep global entangling circuits eventually hit **barren plateaus** (exponentially vanishing gradients). In the paper's 6-qubit benchmarks, $d = 5$–$6$ (90–108 parameters) is the sweet spot.

---

## 4. Cost Observable $\hat{C}$ — `dqc/cost.py`

Measuring a Hermitian observable $\hat{C}$ projects the $2^N$-dimensional quantum state down to a single real number $f(x) = \langle \hat{C} \rangle$.

- **Single-qubit magnetization** $\hat{Z}_0$: eigenvalues live in $[-1, 1]$ — any solution outside $[-1, 1]$ cannot be represented without rescaling.
- **Total $Z$-magnetization** $\hat{C} = \sum_{j=0}^{N-1} \hat{Z}_j$ (used in all Section IV benchmarks): eigenvalues span $[-N, +N]$ (e.g. $[-6, +6]$ for 6 qubits), giving plenty of dynamic range for functions like velocity $V(x) \in [0.1, 1.8]$ in the Navier–Stokes nozzle.
- **Symmetry rule (p. 8)**: Choose ansatz generators $\hat{G}_j$ that do *not* commute with $\hat{C}$ ($[\hat{C}, \hat{G}_j] \neq 0$), otherwise the ansatz cannot change $\langle \hat{C} \rangle$.

---

## 5. Exact Analytical Derivatives $df/dx$ — `dqc/circuit.py`

A central contribution of `2011.10395v2` is avoiding noisy finite-difference approximations of $df/dx$.

- **On quantum hardware (Parameter-Shift Rule, Eq. 2 / Fig. 3b)**:

  $$
  \frac{df}{dx} = \frac{1}{2} \sum_{j=1}^{N} \frac{d\varphi_j(x)}{dx} \Big(\langle \hat{C} \rangle_{\varphi_j + \pi/2} - \langle \hat{C} \rangle_{\varphi_j - \pi/2}\Big)
  $$

  Note: Eq. 9 on p. 6 of the paper has a printed typo with $\frac{1}{4}$; Eq. 2 on p. 3 has the true factor $\frac{1}{2}$, verified to $8.88 \times 10^{-15}$ in `tests/test_derivatives.py`.
- **On a classical simulator (`default.qubit` + PyTorch `backprop`)**:
  We compute $df/dx$ via `torch.autograd.grad(f.sum(), x, create_graph=True)[0]`, which is algebraically identical to the parameter-shift rule (to machine precision) and much faster.
- **Why `create_graph=True` is mandatory**: During Adam training, our loss $\mathcal{L}[df/dx, f, x]$ depends on $df/dx$, so `loss.backward()` must compute $\frac{\partial}{\partial \theta}\left(\frac{\partial f}{\partial x}\right)$ — a second-order mixed derivative. Without `create_graph=True`, gradients stop flowing through $df/dx$ into $\theta$.

---

## 6. Boundary Handling — `dqc/boundary.py`

An ODE without boundary/initial conditions has infinitely many solutions. Section III.E introduces three ways to enforce $f(x_0) = u_0$:

| Mode | Formula | Boundary Loss | Pros & Cons |
| :--- | :--- | :--- | :--- |
| **Pinned** (Eq. 24) | $f(x) = \langle \hat{C} \rangle(x)$ | $\eta (f(x_0) - u_0)^2$ | Simple, but boundary loss competes with ODE loss during Adam. |
| **Optimized** (Eq. 27) | $f(x) = f_c + \langle \hat{C} \rangle(x)$ | $\eta (f(x_0) - u_0)^2$ | Classical shift $f_c$ adapts quickly, but loss competition remains. |
| **Floating** (Eqs. 25–26) | $f(x) = \big(u_0 - \langle \hat{C} \rangle(x_0)\big) + \langle \hat{C} \rangle(x)$ | **None ($0$)** | **Exact** $f(x_0) = u_0$ at every step; zero loss competition! |

### Experimental Takeaway (Fig. 8)
On the strongly coupled ODE system, **Floating** reached $L_Q = 2.00\times 10^{-4}$ in 250 steps — **180–300× more accurate** than **Optimized** ($3.64\times 10^{-2}$) and **Pinned** ($6.18\times 10^{-2}$). Use `floating` whenever you have an exact initial value $u_0$.

---

## 7. Losses, Multi-Function Systems & Staged Regularization — `dqc/losses.py`, `dqc/solver.py`

- **Three metrics tracked in the paper**:
  - $L_F$ (**Full Loss**): $\mathcal{L}^{(\text{diff})} + \mathcal{L}^{(\text{boundary})} + \mathcal{L}^{(\text{reg})}$ — what Adam actually minimizes.
  - $L_D$ (**Differential Loss**): $L_F$ excluding regularization — measures true physics compliance.
  - $L_Q$ (**Quality of Solution**): $\frac{1}{M}\sum_i (f(x_i) - u_{\text{true}}(x_i))^2$ — diagnostic metric when the true solution is known.
- **Systems of ODEs (Sec. III.G)**: Assigning a separate quantum circuit $(\theta_k)$ to each unknown function (e.g., $u_1, u_2$ in Fig. 8; $\rho, T, V$ in Fig. 10) avoids competition between observables on a shared register and lets circuits run in parallel.
- **Avoiding singularities in residuals**:
  - *Algebraic clearing (Fig. 6)*: Multiplying Eq. (30) by $\cos(\lambda x)$ removes the $\tan(\lambda x)$ poles without changing the solution, preventing MSE spikes that pin $f(x) \to 0$.
  - *Excluding singular domains + staged regularization (Fig. 10)*: For the convergent-divergent nozzle, the stationary RHS diverges at the sonic throat $T = V^2$ ($x = 0.5$). Training Stage 1 on $[0, 0.4]$, then Stage 2 on $[0, 0.4] \cup [0.6, 0.9]$ with a reverse-sigmoid regularization schedule $\zeta(n_j) = 1 - \tanh\big(\frac{n_j - n_{\text{drop}}}{\delta n_{\text{iter}}}\big)$ (Eq. 29) smoothly guides the DQC onto the supersonic branch and then hands full control back to the Navier–Stokes equations!

---

## 8. Software Architecture Takeaways

When turning a research paper into clean, reusable code:
1. **Separate the algorithm (`dqc/`) from the problem definitions (`problems/`) and runnable scripts (`experiments/`)** so adding a new ODE never touches the solver.
2. **Mirror the paper's sections 1-to-1** (`feature_maps.py`, `ansatz.py`, `cost.py`, `circuit.py`, `boundary.py`, `losses.py`, `solver.py`) so every equation has an obvious home.
3. **Write fast unit tests (`tests/`) before running long training loops** — analytical identities on 1–2 qubits and finite-difference checks on $df/dx$ catch silent math bugs in seconds.
4. **Separate `examples/` + `docs/examples/` (tiny educational demos) from `experiments/` + `docs/experiments/` (full paper reproductions)** so anyone opening the repo can learn the API in 2 minutes.
