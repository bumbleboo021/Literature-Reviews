# Example 01: Feature Maps, HEA Ansatz, and Cost Observable

**Script:** [`examples/01_feature_maps_and_ansatz.py`](../../examples/01_feature_maps_and_ansatz.py)  
**Paper reference:** Section III.A–III.C, Fig. 2(a), Fig. 5(a) of `arXiv:2011.10395v2`.

## Purpose

This example demonstrates how the three core circuit building blocks fit together before we add analytical differentiation or optimization:

1. **Quantum Feature Map** $\hat{\mathcal{U}}_\varphi(x)$ (`dqc/feature_maps.py`): encodes the scalar input $x \in (-1, 1)$ into qubit rotations (`product`, `chebyshev`, `chebyshev_tower`).
2. **Variational Ansatz** $\hat{\mathcal{U}}_\theta$ (`dqc/ansatz.py`): applies `depth` layers of single-qubit $\hat{R}_z$–$\hat{R}_x$–$\hat{R}_z$ rotations and nearest-neighbor `CNOT` gates.
3. **Cost Observable** $\hat{C}$ (`dqc/cost.py`): reads out a real scalar $f(x) = \langle \emptyset | \hat{\mathcal{U}}_\varphi^\dagger(x)\hat{\mathcal{U}}_\theta^\dagger \hat{C} \hat{\mathcal{U}}_\theta \hat{\mathcal{U}}_\varphi(x) | \emptyset \rangle$ using total $Z$-magnetization $\hat{C} = \sum_j \hat{Z}_j$.

## How to Run

```bash
python 2011.10395v2/examples/01_feature_maps_and_ansatz.py
```

## What It Returns

1. **Parameter shape & count**: For $N = 3$ qubits and `depth = 2`, `init_hea_params(3, 2, seed=42)` returns a `torch.Tensor` of shape `(2, 3, 3)` (18 trainable angles in $[0, 2\pi]$).
2. **Circuit diagram (`qml.draw`)**: Prints the gate-by-gate circuit for `chebyshev_tower` at $x = 0.5$ ($\arccos(0.5) = \pi/3$, so the feature map angles are $2\pi/3 \approx 2.09$, $4\pi/3 \approx 4.19$, and $2\pi \approx 6.28$ on qubits 0, 1, 2):

```text
0: ──RY(2.09)──RZ(0.37)──RX(0.40)──RZ(0.78)─╭●──RZ(0.83)──RX(0.99)──RZ(2.36)───────────╭●────┤ ╭<𝓗>
1: ──RY(4.19)──RZ(0.33)──RX(3.31)──RZ(3.00)─╰X─╭●─────────RZ(5.29)──RX(5.47)──RZ(2.37)─╰X─╭●─┤ ├<𝓗>
2: ──RY(6.28)──RZ(6.00)──RX(5.84)──RZ(0.52)────╰X─────────RZ(3.85)──RX(0.55)──RZ(4.41)────╰X─┤ ╰<𝓗>
```

3. **Expectation values $f(x)$**: Evaluates all three feature maps with the same random $\theta$ at $x \in \{0.0, 0.3, 0.6, 0.9\}$:

```text
  product          -> x=[0.0, 0.3, 0.6, 0.9] -> f(x)=[+0.2410, +0.1505, -0.1223, -0.5230]
  chebyshev        -> x=[0.0, 0.3, 0.6, 0.9] -> f(x)=[+0.2023, +0.0436, -0.4556, -0.3630]
  chebyshev_tower  -> x=[0.0, 0.3, 0.6, 0.9] -> f(x)=[-0.7936, +0.2085, +0.3347, +0.1077]
```

Notice how `chebyshev_tower` varies more rapidly across $x$ even for the exact same random ansatz parameters, because higher-indexed qubits encode higher-degree Chebyshev polynomials $T_j(x)$.
