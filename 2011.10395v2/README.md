# Solving Nonlinear Differential Equations with Differentiable Quantum Circuits

Implementation of **Kyriienko, Paine, & Elfving (2021)**, *arXiv:2011.10395v2*, using PennyLane + PyTorch.

## Structure
- `dqc/` — Core DQC library (feature maps, ansatzes, cost observables, analytical differentiation, boundary handling, losses, solver).
- `problems/` — ODE definitions from Section IV of the paper.
- `examples/` — Small, runnable demos showing how to use each module and inspect outputs.
- `docs/examples/` — Markdown guides explaining each example's math, usage, and output.
- `experiments/` — Scripts reproducing Figures 6, 7, 8, and 10.
- `tests/` — Unit tests verifying circuit math and analytical derivatives.
- `results/` — Generated figures and checkpoints (gitignored).
