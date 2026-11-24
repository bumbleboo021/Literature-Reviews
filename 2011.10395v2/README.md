# Solving Nonlinear Differential Equations with Differentiable Quantum Circuits

Implementation of **Kyriienko, Paine, & Elfving (2021)**, *arXiv:2011.10395v2*, using PennyLane + PyTorch.

## Structure
- `dqc/` — Core DQC library (feature maps, ansatzes, cost observables, analytical differentiation, boundary handling, losses, solver).
- `problems/` — ODE definitions from Section IV of the paper.
- `examples/` — Small, runnable demos showing how to use each module and inspect outputs.
- `docs/foundations.md` — Core theory & practical takeaways (feature maps, ansatz, derivatives, boundary handling, regularization).
- `docs/examples/` — Markdown guides for the simple demos in `examples/`.
- `docs/experiments/` — Markdown guides for each paper figure reproduction in `experiments/`.
- `docs/extensions/` — Follow-up papers and design notes for extending the baseline DQC implementation.
- `experiments/` — Scripts reproducing Figures 6, 7, 8, and 10.
- `tests/` — Unit tests verifying circuit math and analytical derivatives.
- `results/` — Generated figures and checkpoints (gitignored).
