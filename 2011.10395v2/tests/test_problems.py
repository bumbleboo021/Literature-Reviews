"""Unit tests verifying all Section IV problem definitions and evolution-enhanced DQC."""

import torch

from dqc.circuit import DifferentiableQuantumCircuit
from dqc.feature_maps import build_ising_evolution_unitary
from problems.coupled_odes import CoupledODEProblem
from problems.damped_oscillator import DampedOscillatorProblem
from problems.navier_stokes import NavierStokesNozzleProblem
from problems.nontrivial_ode import NontrivialODEProblem


def test_damped_oscillator_exact_solution_zero_residual():
    prob = DampedOscillatorProblem(lam=8.0, kappa=0.1, u0=1.0)
    x = prob.grid().requires_grad_(True)
    u = prob.true_solution(x)[0]
    du = torch.autograd.grad(u.sum(), x)[0]
    res = prob.residual(x, [u], [du])[0]
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-12)


def test_coupled_odes_exact_solution_zero_residual():
    prob = CoupledODEProblem(lam1=5.0, lam2=3.0, u1_0=0.5, u2_0=0.0)
    x = prob.grid().requires_grad_(True)
    u1, u2 = prob.true_solution(x)
    du1 = torch.autograd.grad(u1.sum(), x, retain_graph=True)[0]
    du2 = torch.autograd.grad(u2.sum(), x)[0]
    f1, f2 = prob.residual(x, [u1, u2], [du1, du2])
    assert torch.allclose(f1, torch.zeros_like(f1), atol=1e-12)
    assert torch.allclose(f2, torch.zeros_like(f2), atol=1e-12)


def test_nontrivial_ode_reference_solution_satisfies_ode():
    prob = NontrivialODEProblem(u0=0.75)
    x = torch.linspace(0.0, 0.9, 50, dtype=torch.float64)
    u = prob.true_solution(x)[0]
    # Check boundary condition and finite-difference residual
    assert abs(u[0].item() - 0.75) < 1e-12
    eps = 1e-6
    x_mid = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    u_mid = prob.true_solution(x_mid)[0]
    du_fd = (prob.true_solution(x_mid + eps)[0] - prob.true_solution(x_mid - eps)[0]) / (2.0 * eps)
    res = prob.residual(x_mid, [u_mid], [du_fd])[0]
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-5)


def test_navier_stokes_isentropic_solution_satisfies_stationary_eqs():
    prob = NavierStokesNozzleProblem()
    x_mid = torch.tensor([0.2, 0.35, 0.65, 0.8], dtype=torch.float64)
    eps = 1e-6
    fs = prob.true_solution(x_mid)
    fs_p = prob.true_solution(x_mid + eps)
    fs_m = prob.true_solution(x_mid - eps)
    dfs = [(fp - fm) / (2.0 * eps) for fp, fm in zip(fs_p, fs_m)]
    f_rho, f_temp, f_vel = prob.residual(x_mid, fs, dfs)
    assert torch.allclose(f_rho, torch.zeros_like(f_rho), atol=1e-5)
    assert torch.allclose(f_temp, torch.zeros_like(f_temp), atol=1e-5)
    assert torch.allclose(f_vel, torch.zeros_like(f_vel), atol=1e-5)


def test_evolution_enhanced_dqc_autograd_matches_parameter_shift():
    u_evo = build_ising_evolution_unitary(n_qubits=3, tau=2.0, seed=42)
    dqc = DifferentiableQuantumCircuit(
        n_qubits=3, depth=2, feature_map="chebyshev_tower", evolution_unitary=u_evo, seed=11
    )
    xs = torch.tensor([0.1, 0.4, 0.7], dtype=torch.float64)
    _, df_auto = dqc.eval_with_derivative(xs)
    df_pshift = dqc.df_dx_parameter_shift(xs)
    assert torch.allclose(df_auto, df_pshift, atol=1e-7)
