"""Reusable experiment factories. Small defaults; paper-sized JSON presets in configs/."""

from dataclasses import dataclass
import numpy as np
from .ansatz import HardwareEfficientAnsatz, AlternatingBlocksAnsatz
from .boundaries import Boundary
from .feature_maps import FeatureMap, AffineDomain
from .model import QuantumModel, ParallelModel
from .observables import Readout, ising_hamiltonian
from .problems import NonlinearDecay, DampedOscillator, NonlinearOscillator, CoupledSystem, NozzleSteady
from .grids import collocation


@dataclass
class ExperimentConfig:
    problem: str = "decay"
    n_qubits: int = 3
    depth: int = 2
    axes: str = "YZ"
    feature_map: str = "chebt"
    ansatz: str = "hea"
    block_width: int = 2
    block_depth: int = 1
    boundary: str = "floating"
    trainable_readout: bool = True
    encoding: str = "parallel"
    evolution_time: float = 0.0
    normalize_domain: bool = False
    start: float = 0.0
    stop: float | None = None
    points: int = 31
    grid: str = "uniform"
    optimizer: str = "least_squares"
    steps: int = 250
    learning_rate: float = 0.01
    max_nfev: int = 180
    seed: int = 7
    boundary_weight: float = 1.0
    distance: str = "mse"
    frequency: float = 8.0
    damping: float = 0.1
    damped_residual: str = "original"
    nozzle_residual: str = "conservation"

    def __post_init__(self):
        if self.encoding not in ("parallel", "shared"):
            raise ValueError("encoding must be parallel or shared.")
        if self.ansatz not in ("hea", "aba"):
            raise ValueError("ansatz must be hea or aba.")
        if self.optimizer not in ("adam", "least_squares"):
            raise ValueError("optimizer must be adam or least_squares.")
        if self.start != 0:
            raise ValueError("Bundled benchmarks have initial values at x=0; start must be zero.")


def make_problem(config):
    factories = {
        "decay": lambda: NonlinearDecay(),
        "damped": lambda: DampedOscillator(config.frequency, config.damping, residual_form=config.damped_residual),
        "nonlinear": lambda: NonlinearOscillator(),
        "coupled": lambda: CoupledSystem(),
        "nozzle": lambda: NozzleSteady(residual_form=config.nozzle_residual),
    }
    if config.problem not in factories:
        raise ValueError(f"Unknown problem: {config.problem}. Choose {', '.join(factories)}.")
    return factories[config.problem]()


def make_model(config, problem):
    stop = problem.domain[1] if config.stop is None else config.stop
    domain = AffineDomain.interval(config.start, stop) if config.normalize_domain else AffineDomain()
    h = None
    if config.evolution_time:
        rng = np.random.default_rng(config.seed)
        h = ising_hamiltonian(config.n_qubits,
                              couplings=-rng.uniform(0.01, 1., config.n_qubits-1),
                              transverse=rng.uniform(0.01, 1., config.n_qubits))

    def one(values, index):
        fmap = FeatureMap(config.n_qubits, config.feature_map, domain, h, config.evolution_time)
        if config.ansatz == "hea":
            ansatz = HardwareEfficientAnsatz(config.n_qubits, config.depth, config.axes)
        else:
            ansatz = AlternatingBlocksAnsatz(config.n_qubits, config.depth,
                                             config.block_width, config.block_depth, config.axes)
        if len(values) > 1 or config.trainable_readout:
            readout = Readout.weighted_z(config.n_qubits, len(values),
                                         config.trainable_readout, config.seed+index)
        else:
            readout = Readout.magnetization(config.n_qubits)
        return QuantumModel(fmap, ansatz, readout, Boundary(config.boundary, 0., values))

    if config.encoding == "shared" or problem.n_outputs == 1:
        return one(problem.initial_values, 0)
    return ParallelModel([one([u0], i) for i, u0 in enumerate(problem.initial_values)])


def make_experiment(config):
    problem = make_problem(config)
    stop = problem.domain[1] if config.stop is None else config.stop
    grid = collocation(config.start, stop, config.points, config.grid, config.seed)
    return make_model(config, problem), problem, grid

