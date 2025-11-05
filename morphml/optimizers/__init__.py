"""Optimization algorithms for neural architecture search.

Phase 1 (Evolutionary):
    - GeneticAlgorithm, RandomSearch, HillClimbing, SimulatedAnnealing, DifferentialEvolution

Phase 2 (Advanced):
    - Bayesian: GaussianProcessOptimizer, TPEOptimizer, SMACOptimizer
    - Gradient-based: DARTS, ENAS
    - Multi-objective: NSGA2Optimizer
    - Advanced Evolutionary: CMAESOptimizer, ParticleSwarmOptimizer, DifferentialEvolutionOptimizer
"""

# Phase 1: Core evolutionary algorithms
from morphml.optimizers.differential_evolution import DifferentialEvolution
from morphml.optimizers.genetic_algorithm import GeneticAlgorithm
from morphml.optimizers.hill_climbing import HillClimbing
from morphml.optimizers.random_search import RandomSearch
from morphml.optimizers.simulated_annealing import SimulatedAnnealing

# Phase 2: Bayesian Optimization
from morphml.optimizers.bayesian import (
    GaussianProcessOptimizer,
    SMACOptimizer,
    TPEOptimizer,
)

# Phase 2: Gradient-based NAS (requires PyTorch)
try:
    from morphml.optimizers.gradient_based import DARTS, ENAS
    _GRADIENT_BASED_AVAILABLE = True
except ImportError:
    DARTS = None
    ENAS = None
    _GRADIENT_BASED_AVAILABLE = False

# Phase 2: Multi-objective (canonical NSGA-II from multi_objective module)
from morphml.optimizers.multi_objective import NSGA2Optimizer

# Phase 2: Advanced Evolutionary
from morphml.optimizers.evolutionary import (
    CMAESOptimizer,
    DifferentialEvolutionOptimizer,
    ParticleSwarmOptimizer,
)

# Legacy alias for backward compatibility
from morphml.optimizers.nsga2 import NSGA2 as NSGA2Legacy

# Primary exports
__all__ = [
    # Phase 1
    "GeneticAlgorithm",
    "RandomSearch",
    "HillClimbing",
    "SimulatedAnnealing",
    "DifferentialEvolution",
    # Phase 2: Bayesian
    "GaussianProcessOptimizer",
    "TPEOptimizer",
    "SMACOptimizer",
    # Phase 2: Gradient-based
    "DARTS",
    "ENAS",
    # Phase 2: Multi-objective
    "NSGA2Optimizer",
    # Phase 2: Advanced Evolutionary
    "CMAESOptimizer",
    "ParticleSwarmOptimizer",
    "DifferentialEvolutionOptimizer",
    # Legacy
    "NSGA2Legacy",
]

# Convenience alias: NSGA2 points to canonical multi-objective implementation
NSGA2 = NSGA2Optimizer
