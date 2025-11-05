"""Advanced evolutionary optimization algorithms.

This module contains sophisticated evolutionary algorithms beyond basic GA:
- Particle Swarm Optimization (PSO) - Swarm intelligence
- Differential Evolution (DE) - Vector difference mutations
- CMA-ES (Covariance Matrix Adaptation Evolution Strategy) - Adaptive covariance

These algorithms work in continuous spaces using architecture encoding.

Example:
    >>> from morphml.optimizers.evolutionary import ParticleSwarmOptimizer, optimize_with_pso
    >>> optimizer = ParticleSwarmOptimizer(
    ...     search_space=space,
    ...     config={'num_particles': 30, 'max_iterations': 100}
    ... )
    >>> best = optimizer.optimize(evaluator)

    # Or use convenience function
    >>> best = optimize_with_pso(space, evaluator, num_particles=30)
"""

from morphml.optimizers.evolutionary.cma_es import CMAES, optimize_with_cmaes
from morphml.optimizers.evolutionary.differential_evolution import (
    DifferentialEvolution,
    optimize_with_de,
)
from morphml.optimizers.evolutionary.encoding import (
    ArchitectureEncoder,
    ContinuousArchitectureSpace,
    decode_architecture,
    encode_architecture,
)
from morphml.optimizers.evolutionary.particle_swarm import (
    Particle,
    ParticleSwarmOptimizer,
    optimize_with_pso,
)

__all__ = [
    # Encoding utilities
    "ArchitectureEncoder",
    "ContinuousArchitectureSpace",
    "encode_architecture",
    "decode_architecture",
    # Particle Swarm Optimization
    "ParticleSwarmOptimizer",
    "Particle",
    "optimize_with_pso",
    # Differential Evolution
    "DifferentialEvolution",
    "optimize_with_de",
    # CMA-ES
    "CMAES",
    "optimize_with_cmaes",
]
