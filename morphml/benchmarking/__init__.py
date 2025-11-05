"""Benchmarking and comparison tools for optimizer evaluation.

This module provides tools to systematically compare and evaluate
different optimization algorithms.

Features:
- Optimizer comparison with statistical analysis
- Convergence visualization
- Sample efficiency analysis
- Result reporting

Example:
    >>> from morphml.benchmarking import OptimizerComparison, compare_optimizers
    >>> from morphml.optimizers import GeneticAlgorithm, optimize_with_pso
    >>>
    >>> # Quick comparison
    >>> results = compare_optimizers(
    ...     optimizers={
    ...         'GA': GeneticAlgorithm(space, config),
    ...         'PSO': ParticleSwarmOptimizer(space, config)
    ...     },
    ...     search_space=space,
    ...     evaluator=my_evaluator,
    ...     budget=100
    ... )
"""

from morphml.benchmarking.comparison import (
    OptimizerComparison,
    compare_optimizers,
)

__all__ = [
    "OptimizerComparison",
    "compare_optimizers",
]
