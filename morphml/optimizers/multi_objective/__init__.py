"""Multi-objective optimization algorithms.

This module implements algorithms for optimizing multiple conflicting objectives
simultaneously, discovering Pareto-optimal solutions.

Algorithms:
- NSGA-II (Non-dominated Sorting Genetic Algorithm II)
- Pareto dominance and ranking
- Quality indicators (Hypervolume, IGD, Spacing, Spread)
- Visualization tools

Example:
    >>> from morphml.optimizers.multi_objective import NSGA2Optimizer, optimize_with_nsga2
    >>> optimizer = NSGA2Optimizer(
    ...     search_space=space,
    ...     config={
    ...         'population_size': 100,
    ...         'objectives': [
    ...             {'name': 'accuracy', 'maximize': True},
    ...             {'name': 'latency', 'maximize': False}
    ...         ]
    ...     }
    ... )
    >>> pareto_front = optimizer.optimize(evaluator)
"""

from morphml.optimizers.multi_objective.indicators import (
    QualityIndicators,
    calculate_all_indicators,
    compare_pareto_fronts,
)
from morphml.optimizers.multi_objective.nsga2 import (
    MultiObjectiveIndividual,
    NSGA2Optimizer,
    optimize_with_nsga2,
)
from morphml.optimizers.multi_objective.visualization import (
    ParetoVisualizer,
    quick_visualize_2d,
    quick_visualize_3d,
)

__all__ = [
    # Core optimizer
    "NSGA2Optimizer",
    "MultiObjectiveIndividual",
    "optimize_with_nsga2",
    # Quality indicators
    "QualityIndicators",
    "calculate_all_indicators",
    "compare_pareto_fronts",
    # Visualization
    "ParetoVisualizer",
    "quick_visualize_2d",
    "quick_visualize_3d",
]
