"""Objective functions and multi-objective support.

This module provides utilities for defining and evaluating optimization objectives:
- Single objective evaluation
- Multi-objective evaluation
- Pareto dominance relationships
- Quality indicators (hypervolume, IGD)

Example:
    >>> from morphml.core.objectives import MultiObjectiveEvaluator
    >>> evaluator = MultiObjectiveEvaluator(
    ...     objectives={
    ...         'accuracy': lambda g: train_and_evaluate(g),
    ...         'latency': lambda g: estimate_latency(g),
    ...         'parameters': lambda g: count_parameters(g)
    ...     }
    ... )
"""

__all__ = []  # Will be populated as components are implemented
