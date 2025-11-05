"""Bayesian Optimization algorithms for neural architecture search.

This module implements sample-efficient Bayesian optimization methods including:
- Gaussian Process (GP) optimization with various acquisition functions
- Tree-structured Parzen Estimator (TPE)
- Sequential Model-based Algorithm Configuration (SMAC)

Example:
    >>> from morphml.optimizers.bayesian import GaussianProcessOptimizer
    >>> optimizer = GaussianProcessOptimizer(
    ...     search_space=space,
    ...     acquisition='ei',
    ...     n_initial_points=10
    ... )
    >>> best = optimizer.optimize(evaluator)
"""

from morphml.optimizers.bayesian.acquisition import (
    AcquisitionOptimizer,
    expected_improvement,
    get_acquisition_function,
    probability_of_improvement,
    thompson_sampling,
    upper_confidence_bound,
)
from morphml.optimizers.bayesian.base import BaseBayesianOptimizer
from morphml.optimizers.bayesian.gaussian_process import (
    GaussianProcessOptimizer,
    optimize_with_gp,
)
from morphml.optimizers.bayesian.smac import SMACOptimizer, optimize_with_smac
from morphml.optimizers.bayesian.tpe import TPEOptimizer, optimize_with_tpe

__all__ = [
    # Base class
    "BaseBayesianOptimizer",
    # Optimizers
    "GaussianProcessOptimizer",
    "TPEOptimizer",
    "SMACOptimizer",
    # Convenience functions
    "optimize_with_gp",
    "optimize_with_tpe",
    "optimize_with_smac",
    # Acquisition functions
    "expected_improvement",
    "upper_confidence_bound",
    "probability_of_improvement",
    "thompson_sampling",
    "get_acquisition_function",
    "AcquisitionOptimizer",
]
