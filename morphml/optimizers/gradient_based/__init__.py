"""Gradient-based Neural Architecture Search algorithms.

This module implements differentiable NAS methods that use gradient descent:
- DARTS (Differentiable Architecture Search)
- ENAS (Efficient Neural Architecture Search with weight sharing)

These methods require GPU acceleration and PyTorch.

Example:
    >>> from morphml.optimizers.gradient_based import DARTS
    >>> optimizer = DARTS(
    ...     search_space=space,
    ...     epochs=50,
    ...     learning_rate=0.025
    ... )
    >>> best = optimizer.optimize(evaluator)
"""

from morphml.optimizers.gradient_based.darts import DARTSOptimizer as DARTS
from morphml.optimizers.gradient_based.enas import ENASOptimizer as ENAS

__all__ = ["DARTS", "ENAS"]
