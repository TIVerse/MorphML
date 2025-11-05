"""Strategy evolution and adaptive optimization.

Learns which optimization strategies work best through:
- Multi-armed bandits
- Adaptive strategy selection
- Portfolio optimization
- Hyperparameter tuning

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from morphml.meta_learning.strategy_evolution.adaptive_optimizer import AdaptiveOptimizer
from morphml.meta_learning.strategy_evolution.bandit import (
    StrategySelector,
    ThompsonSamplingSelector,
    UCBSelector,
)
from morphml.meta_learning.strategy_evolution.portfolio import PortfolioOptimizer

__all__ = [
    "StrategySelector",
    "UCBSelector",
    "ThompsonSamplingSelector",
    "AdaptiveOptimizer",
    "PortfolioOptimizer",
]
