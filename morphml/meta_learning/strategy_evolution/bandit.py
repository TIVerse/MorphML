"""Multi-armed bandit algorithms for strategy selection.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from morphml.logging_config import get_logger

logger = get_logger(__name__)


class StrategySelector(ABC):
    """
    Base class for strategy selection algorithms.

    Subclasses implement different bandit algorithms:
    - UCB (Upper Confidence Bound)
    - Thompson Sampling
    - Epsilon-Greedy
    """

    def __init__(self, strategies: List[str]):
        """
        Initialize selector.

        Args:
            strategies: List of strategy names
        """
        self.strategies = strategies
        self.num_strategies = len(strategies)

        # Statistics
        self.counts = np.zeros(self.num_strategies)
        self.rewards = np.zeros(self.num_strategies)

        logger.info(f"Initialized {self.__class__.__name__} with {self.num_strategies} strategies")

    @abstractmethod
    def select_strategy(self) -> str:
        """Select a strategy to use."""
        pass

    def update(self, strategy: str, reward: float) -> None:
        """
        Update statistics after using a strategy.

        Args:
            strategy: Strategy that was used
            reward: Reward obtained (e.g., fitness improvement)
        """
        try:
            idx = self.strategies.index(strategy)
        except ValueError:
            logger.warning(f"Unknown strategy: {strategy}")
            return

        self.counts[idx] += 1
        self.rewards[idx] += reward

        logger.debug(
            f"Updated {strategy}: pulls={self.counts[idx]:.0f}, "
            f"avg_reward={self.rewards[idx]/self.counts[idx]:.4f}"
        )

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all strategies.

        Returns:
            Dict mapping strategy name to statistics
        """
        stats = {}

        for i, strategy in enumerate(self.strategies):
            if self.counts[i] > 0:
                stats[strategy] = {
                    "pulls": int(self.counts[i]),
                    "total_reward": float(self.rewards[i]),
                    "avg_reward": float(self.rewards[i] / self.counts[i]),
                }
            else:
                stats[strategy] = {
                    "pulls": 0,
                    "total_reward": 0.0,
                    "avg_reward": 0.0,
                }

        return stats

    def get_best_strategy(self) -> str:
        """Get strategy with highest average reward."""
        mean_rewards = self.rewards / (self.counts + 1e-8)
        best_idx = np.argmax(mean_rewards)
        return self.strategies[best_idx]


class UCBSelector(StrategySelector):
    """
    Upper Confidence Bound (UCB) strategy selector.

    UCB balances exploration and exploitation using:
        UCB(i) = mean_reward(i) + c * sqrt(log(total) / pulls(i))

    Args:
        strategies: List of strategy names
        exploration_factor: Exploration constant (default: 2.0)

    Example:
        >>> selector = UCBSelector(['GA', 'BO', 'DE'])
        >>> strategy = selector.select_strategy()
        >>> # ... run strategy ...
        >>> selector.update(strategy, reward=0.15)
    """

    def __init__(self, strategies: List[str], exploration_factor: float = 2.0):
        """Initialize UCB selector."""
        super().__init__(strategies)
        self.exploration_factor = exploration_factor

    def select_strategy(self) -> str:
        """
        Select strategy using UCB algorithm.

        Returns:
            Selected strategy name
        """
        total_pulls = self.counts.sum()

        # Pull each arm at least once
        if total_pulls < self.num_strategies:
            idx = int(total_pulls)
            return self.strategies[idx]

        # Compute UCB scores
        mean_rewards = self.rewards / (self.counts + 1e-8)
        exploration_bonus = self.exploration_factor * np.sqrt(
            np.log(total_pulls) / (self.counts + 1e-8)
        )

        ucb_scores = mean_rewards + exploration_bonus

        # Select strategy with highest UCB
        best_idx = np.argmax(ucb_scores)

        logger.debug(
            f"UCB scores: {dict(zip(self.strategies, ucb_scores))}, "
            f"selected: {self.strategies[best_idx]}"
        )

        return self.strategies[best_idx]


class ThompsonSamplingSelector(StrategySelector):
    """
    Thompson Sampling strategy selector.

    Uses Beta distribution for each strategy:
        Beta(alpha, beta) where alpha = successes, beta = failures

    Args:
        strategies: List of strategy names
        prior_alpha: Prior alpha (pseudo-successes)
        prior_beta: Prior beta (pseudo-failures)

    Example:
        >>> selector = ThompsonSamplingSelector(['GA', 'BO'])
        >>> strategy = selector.select_strategy()
    """

    def __init__(self, strategies: List[str], prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """Initialize Thompson Sampling selector."""
        super().__init__(strategies)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        # Track successes and failures
        self.successes = np.ones(self.num_strategies) * prior_alpha
        self.failures = np.ones(self.num_strategies) * prior_beta

    def select_strategy(self) -> str:
        """
        Select strategy using Thompson Sampling.

        Returns:
            Selected strategy name
        """
        # Sample from Beta distribution for each strategy
        samples = np.random.beta(self.successes, self.failures)

        # Select strategy with highest sample
        best_idx = np.argmax(samples)

        logger.debug(
            f"Thompson samples: {dict(zip(self.strategies, samples))}, "
            f"selected: {self.strategies[best_idx]}"
        )

        return self.strategies[best_idx]

    def update(self, strategy: str, reward: float) -> None:
        """
        Update with reward (interpreted as success probability).

        Args:
            strategy: Strategy used
            reward: Reward in [0, 1] (treated as success probability)
        """
        super().update(strategy, reward)

        try:
            idx = self.strategies.index(strategy)
        except ValueError:
            return

        # Interpret reward as success probability
        # Use reward to update Beta distribution
        self.successes[idx] += reward
        self.failures[idx] += 1.0 - reward


class EpsilonGreedySelector(StrategySelector):
    """
    Epsilon-Greedy strategy selector.

    With probability epsilon, explore (random choice).
    With probability 1-epsilon, exploit (best strategy).

    Args:
        strategies: List of strategy names
        epsilon: Exploration probability (default: 0.1)
        epsilon_decay: Decay rate for epsilon (default: 0.99)
    """

    def __init__(self, strategies: List[str], epsilon: float = 0.1, epsilon_decay: float = 0.99):
        """Initialize Epsilon-Greedy selector."""
        super().__init__(strategies)
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def select_strategy(self) -> str:
        """
        Select strategy using epsilon-greedy.

        Returns:
            Selected strategy name
        """
        # Explore
        if np.random.rand() < self.epsilon:
            idx = np.random.randint(self.num_strategies)
            logger.debug(f"Exploring: selected {self.strategies[idx]}")
            return self.strategies[idx]

        # Exploit
        mean_rewards = self.rewards / (self.counts + 1e-8)
        best_idx = np.argmax(mean_rewards)

        logger.debug(f"Exploiting: selected {self.strategies[best_idx]}")

        return self.strategies[best_idx]

    def update(self, strategy: str, reward: float) -> None:
        """Update and decay epsilon."""
        super().update(strategy, reward)

        # Decay epsilon
        self.epsilon *= self.epsilon_decay

    def reset_epsilon(self) -> None:
        """Reset epsilon to initial value."""
        self.epsilon = self.initial_epsilon
