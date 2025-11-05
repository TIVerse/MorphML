"""Adaptive optimizer that switches strategies dynamically.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Any, Callable, Dict, List, Optional

from morphml.core.dsl import SearchSpace
from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger
from morphml.meta_learning.strategy_evolution.bandit import StrategySelector, UCBSelector

logger = get_logger(__name__)


class AdaptiveOptimizer:
    """
    Adaptive optimizer that switches between strategies.
    
    Uses multi-armed bandits to learn which optimizer works best
    and dynamically switches during search.
    
    Args:
        search_space: Search space definition
        evaluator: Architecture evaluation function
        strategy_configs: Dict mapping strategy name to config
        selector_type: Type of strategy selector ('ucb', 'thompson', 'epsilon')
        selector_config: Config for selector
    
    Example:
        >>> from morphml.optimizers import GeneticAlgorithm, RandomSearch
        >>> 
        >>> configs = {
        ...     'ga': {'population_size': 50, 'num_generations': 10},
        ...     'random': {'num_samples': 100}
        ... }
        >>> 
        >>> optimizer = AdaptiveOptimizer(
        ...     search_space=space,
        ...     evaluator=evaluator,
        ...     strategy_configs=configs
        ... )
        >>> 
        >>> best = optimizer.search(budget=500)
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: Callable[[ModelGraph], float],
        strategy_configs: Dict[str, Dict[str, Any]],
        selector_type: str = "ucb",
        selector_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize adaptive optimizer."""
        self.search_space = search_space
        self.evaluator = evaluator
        self.strategy_configs = strategy_configs
        
        # Create strategy selector
        strategies = list(strategy_configs.keys())
        selector_config = selector_config or {}
        
        if selector_type == "ucb":
            from morphml.meta_learning.strategy_evolution.bandit import UCBSelector
            self.selector = UCBSelector(strategies, **selector_config)
        elif selector_type == "thompson":
            from morphml.meta_learning.strategy_evolution.bandit import ThompsonSamplingSelector
            self.selector = ThompsonSamplingSelector(strategies, **selector_config)
        elif selector_type == "epsilon":
            from morphml.meta_learning.strategy_evolution.bandit import EpsilonGreedySelector
            self.selector = EpsilonGreedySelector(strategies, **selector_config)
        else:
            raise ValueError(f"Unknown selector type: {selector_type}")
        
        # Track progress
        self.history = []
        self.best_fitness = -float('inf')
        self.best_architecture = None
        
        logger.info(
            f"Initialized AdaptiveOptimizer with {len(strategies)} strategies: {strategies}"
        )
    
    def search(self, budget: int = 500, checkpoint_interval: int = 50) -> ModelGraph:
        """
        Run adaptive search.
        
        Args:
            budget: Total number of evaluations
            checkpoint_interval: How often to report progress
        
        Returns:
            Best architecture found
        """
        logger.info(f"Starting adaptive search with budget={budget}")
        
        evaluations_used = 0
        
        while evaluations_used < budget:
            # Select strategy
            strategy = self.selector.select_strategy()
            
            logger.info(
                f"Evaluations: {evaluations_used}/{budget}, Using strategy: {strategy}"
            )
            
            # Run strategy for a batch
            batch_size = min(50, budget - evaluations_used)
            batch_results = self._run_strategy_batch(strategy, batch_size)
            
            # Track results
            self.history.extend(batch_results)
            evaluations_used += len(batch_results)
            
            # Compute reward (improvement in best fitness)
            prev_best = self.best_fitness
            for arch, fitness in batch_results:
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_architecture = arch
            
            improvement = self.best_fitness - prev_best
            
            # Update selector
            self.selector.update(strategy, reward=improvement)
            
            # Checkpoint
            if evaluations_used % checkpoint_interval == 0:
                stats = self.selector.get_statistics()
                logger.info(
                    f"Progress: {evaluations_used}/{budget}, "
                    f"Best fitness: {self.best_fitness:.4f}"
                )
                logger.info(f"Strategy stats: {stats}")
        
        # Final summary
        stats = self.selector.get_statistics()
        best_strategy = self.selector.get_best_strategy()
        
        logger.info(f"Search complete!")
        logger.info(f"Best strategy: {best_strategy}")
        logger.info(f"Final statistics: {stats}")
        logger.info(f"Best fitness: {self.best_fitness:.4f}")
        
        return self.best_architecture
    
    def _run_strategy_batch(
        self, strategy: str, batch_size: int
    ) -> List[tuple]:
        """
        Run a strategy for a batch of evaluations.
        
        Args:
            strategy: Strategy name
            batch_size: Number of evaluations
        
        Returns:
            List of (architecture, fitness) tuples
        """
        results = []
        
        # Create appropriate optimizer
        config = self.strategy_configs[strategy]
        
        if strategy == "random":
            # Random sampling
            for _ in range(batch_size):
                arch = self.search_space.sample()
                fitness = self.evaluator(arch)
                results.append((arch, fitness))
        
        elif strategy in ["ga", "genetic"]:
            # Genetic algorithm
            from morphml.optimizers import GeneticAlgorithm
            
            # Run for mini-generations
            mini_generations = batch_size // config.get("population_size", 20)
            mini_generations = max(1, mini_generations)
            
            ga = GeneticAlgorithm(
                search_space=self.search_space,
                evaluator=self.evaluator,
                population_size=config.get("population_size", 20),
                num_generations=mini_generations,
                mutation_prob=config.get("mutation_prob", 0.2),
                crossover_prob=config.get("crossover_prob", 0.8),
            )
            
            ga.search()
            
            # Extract results
            results = [(ind.architecture, ind.fitness) for ind in ga.history]
        
        elif strategy in ["bo", "bayesian"]:
            # Bayesian optimization (simplified)
            # Just use best from random for now
            for _ in range(batch_size):
                arch = self.search_space.sample()
                fitness = self.evaluator(arch)
                results.append((arch, fitness))
        
        else:
            logger.warning(f"Unknown strategy {strategy}, using random")
            for _ in range(batch_size):
                arch = self.search_space.sample()
                fitness = self.evaluator(arch)
                results.append((arch, fitness))
        
        return results[:batch_size]  # Ensure exact batch size
    
    def get_search_trajectory(self) -> List[float]:
        """
        Get fitness trajectory over search.
        
        Returns:
            List of best fitness values over time
        """
        trajectory = []
        best_so_far = -float('inf')
        
        for _, fitness in self.history:
            if fitness > best_so_far:
                best_so_far = fitness
            trajectory.append(best_so_far)
        
        return trajectory
