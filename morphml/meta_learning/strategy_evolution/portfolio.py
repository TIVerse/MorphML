"""Portfolio optimization - run multiple strategies in parallel.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List

from morphml.core.dsl import SearchSpace
from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class PortfolioOptimizer:
    """
    Run multiple optimization strategies in parallel.

    Allocates computational budget across different strategies
    and combines their results.

    Args:
        search_space: Search space definition
        evaluator: Architecture evaluation function
        strategies: Dict mapping strategy name to configuration
        budget_allocation: Budget allocation strategy
            - 'equal': Equal budget to each
            - 'performance': Allocate based on past performance

    Example:
        >>> strategies = {
        ...     'random': {'num_samples': 100},
        ...     'ga': {'population_size': 50, 'num_generations': 10},
        ...     'hc': {'num_iterations': 200}
        ... }
        >>>
        >>> portfolio = PortfolioOptimizer(
        ...     search_space=space,
        ...     evaluator=evaluator,
        ...     strategies=strategies
        ... )
        >>>
        >>> best_archs = portfolio.optimize(total_budget=500)
    """

    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: Callable[[ModelGraph], float],
        strategies: Dict[str, Dict[str, Any]],
        budget_allocation: str = "equal",
        parallel: bool = False,
    ):
        """Initialize portfolio optimizer."""
        self.search_space = search_space
        self.evaluator = evaluator
        self.strategies = strategies
        self.budget_allocation = budget_allocation
        self.parallel = parallel

        self.results = {}

        logger.info(
            f"Initialized PortfolioOptimizer with {len(strategies)} strategies: "
            f"{list(strategies.keys())}"
        )

    def optimize(self, total_budget: int = 500, top_k: int = 10) -> List[ModelGraph]:
        """
        Run portfolio optimization.

        Args:
            total_budget: Total evaluation budget
            top_k: Number of best architectures to return

        Returns:
            List of best architectures across all strategies
        """
        logger.info(f"Starting portfolio optimization with budget={total_budget}")

        # Allocate budget
        budgets = self._allocate_budget(total_budget)

        logger.info(f"Budget allocation: {budgets}")

        # Run strategies
        if self.parallel:
            all_results = self._run_parallel(budgets)
        else:
            all_results = self._run_sequential(budgets)

        # Combine and sort
        combined = []
        for _strategy, results in all_results.items():
            combined.extend(results)

        # Sort by fitness
        combined.sort(key=lambda x: x[1], reverse=True)

        # Extract top-k architectures
        best_architectures = [arch for arch, _ in combined[:top_k]]

        logger.info(f"Portfolio optimization complete! " f"Best fitness: {combined[0][1]:.4f}")

        return best_architectures

    def _allocate_budget(self, total_budget: int) -> Dict[str, int]:
        """
        Allocate budget across strategies.

        Args:
            total_budget: Total budget

        Returns:
            Dict mapping strategy to allocated budget
        """
        budgets = {}

        if self.budget_allocation == "equal":
            # Equal allocation
            budget_per_strategy = total_budget // len(self.strategies)
            for strategy in self.strategies:
                budgets[strategy] = budget_per_strategy

        elif self.budget_allocation == "performance":
            # Allocate based on past performance
            # (simplified - would use historical data in practice)
            budget_per_strategy = total_budget // len(self.strategies)
            for strategy in self.strategies:
                budgets[strategy] = budget_per_strategy

        else:
            raise ValueError(f"Unknown allocation: {self.budget_allocation}")

        return budgets

    def _run_sequential(self, budgets: Dict[str, int]) -> Dict[str, List[tuple]]:
        """Run strategies sequentially."""
        all_results = {}

        for strategy, budget in budgets.items():
            logger.info(f"Running {strategy} with budget {budget}")

            results = self._run_strategy(strategy, budget)
            all_results[strategy] = results

            best_fitness = max(f for _, f in results) if results else 0.0
            logger.info(
                f"{strategy} complete: {len(results)} evaluations, " f"best={best_fitness:.4f}"
            )

        return all_results

    def _run_parallel(self, budgets: Dict[str, int]) -> Dict[str, List[tuple]]:
        """Run strategies in parallel."""
        all_results = {}

        with ThreadPoolExecutor(max_workers=len(self.strategies)) as executor:
            # Submit all strategies
            futures = {
                executor.submit(self._run_strategy, strategy, budget): strategy
                for strategy, budget in budgets.items()
            }

            # Collect results
            for future in as_completed(futures):
                strategy = futures[future]
                try:
                    results = future.result()
                    all_results[strategy] = results

                    best_fitness = max(f for _, f in results) if results else 0.0
                    logger.info(
                        f"{strategy} complete: {len(results)} evaluations, "
                        f"best={best_fitness:.4f}"
                    )
                except Exception as e:
                    logger.error(f"{strategy} failed: {e}")
                    all_results[strategy] = []

        return all_results

    def _run_strategy(self, strategy: str, budget: int) -> List[tuple]:
        """
        Run a single strategy.

        Args:
            strategy: Strategy name
            budget: Evaluation budget

        Returns:
            List of (architecture, fitness) tuples
        """
        results = []
        config = self.strategies[strategy]

        if strategy == "random":
            # Random search
            for _ in range(budget):
                arch = self.search_space.sample()
                fitness = self.evaluator(arch)
                results.append((arch, fitness))

        elif strategy in ["ga", "genetic"]:
            # Genetic algorithm
            from morphml.optimizers import GeneticAlgorithm

            pop_size = config.get("population_size", 20)
            num_gen = budget // pop_size

            ga = GeneticAlgorithm(
                search_space=self.search_space,
                evaluator=self.evaluator,
                population_size=pop_size,
                num_generations=num_gen,
            )

            ga.search()
            results = [(ind.architecture, ind.fitness) for ind in ga.history]

        else:
            # Fallback to random
            for _ in range(budget):
                arch = self.search_space.sample()
                fitness = self.evaluator(arch)
                results.append((arch, fitness))

        return results
