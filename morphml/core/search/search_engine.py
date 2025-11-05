"""Base search engine for optimization algorithms.

Provides a unified interface for all search/optimization algorithms
with common functionality for initialization, sampling, and termination.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from morphml.core.dsl.search_space import SearchSpace
from morphml.core.graph import ModelGraph
from morphml.core.search.individual import Individual
from morphml.core.search.population import Population
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class SearchEngine(ABC):
    """
    Base class for all search/optimization algorithms.

    Provides common interface and functionality for:
    - Population initialization
    - Architecture sampling
    - Evolution/search steps
    - History tracking
    - Termination criteria

    Subclass this to implement specific search algorithms
    (genetic algorithms, Bayesian optimization, random search, etc.)
    """

    def __init__(
        self,
        search_space: SearchSpace,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize search engine.

        Args:
            search_space: Search space definition
            config: Configuration dictionary
        """
        self.search_space = search_space
        self.config = config or {}
        self.generation = 0
        self.history: List[Dict[str, Any]] = []
        self.best_individual: Optional[Individual] = None
        self.num_evaluations = 0

        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def initialize_population(self, size: int) -> Population:
        """
        Create initial population.

        Args:
            size: Population size

        Returns:
            Initialized population
        """
        pass

    @abstractmethod
    def step(self, population: Population, evaluator: Callable) -> Population:
        """
        Execute one search iteration.

        Args:
            population: Current population
            evaluator: Function to evaluate fitness

        Returns:
            Updated population after one step
        """
        pass

    def should_stop(self) -> bool:
        """
        Check if search should terminate.

        Returns:
            True if termination criteria met
        """
        max_generations = self.config.get("max_generations", float("inf"))
        max_evaluations = self.config.get("max_evaluations", float("inf"))

        if self.generation >= max_generations:
            logger.info(f"Stopping: reached max generations ({max_generations})")
            return True

        if self.num_evaluations >= max_evaluations:
            logger.info(f"Stopping: reached max evaluations ({max_evaluations})")
            return True

        # Early stopping based on improvement
        patience = self.config.get("early_stopping_patience", 0)
        if patience > 0 and len(self.history) >= patience:
            recent_fitness = [h["best_fitness"] for h in self.history[-patience:]]
            if len(set(recent_fitness)) == 1:  # No improvement
                logger.info(f"Stopping: no improvement for {patience} generations")
                return True

        return False

    def search(
        self,
        evaluator: Callable,
        population_size: int,
        max_generations: int,
        callbacks: Optional[List[Callable]] = None,
    ) -> Individual:
        """
        Main search loop.

        Args:
            evaluator: Function to evaluate fitness
            population_size: Size of population
            max_generations: Maximum number of generations
            callbacks: Optional callbacks to call each generation

        Returns:
            Best individual found

        Example:
            >>> engine = MySearchEngine(search_space)
            >>> best = engine.search(evaluator, population_size=50, max_generations=100)
        """
        self.config["max_generations"] = max_generations
        callbacks = callbacks or []

        logger.info(
            f"Starting search with population_size={population_size}, max_generations={max_generations}"
        )

        # Initialize population
        population = self.initialize_population(population_size)

        # Evaluate initial population
        self._evaluate_population(population, evaluator)

        # Track best
        self._update_best(population)

        # Main search loop
        while not self.should_stop():
            # Evolution/search step
            population = self.step(population, evaluator)

            # Evaluate new individuals
            self._evaluate_population(population, evaluator)

            # Update tracking
            self._update_best(population)
            self._record_history(population)

            # Call callbacks
            for callback in callbacks:
                callback(self, population)

            # Log progress
            if self.generation % 10 == 0:
                stats = self.get_statistics()
                logger.info(
                    f"Generation {self.generation}: "
                    f"best={stats['best_fitness']:.4f}, "
                    f"mean={stats['mean_fitness']:.4f}, "
                    f"evaluations={self.num_evaluations}"
                )

            self.generation += 1

        logger.info(
            f"Search complete after {self.generation} generations, {self.num_evaluations} evaluations"
        )
        logger.info(f"Best fitness: {self.best_individual.fitness:.4f}")

        return self.best_individual

    def _evaluate_population(self, population: Population, evaluator: Callable) -> None:
        """Evaluate all unevaluated individuals."""
        for individual in population.individuals:
            if individual.fitness is None:
                individual.set_fitness(evaluator(individual.graph))
                self.num_evaluations += 1

    def _update_best(self, population: Population) -> None:
        """Update best individual if improvement found."""
        current_best = population.get_best(1)[0]

        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best.clone()
            logger.debug(f"New best: {self.best_individual.fitness:.4f}")

    def _record_history(self, population: Population) -> None:
        """Record generation statistics."""
        entry = {
            "generation": self.generation,
            "best_fitness": population.best_fitness(),
            "mean_fitness": population.average_fitness(),
            "worst_fitness": min(
                ind.fitness for ind in population.individuals if ind.fitness is not None
            ),
            "diversity": population.diversity_metric(),
            "num_evaluations": self.num_evaluations,
        }
        self.history.append(entry)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics.

        Returns:
            Dictionary with statistics
        """
        if not self.history:
            return {
                "generation": 0,
                "num_evaluations": 0,
                "best_fitness": 0.0,
                "mean_fitness": 0.0,
            }

        latest = self.history[-1]
        return {
            "generation": self.generation,
            "num_evaluations": self.num_evaluations,
            "best_fitness": latest["best_fitness"],
            "mean_fitness": latest["mean_fitness"],
            "diversity": latest["diversity"],
        }

    def get_best(self) -> ModelGraph:
        """
        Get best architecture found.

        Returns:
            Best ModelGraph
        """
        if self.best_individual is None:
            raise ValueError("No evaluations performed yet")

        return self.best_individual.graph

    def get_best_fitness(self) -> float:
        """
        Get best fitness found.

        Returns:
            Best fitness value
        """
        if self.best_individual is None:
            raise ValueError("No evaluations performed yet")

        return self.best_individual.fitness

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get complete search history.

        Returns:
            List of generation statistics
        """
        return self.history

    def reset(self) -> None:
        """Reset search engine state."""
        self.generation = 0
        self.history = []
        self.best_individual = None
        self.num_evaluations = 0
        logger.info("Search engine reset")


class RandomSearchEngine(SearchEngine):
    """
    Random search engine (baseline).

    Simply samples random architectures from search space
    without any guided search.

    Example:
        >>> engine = RandomSearchEngine(search_space)
        >>> best = engine.search(evaluator, population_size=1, max_generations=100)
    """

    def initialize_population(self, size: int) -> Population:
        """Create initial random population."""
        individuals = []
        for _ in range(size):
            graph = self.search_space.sample()
            individual = Individual(graph=graph)
            individuals.append(individual)

        return Population(individuals)

    def step(self, population: Population, evaluator: Callable) -> Population:
        """Sample new random individuals."""
        # Replace entire population with new random samples
        return self.initialize_population(len(population))


class GridSearchEngine(SearchEngine):
    """
    Grid search engine.

    Systematically explores all combinations of parameter values.
    Only practical for small discrete search spaces.

    Note: This is a placeholder - full grid search requires
    combinatorial enumeration of the search space.
    """

    def __init__(self, search_space: SearchSpace, config: Optional[Dict[str, Any]] = None):
        """Initialize grid search engine."""
        super().__init__(search_space, config)
        self.grid_iterator = None

    def initialize_population(self, size: int) -> Population:
        """Initialize with first batch from grid."""
        individuals = []
        for _ in range(size):
            graph = self.search_space.sample()
            individual = Individual(graph=graph)
            individuals.append(individual)

        return Population(individuals)

    def step(self, population: Population, evaluator: Callable) -> Population:
        """Get next batch from grid."""
        # Simplified: just sample randomly
        # Full implementation would enumerate combinations
        return self.initialize_population(len(population))
