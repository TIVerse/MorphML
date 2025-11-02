"""Random Search optimizer - baseline for NAS.

Simplest possible search strategy: randomly sample architectures
and return the best one. Serves as a baseline for comparison.

Example:
    >>> from morphml.optimizers import RandomSearch
    >>> from morphml.core.dsl import create_cnn_space
    >>>
    >>> space = create_cnn_space(num_classes=10)
    >>> rs = RandomSearch(search_space=space, num_samples=100)
    >>> best = rs.optimize(evaluator=my_evaluator)
"""

from typing import Callable, List, Optional

from morphml.core.dsl.search_space import SearchSpace
from morphml.core.graph import ModelGraph
from morphml.core.search import Individual
from morphml.exceptions import OptimizerError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class RandomSearch:
    """
    Random Search optimizer - baseline for Neural Architecture Search.

    Randomly samples N architectures from the search space, evaluates
    them all, and returns the best one. Despite its simplicity, often
    surprisingly effective and serves as an important baseline.

    Attributes:
        search_space: SearchSpace to sample from
        num_samples: Number of architectures to evaluate
        evaluated: List of evaluated individuals
        best_individual: Best architecture found

    Example:
        >>> rs = RandomSearch(
        ...     search_space=space,
        ...     num_samples=100,
        ...     allow_duplicates=False
        ... )
        >>> best = rs.optimize(evaluator=evaluate_func)
    """

    def __init__(
        self,
        search_space: SearchSpace,
        num_samples: int = 100,
        allow_duplicates: bool = False,
        **kwargs,
    ):
        """
        Initialize Random Search optimizer.

        Args:
            search_space: SearchSpace to sample architectures from
            num_samples: Number of architectures to sample and evaluate
            allow_duplicates: Whether to allow duplicate architectures
            **kwargs: Additional configuration (for compatibility)
        """
        self.search_space = search_space
        self.num_samples = num_samples
        self.allow_duplicates = allow_duplicates

        self.evaluated: List[Individual] = []
        self.best_individual: Optional[Individual] = None

        logger.info(f"Created RandomSearch: num_samples={num_samples}")

    def optimize(self, evaluator: Callable[[ModelGraph], float]) -> Individual:
        """
        Run random search optimization.

        Args:
            evaluator: Function to evaluate fitness of architectures

        Returns:
            Best individual found

        Raises:
            OptimizerError: If optimization fails

        Example:
            >>> def evaluate(graph):
            ...     return accuracy_score
            >>> best = rs.optimize(evaluate)
        """
        try:
            logger.info(f"Starting random search with {self.num_samples} samples")

            seen_hashes = set()
            sampled = 0

            while sampled < self.num_samples:
                try:
                    # Sample architecture
                    graph = self.search_space.sample()

                    # Check for duplicates
                    if not self.allow_duplicates:
                        graph_hash = graph.hash()
                        if graph_hash in seen_hashes:
                            logger.debug("Skipping duplicate architecture")
                            continue
                        seen_hashes.add(graph_hash)

                    # Create individual and evaluate
                    individual = Individual(graph)
                    fitness = evaluator(graph)
                    individual.set_fitness(fitness)

                    self.evaluated.append(individual)
                    sampled += 1

                    # Track best
                    if self.best_individual is None or fitness > self.best_individual.fitness:
                        self.best_individual = individual
                        logger.info(f"New best: {fitness:.4f} (sample {sampled})")

                    if sampled % 10 == 0:
                        logger.debug(f"Evaluated {sampled}/{self.num_samples} architectures")

                except Exception as e:
                    logger.warning(f"Sample failed: {e}")
                    continue

            # Final results
            logger.info(
                f"Random search complete: " f"Best fitness = {self.best_individual.fitness:.4f}"
            )

            return self.best_individual

        except Exception as e:
            logger.error(f"Random search failed: {e}")
            raise OptimizerError(f"Random search optimization failed: {e}") from e

    def get_all_evaluated(self) -> List[Individual]:
        """
        Get all evaluated individuals.

        Returns:
            List of all evaluated individuals
        """
        return self.evaluated

    def get_best_n(self, n: int = 10) -> List[Individual]:
        """
        Get top N individuals.

        Args:
            n: Number of individuals to return

        Returns:
            List of best individuals sorted by fitness
        """
        sorted_inds = sorted(self.evaluated, key=lambda x: x.fitness or 0, reverse=True)
        return sorted_inds[:n]

    def reset(self) -> None:
        """Reset optimizer state."""
        self.evaluated.clear()
        self.best_individual = None
        logger.info("Random search reset")

    def __repr__(self) -> str:
        """String representation."""
        return f"RandomSearch(num_samples={self.num_samples}, " f"evaluated={len(self.evaluated)})"
