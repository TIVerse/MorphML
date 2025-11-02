"""Hill Climbing optimizer for NAS.

Local search strategy that iteratively improves a single architecture
by exploring its neighborhood through mutations.

Example:
    >>> from morphml.optimizers import HillClimbing
    >>> 
    >>> hc = HillClimbing(search_space=space, max_iterations=100)
    >>> best = hc.optimize(evaluator=my_evaluator)
"""

from typing import Callable, List, Optional

from morphml.core.dsl.search_space import SearchSpace
from morphml.core.graph import GraphMutator, ModelGraph
from morphml.core.search import Individual
from morphml.exceptions import OptimizerError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class HillClimbing:
    """
    Hill Climbing optimizer for Neural Architecture Search.

    Iteratively improves a single architecture by:
    1. Mutating the current architecture
    2. Evaluating the mutated version
    3. Accepting if better
    4. Repeating until no improvement

    Simple but effective for refinement and local optimization.

    Attributes:
        search_space: SearchSpace for initialization
        max_iterations: Maximum number of iterations
        current: Current best individual
        history: List of fitness values over iterations

    Example:
        >>> hc = HillClimbing(
        ...     search_space=space,
        ...     max_iterations=100,
        ...     patience=10
        ... )
        >>> best = hc.optimize(evaluator)
    """

    def __init__(
        self,
        search_space: SearchSpace,
        max_iterations: int = 100,
        patience: int = 10,
        num_mutations: int = 3,
        mutation_rate: float = 0.3,
        **kwargs,
    ):
        """
        Initialize Hill Climbing optimizer.

        Args:
            search_space: SearchSpace for initialization
            max_iterations: Maximum iterations
            patience: Stop if no improvement for N iterations
            num_mutations: Number of mutations per neighbor
            mutation_rate: Mutation rate for GraphMutator
            **kwargs: Additional configuration
        """
        self.search_space = search_space
        self.max_iterations = max_iterations
        self.patience = patience
        self.num_mutations = num_mutations
        self.mutation_rate = mutation_rate

        self.mutator = GraphMutator()
        self.current: Optional[Individual] = None
        self.history: List[float] = []

        logger.info(f"Created HillClimbing: max_iterations={max_iterations}, patience={patience}")

    def optimize(self, evaluator: Callable[[ModelGraph], float]) -> Individual:
        """
        Run hill climbing optimization.

        Args:
            evaluator: Function to evaluate fitness

        Returns:
            Best individual found

        Raises:
            OptimizerError: If optimization fails
        """
        try:
            # Initialize with random architecture
            logger.info("Initializing hill climbing")
            init_graph = self.search_space.sample()
            self.current = Individual(init_graph)
            fitness = evaluator(self.current.graph)
            self.current.set_fitness(fitness)
            self.history.append(fitness)

            logger.info(f"Initial fitness: {fitness:.4f}")

            iterations_without_improvement = 0
            iteration = 0

            while iteration < self.max_iterations:
                iteration += 1

                # Generate neighbor by mutation
                mutated_graph = self.mutator.mutate(
                    self.current.graph,
                    mutation_rate=self.mutation_rate,
                    max_mutations=self.num_mutations,
                )

                # Evaluate neighbor
                neighbor = Individual(mutated_graph)
                neighbor_fitness = evaluator(neighbor.graph)
                neighbor.set_fitness(neighbor_fitness)

                # Accept if better
                if neighbor_fitness > self.current.fitness:
                    self.current = neighbor
                    self.history.append(neighbor_fitness)
                    iterations_without_improvement = 0
                    logger.info(f"Iteration {iteration}: Improved to {neighbor_fitness:.4f}")
                else:
                    self.history.append(self.current.fitness)
                    iterations_without_improvement += 1

                # Check patience
                if iterations_without_improvement >= self.patience:
                    logger.info(f"Stopping: No improvement for {self.patience} iterations")
                    break

                if iteration % 10 == 0:
                    logger.debug(
                        f"Iteration {iteration}/{self.max_iterations}: "
                        f"fitness={self.current.fitness:.4f}"
                    )

            logger.info(f"Hill climbing complete: Best fitness = {self.current.fitness:.4f}")

            return self.current

        except Exception as e:
            logger.error(f"Hill climbing failed: {e}")
            raise OptimizerError(f"Hill climbing optimization failed: {e}") from e

    def get_history(self) -> List[float]:
        """Get optimization history."""
        return self.history

    def reset(self) -> None:
        """Reset optimizer state."""
        self.current = None
        self.history.clear()
        logger.info("Hill climbing reset")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HillClimbing(max_iterations={self.max_iterations}, "
            f"current_fitness={self.current.fitness if self.current else None})"
        )
