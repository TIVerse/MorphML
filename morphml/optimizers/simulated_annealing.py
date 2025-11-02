"""Simulated Annealing optimizer for NAS.

Probabilistic hill climbing that accepts worse solutions with
decreasing probability to escape local optima.
"""

import math
import random
from typing import Callable, List, Optional

from morphml.core.dsl.search_space import SearchSpace
from morphml.core.graph import GraphMutator, ModelGraph
from morphml.core.search import Individual
from morphml.exceptions import OptimizerError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class SimulatedAnnealing:
    """
    Simulated Annealing optimizer.

    Accepts worse solutions with probability based on temperature
    schedule, enabling escape from local optima.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        max_iterations: int = 1000,
        initial_temp: float = 100.0,
        final_temp: float = 0.1,
        cooling_rate: float = 0.95,
        cooling_schedule: str = "exponential",
        num_mutations: int = 3,
        mutation_rate: float = 0.3,
        **kwargs,
    ):
        """Initialize SA optimizer."""
        self.search_space = search_space
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.cooling_schedule = cooling_schedule
        self.num_mutations = num_mutations
        self.mutation_rate = mutation_rate

        self.mutator = GraphMutator()
        self.current: Optional[Individual] = None
        self.best: Optional[Individual] = None
        self.history: List[dict] = []

        logger.info(f"Created SimulatedAnnealing: T0={initial_temp}, Tf={final_temp}")

    def get_temperature(self, iteration: int) -> float:
        """Calculate temperature at iteration."""
        progress = iteration / self.max_iterations

        if self.cooling_schedule == "exponential":
            return self.initial_temp * (self.cooling_rate**iteration)
        elif self.cooling_schedule == "linear":
            return self.initial_temp - (self.initial_temp - self.final_temp) * progress
        elif self.cooling_schedule == "logarithmic":
            return self.initial_temp / (1 + math.log(1 + iteration))
        else:
            return self.initial_temp * (self.cooling_rate**iteration)

    def acceptance_probability(self, old_fitness: float, new_fitness: float, temp: float) -> float:
        """Calculate acceptance probability."""
        if new_fitness > old_fitness:
            return 1.0

        if temp <= 0:
            return 0.0

        delta = new_fitness - old_fitness
        return math.exp(delta / temp)

    def optimize(self, evaluator: Callable[[ModelGraph], float]) -> Individual:
        """Run simulated annealing."""
        try:
            # Initialize
            logger.info("Initializing simulated annealing")
            init_graph = self.search_space.sample()
            self.current = Individual(init_graph)
            fitness = evaluator(self.current.graph)
            self.current.set_fitness(fitness)
            self.best = self.current

            self.history.append(
                {
                    "iteration": 0,
                    "fitness": fitness,
                    "temperature": self.initial_temp,
                    "accepted": True,
                }
            )

            logger.info(f"Initial fitness: {fitness:.4f}")

            for iteration in range(1, self.max_iterations + 1):
                # Get temperature
                temp = self.get_temperature(iteration)

                # Generate neighbor
                mutated_graph = self.mutator.mutate(
                    self.current.graph,
                    mutation_rate=self.mutation_rate,
                    max_mutations=self.num_mutations,
                )

                neighbor = Individual(mutated_graph)
                neighbor_fitness = evaluator(neighbor.graph)
                neighbor.set_fitness(neighbor_fitness)

                # Accept or reject
                prob = self.acceptance_probability(self.current.fitness, neighbor_fitness, temp)

                accepted = random.random() < prob

                if accepted:
                    self.current = neighbor

                    # Update best
                    if neighbor_fitness > self.best.fitness:
                        self.best = neighbor
                        logger.info(
                            f"Iteration {iteration}: New best {neighbor_fitness:.4f} "
                            f"(T={temp:.3f})"
                        )

                self.history.append(
                    {
                        "iteration": iteration,
                        "fitness": neighbor_fitness,
                        "temperature": temp,
                        "accepted": accepted,
                    }
                )

                if iteration % 100 == 0:
                    logger.debug(
                        f"Iteration {iteration}/{self.max_iterations}: "
                        f"Current={self.current.fitness:.4f}, "
                        f"Best={self.best.fitness:.4f}, "
                        f"T={temp:.3f}"
                    )

            logger.info(f"SA complete: Best fitness = {self.best.fitness:.4f}")
            return self.best

        except Exception as e:
            logger.error(f"Simulated annealing failed: {e}")
            raise OptimizerError(f"SA optimization failed: {e}") from e

    def get_history(self) -> List[dict]:
        """Get optimization history."""
        return self.history

    def get_acceptance_rate(self) -> float:
        """Calculate overall acceptance rate."""
        if not self.history:
            return 0.0
        accepted = sum(1 for h in self.history if h["accepted"])
        return accepted / len(self.history)

    def reset(self) -> None:
        """Reset optimizer."""
        self.current = None
        self.best = None
        self.history.clear()
        logger.info("SA reset")

    def __repr__(self) -> str:
        return (
            f"SimulatedAnnealing(T0={self.initial_temp}, "
            f"Tf={self.final_temp}, "
            f"iterations={self.max_iterations})"
        )
