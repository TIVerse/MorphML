"""NSGA-II - Non-dominated Sorting Genetic Algorithm II.

Multi-objective optimization using Pareto dominance and crowding distance.
"""

import random
from typing import Callable, Dict, List, Optional, Tuple

from morphml.core.dsl.search_space import SearchSpace
from morphml.core.graph import GraphMutator, ModelGraph
from morphml.core.search import Individual, Population
from morphml.exceptions import OptimizerError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class NSGA2:
    """
    NSGA-II for multi-objective NAS.

    Optimizes multiple objectives simultaneously using:
    - Non-dominated sorting
    - Crowding distance
    - Pareto front preservation
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objectives: List[str],
        population_size: int = 100,
        num_generations: int = 100,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.8,
        **kwargs,
    ):
        """Initialize NSGA-II."""
        self.search_space = search_space
        self.objectives = objectives
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.population = Population(max_size=population_size * 2, elitism=0)
        self.mutator = GraphMutator()
        self.pareto_front: List[Individual] = []
        self.history: List[dict] = []

        logger.info(f"Created NSGA2: objectives={objectives}, pop={population_size}")

    def dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Check if ind1 dominates ind2."""
        better_in_any = False

        for obj in self.objectives:
            val1 = ind1.get_metric(obj, 0.0)
            val2 = ind2.get_metric(obj, 0.0)

            if val1 < val2:
                return False  # Worse in this objective
            if val1 > val2:
                better_in_any = True

        return better_in_any

    def fast_non_dominated_sort(self, individuals: List[Individual]) -> List[List[Individual]]:
        """Perform fast non-dominated sorting."""
        fronts: List[List[Individual]] = [[]]
        domination_count: Dict[str, int] = {}
        dominated_solutions: Dict[str, List[Individual]] = {}

        # Initialize
        for ind in individuals:
            ind_id = ind.id
            domination_count[ind_id] = 0
            dominated_solutions[ind_id] = []

        # Find domination relationships
        for i, ind1 in enumerate(individuals):
            for ind2 in individuals[i + 1 :]:
                if self.dominates(ind1, ind2):
                    dominated_solutions[ind1.id].append(ind2)
                    domination_count[ind2.id] += 1
                elif self.dominates(ind2, ind1):
                    dominated_solutions[ind2.id].append(ind1)
                    domination_count[ind1.id] += 1

            # If not dominated by anyone, belongs to first front
            if domination_count[ind1.id] == 0:
                fronts[0].append(ind1)

        # Build subsequent fronts
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for ind in fronts[current_front]:
                for dominated in dominated_solutions[ind.id]:
                    domination_count[dominated.id] -= 1
                    if domination_count[dominated.id] == 0:
                        next_front.append(dominated)

            current_front += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return [f for f in fronts if f]  # Remove empty fronts

    def calculate_crowding_distance(self, front: List[Individual]) -> Dict[str, float]:
        """Calculate crowding distance for individuals in front."""
        distances: Dict[str, float] = {ind.id: 0.0 for ind in front}

        if len(front) <= 2:
            for ind in front:
                distances[ind.id] = float("inf")
            return distances

        # For each objective
        for obj in self.objectives:
            # Sort by objective value
            sorted_front = sorted(front, key=lambda x: x.get_metric(obj, 0.0))

            # Boundary points get infinite distance
            distances[sorted_front[0].id] = float("inf")
            distances[sorted_front[-1].id] = float("inf")

            # Get objective range
            obj_min = sorted_front[0].get_metric(obj, 0.0)
            obj_max = sorted_front[-1].get_metric(obj, 0.0)
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            # Calculate crowding distance
            for i in range(1, len(sorted_front) - 1):
                prev_val = sorted_front[i - 1].get_metric(obj, 0.0)
                next_val = sorted_front[i + 1].get_metric(obj, 0.0)
                distances[sorted_front[i].id] += (next_val - prev_val) / obj_range

        return distances

    def environmental_selection(self, individuals: List[Individual]) -> List[Individual]:
        """Select next generation using NSGA-II selection."""
        # Non-dominated sorting
        fronts = self.fast_non_dominated_sort(individuals)

        # Select individuals
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= self.population_size:
                selected.extend(front)
            else:
                # Calculate crowding distance for this front
                distances = self.calculate_crowding_distance(front)

                # Sort by crowding distance (descending)
                front_sorted = sorted(front, key=lambda x: distances[x.id], reverse=True)

                # Add individuals until population is full
                remaining = self.population_size - len(selected)
                selected.extend(front_sorted[:remaining])
                break

        return selected

    def initialize_population(self) -> None:
        """Initialize population."""
        logger.info(f"Initializing population of size {self.population_size}")

        for i in range(self.population_size):
            try:
                graph = self.search_space.sample()
                individual = Individual(graph)
                self.population.add(individual)
            except Exception as e:
                logger.warning(f"Failed to sample individual {i}: {e}")
                continue

    def evaluate_population(self, evaluator: Callable[[ModelGraph], Dict[str, float]]) -> None:
        """Evaluate population on multiple objectives."""
        unevaluated = self.population.get_unevaluated()

        if not unevaluated:
            return

        logger.info(f"Evaluating {len(unevaluated)} individuals")

        for ind in unevaluated:
            try:
                # Evaluator returns dict of objective values
                metrics = evaluator(ind.graph)

                # Set fitness as primary objective
                if self.objectives:
                    primary_fitness = metrics.get(self.objectives[0], 0.0)
                    ind.set_fitness(primary_fitness, **metrics)
                else:
                    ind.set_fitness(sum(metrics.values()) / len(metrics), **metrics)

            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                ind.set_fitness(0.0)

    def generate_offspring(self, parents: List[Individual]) -> List[Individual]:
        """Generate offspring through crossover and mutation."""
        offspring = []

        while len(offspring) < len(parents):
            # Select parents
            parent1, parent2 = random.sample(parents, 2)

            # Crossover
            if random.random() < self.crossover_rate:
                child = parent1.clone(keep_fitness=False)
                child.parent_ids = [parent1.id, parent2.id]
            else:
                child = random.choice([parent1, parent2]).clone(keep_fitness=False)

            # Mutation
            if random.random() < self.mutation_rate:
                mutated_graph = self.mutator.mutate(child.graph)
                child = Individual(mutated_graph, parent_ids=child.parent_ids)

            offspring.append(child)

        return offspring

    def optimize(
        self,
        evaluator: Callable[[ModelGraph], Dict[str, float]],
        callback: Optional[Callable[[int, List[Individual]], None]] = None,
    ) -> List[Individual]:
        """Run NSGA-II optimization."""
        try:
            # Initialize
            self.initialize_population()
            self.evaluate_population(evaluator)

            # Evolution loop
            for gen in range(self.num_generations):
                logger.info(f"Generation {gen + 1}/{self.num_generations}")

                # Generate offspring
                parents = list(self.population.individuals)
                offspring = self.generate_offspring(parents)

                # Evaluate offspring
                self.population.add_many(offspring)
                self.evaluate_population(evaluator)

                # Environmental selection
                combined = list(self.population.individuals)
                selected = self.environmental_selection(combined)

                # Update population
                self.population.clear()
                self.population.add_many(selected)

                # Update Pareto front
                fronts = self.fast_non_dominated_sort(selected)
                self.pareto_front = fronts[0] if fronts else []

                # Record stats
                stats = {
                    "generation": gen + 1,
                    "pareto_size": len(self.pareto_front),
                    "population_size": len(selected),
                }
                self.history.append(stats)

                logger.info(f"Gen {gen + 1}: Pareto front size = {len(self.pareto_front)}")

                # Callback
                if callback:
                    callback(gen + 1, self.pareto_front)

            logger.info(f"NSGA-II complete: Pareto front size = {len(self.pareto_front)}")
            return self.pareto_front

        except Exception as e:
            logger.error(f"NSGA-II optimization failed: {e}")
            raise OptimizerError(f"NSGA-II optimization failed: {e}") from e

    def get_pareto_front(self) -> List[Individual]:
        """Get current Pareto front."""
        return self.pareto_front

    def get_history(self) -> List[dict]:
        """Get optimization history."""
        return self.history

    def reset(self) -> None:
        """Reset optimizer."""
        self.population.clear()
        self.pareto_front.clear()
        self.history.clear()
        logger.info("NSGA-II reset")

    def __repr__(self) -> str:
        return (
            f"NSGA2(objectives={self.objectives}, "
            f"pop={self.population_size}, "
            f"pareto_size={len(self.pareto_front)})"
        )
