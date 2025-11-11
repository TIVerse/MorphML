"""Genetic Algorithm optimizer for neural architecture search.

A complete implementation of evolutionary NAS using genetic algorithms.

Example:
    >>> from morphml.optimizers import GeneticAlgorithm
    >>> from morphml.core.dsl import create_cnn_space
    >>>
    >>> # Define search space
    >>> space = create_cnn_space(num_classes=10)
    >>>
    >>> # Create optimizer
    >>> ga = GeneticAlgorithm(
    ...     search_space=space,
    ...     population_size=50,
    ...     num_generations=100,
    ...     mutation_rate=0.2,
    ...     crossover_rate=0.8,
    ...     elitism=5
    ... )
    >>>
    >>> # Run optimization
    >>> best_individual = ga.optimize(evaluator=my_evaluator)
"""

from typing import Any, Callable, Dict, List, Optional

from morphml.core.dsl.search_space import SearchSpace
from morphml.core.graph import GraphMutator, ModelGraph
from morphml.core.search import Individual, Population
from morphml.exceptions import OptimizerError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class GeneticAlgorithm:
    """
    Genetic Algorithm optimizer for neural architecture search.

    Implements a complete evolutionary algorithm with:
    - Population initialization from search space
    - Selection (tournament, roulette, rank, random)
    - Crossover (graph-level)
    - Mutation (using GraphMutator)
    - Elitism preservation
    - Convergence tracking
    - History recording

    Attributes:
        search_space: SearchSpace to sample from
        population: Current population
        mutator: GraphMutator for mutations
        config: Algorithm configuration
        history: Optimization history
        best_individual: Best individual found so far

    Example:
        >>> ga = GeneticAlgorithm(
        ...     search_space=space,
        ...     population_size=50,
        ...     num_generations=100
        ... )
        >>> result = ga.optimize(evaluator=evaluate_func)
    """

    def __init__(
        self,
        search_space: SearchSpace,
        population_size: int = 50,
        num_generations: int = 100,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.8,
        elitism: int = 5,
        selection_method: str = "tournament",
        tournament_size: int = 3,
        max_mutations: int = 3,
        early_stopping_patience: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Initialize Genetic Algorithm optimizer.

        Args:
            search_space: SearchSpace to sample architectures from
            population_size: Number of individuals in population
            num_generations: Maximum number of generations
            mutation_rate: Probability of mutating an offspring
            crossover_rate: Probability of crossover
            elitism: Number of best individuals to preserve
            selection_method: Selection strategy ('tournament', 'roulette', 'rank')
            tournament_size: Size of tournament for tournament selection
            max_mutations: Maximum mutations per individual
            early_stopping_patience: Stop if no improvement for N generations
            **kwargs: Additional configuration
        """
        self.search_space = search_space
        self.config = {
            "population_size": population_size,
            "num_generations": num_generations,
            "mutation_rate": mutation_rate,
            "crossover_rate": crossover_rate,
            "elitism": elitism,
            "selection_method": selection_method,
            "tournament_size": tournament_size,
            "max_mutations": max_mutations,
            "early_stopping_patience": early_stopping_patience,
            **kwargs,
        }

        # Initialize components
        self.population = Population(max_size=population_size, elitism=elitism)
        self.mutator = GraphMutator()

        # State tracking
        self.history: List[Dict[str, Any]] = []
        self.best_individual: Optional[Individual] = None
        self._generations_without_improvement = 0
        self._initialized = False

        logger.info(
            f"Created GeneticAlgorithm: "
            f"pop_size={population_size}, "
            f"generations={num_generations}, "
            f"mutation_rate={mutation_rate}"
        )

    def initialize_population(self) -> None:
        """
        Initialize population by sampling from search space.

        Creates population_size individuals by sampling random
        architectures from the search space.
        """
        logger.info(f"Initializing population of size {self.config['population_size']}")

        self.population.clear()

        for i in range(self.config["population_size"]):
            try:
                graph = self.search_space.sample()
                individual = Individual(graph)
                self.population.add(individual)

                if (i + 1) % 10 == 0:
                    logger.debug(
                        f"Initialized {i + 1}/{self.config['population_size']} individuals"
                    )

            except Exception as e:
                logger.warning(f"Failed to sample individual {i}: {e}")
                continue

        self._initialized = True
        logger.info(f"Population initialized with {self.population.size()} individuals")

    def evaluate_population(self, evaluator: Callable[[ModelGraph], float]) -> None:
        """
        Evaluate all unevaluated individuals in the population.

        Args:
            evaluator: Function that takes ModelGraph and returns fitness score

        Example:
            >>> def my_evaluator(graph):
            ...     # Your evaluation logic
            ...     return accuracy_score
            >>> ga.evaluate_population(my_evaluator)
        """
        unevaluated = self.population.get_unevaluated()

        if not unevaluated:
            logger.debug("No unevaluated individuals")
            return

        logger.info(f"Evaluating {len(unevaluated)} individuals")

        for i, individual in enumerate(unevaluated):
            try:
                # Evaluate architecture
                fitness = evaluator(individual.graph)
                individual.set_fitness(fitness)

                # Track best
                if (
                    self.best_individual is None
                    or self.best_individual.fitness is None
                    or fitness > self.best_individual.fitness
                ):
                    self.best_individual = individual
                    self._generations_without_improvement = 0
                    logger.info(f"New best fitness: {fitness:.4f}")

                if (i + 1) % 10 == 0:
                    logger.debug(f"Evaluated {i + 1}/{len(unevaluated)} individuals")

            except Exception as e:
                logger.error(f"Evaluation failed for individual {individual.id[:12]}: {e}")
                # Assign low fitness on failure
                individual.set_fitness(0.0)

    def select_parents(self, n: int) -> List[Individual]:
        """
        Select parent individuals for breeding.

        Args:
            n: Number of parents to select

        Returns:
            List of selected parent individuals
        """
        return self.population.select(
            n=n,
            method=self.config["selection_method"],
            k=self.config.get("tournament_size", 3),
        )

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Perform crossover between two parents.

        Uses single-point crossover to combine parent graphs.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Offspring individual
        """
        from morphml.core.graph.mutations import crossover as graph_crossover
        import random

        # Perform graph crossover
        offspring_graph1, offspring_graph2 = graph_crossover(parent1.graph, parent2.graph)
        
        # Randomly select one of the two offspring
        selected_graph = random.choice([offspring_graph1, offspring_graph2])
        
        # Create new individual
        offspring = Individual(selected_graph)
        offspring.parent_ids = [parent1.id, parent2.id]
        offspring.metadata["crossover"] = "single_point"

        return offspring

    def mutate(self, individual: Individual) -> Individual:
        """
        Mutate an individual's architecture.

        Args:
            individual: Individual to mutate

        Returns:
            Mutated individual (new instance)
        """
        mutated_graph = self.mutator.mutate(
            individual.graph,
            mutation_rate=self.config["mutation_rate"],
            max_mutations=self.config.get("max_mutations", 3),
        )

        mutated_individual = Individual(mutated_graph, parent_ids=[individual.id])

        return mutated_individual

    def generate_offspring(self, num_offspring: int) -> List[Individual]:
        """
        Generate offspring through selection, crossover, and mutation.

        Args:
            num_offspring: Number of offspring to generate

        Returns:
            List of offspring individuals
        """
        offspring: List[Individual] = []

        while len(offspring) < num_offspring:
            # Select parents
            parents = self.select_parents(n=2)

            if len(parents) < 2:
                logger.warning("Not enough parents for crossover")
                break

            # Crossover
            import random

            if random.random() < self.config["crossover_rate"]:
                child = self.crossover(parents[0], parents[1])
            else:
                # No crossover, just clone
                child = random.choice(parents).clone(keep_fitness=False)

            # Mutation
            if random.random() < self.config["mutation_rate"]:
                child = self.mutate(child)

            offspring.append(child)

        return offspring

    def evolve_generation(self) -> None:
        """
        Evolve one generation.

        Steps:
        1. Generate offspring
        2. Add to population
        3. Trim to max size (keeps elite + best)
        4. Advance generation
        """
        # Generate offspring
        num_offspring = self.config["population_size"] - self.config["elitism"]
        offspring = self.generate_offspring(num_offspring)

        logger.debug(f"Generated {len(offspring)} offspring")

        # Add offspring to population
        self.population.add_many(offspring)

        # Trim to max size
        self.population.trim()

        # Advance generation
        self.population.next_generation()

    def check_convergence(self) -> bool:
        """
        Check if optimization should stop.

        Returns:
            True if converged (should stop), False otherwise
        """
        # Check generation limit
        if self.population.generation >= self.config["num_generations"]:
            logger.info(f"Reached max generations: {self.config['num_generations']}")
            return True

        # Check early stopping
        patience = self.config.get("early_stopping_patience")
        if patience and self._generations_without_improvement >= patience:
            logger.info(
                f"Early stopping: No improvement for "
                f"{self._generations_without_improvement} generations"
            )
            return True

        return False

    def optimize(
        self,
        evaluator: Callable[[ModelGraph], float],
        callback: Optional[Callable[[int, Population], None]] = None,
    ) -> Individual:
        """
        Run the genetic algorithm optimization.

        Args:
            evaluator: Function to evaluate fitness of architectures
            callback: Optional callback function called each generation
                     callback(generation: int, population: Population)

        Returns:
            Best individual found

        Raises:
            OptimizerError: If optimization fails

        Example:
            >>> def evaluate(graph):
            ...     # Your evaluation
            ...     return accuracy
            >>>
            >>> def progress_callback(gen, pop):
            ...     stats = pop.get_statistics()
            ...     print(f"Gen {gen}: {stats['best_fitness']:.4f}")
            >>>
            >>> best = ga.optimize(evaluate, callback=progress_callback)
        """
        try:
            # Initialize if needed
            if not self._initialized:
                self.initialize_population()

            # Evaluate initial population
            logger.info("Evaluating initial population")
            self.evaluate_population(evaluator)

            # Record initial stats
            self._record_generation()

            # Evolution loop
            while not self.check_convergence():
                gen = self.population.generation

                logger.info(f"Generation {gen + 1}/{self.config['num_generations']}")

                # Evolve
                self.evolve_generation()

                # Evaluate new individuals
                self.evaluate_population(evaluator)

                # Record statistics
                self._record_generation()

                # Track improvement
                current_best = self.population.get_best(n=1)[0]
                if (
                    self.best_individual
                    and self.best_individual.fitness is not None
                    and current_best.fitness is not None
                    and current_best.fitness <= self.best_individual.fitness
                ):
                    self._generations_without_improvement += 1
                else:
                    self._generations_without_improvement = 0

                # Callback
                if callback:
                    callback(self.population.generation, self.population)

            # Final results
            logger.info("Optimization complete")
            stats = self.population.get_statistics()
            logger.info(
                f"Final: Best={stats['best_fitness']:.4f}, "
                f"Mean={stats['mean_fitness']:.4f}, "
                f"Generation={self.population.generation}"
            )

            return self.best_individual or self.population.get_best(n=1)[0]

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise OptimizerError(f"Genetic algorithm optimization failed: {e}") from e

    def _record_generation(self) -> None:
        """Record current generation statistics to history."""
        stats = self.population.get_statistics()
        stats["diversity"] = self.population.get_diversity()
        stats["best_individual_id"] = self.best_individual.id[:12] if self.best_individual else None
        self.history.append(stats)

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get optimization history.

        Returns:
            List of dictionaries with statistics for each generation
        """
        return self.history

    def get_best_n(self, n: int = 10) -> List[Individual]:
        """
        Get top N individuals from final population.

        Args:
            n: Number of individuals to return

        Returns:
            List of best individuals
        """
        return self.population.get_best(n=n)

    def reset(self) -> None:
        """Reset optimizer state."""
        self.population.clear()
        self.history.clear()
        self.best_individual = None
        self._generations_without_improvement = 0
        self._initialized = False
        logger.info("Optimizer reset")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GeneticAlgorithm("
            f"pop_size={self.config['population_size']}, "
            f"generations={self.config['num_generations']}, "
            f"mutation_rate={self.config['mutation_rate']}, "
            f"current_gen={self.population.generation})"
        )
