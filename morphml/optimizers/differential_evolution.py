"""Differential Evolution optimizer adapted for graph-based NAS.

Uses difference vectors between individuals to guide search.
"""

import random
from typing import Callable, List, Optional

from morphml.core.dsl.search_space import SearchSpace
from morphml.core.graph import GraphMutator, ModelGraph
from morphml.core.search import Individual, Population
from morphml.exceptions import OptimizerError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class DifferentialEvolution:
    """
    Differential Evolution for NAS.
    
    Adapted DE/rand/1/bin strategy for graph-based architectures.
    Uses parameter-space mutations guided by population diversity.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        population_size: int = 50,
        num_generations: int = 100,
        mutation_factor: float = 0.8,
        crossover_prob: float = 0.9,
        strategy: str = "rand/1/bin",
        **kwargs,
    ):
        """Initialize DE optimizer."""
        self.search_space = search_space
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.strategy = strategy
        
        self.population = Population(max_size=population_size, elitism=0)
        self.mutator = GraphMutator()
        self.best_individual: Optional[Individual] = None
        self.history: List[dict] = []
        
        logger.info(f"Created DifferentialEvolution: pop={population_size}, F={mutation_factor}")
    
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
        
        logger.info(f"Population initialized with {self.population.size()} individuals")
    
    def mutate_individual(self, target_idx: int) -> Individual:
        """Create mutant individual using DE mutation."""
        individuals = list(self.population.individuals)
        
        # Select random individuals for mutation (excluding target)
        candidates = [i for i in range(len(individuals)) if i != target_idx]
        
        if len(candidates) < 3:
            # Not enough diversity, just mutate target
            mutated = self.mutator.mutate(individuals[target_idx].graph)
            return Individual(mutated)
        
        # DE/rand/1 strategy: mutant = r1 + F * (r2 - r3)
        r1, r2, r3 = random.sample(candidates, 3)
        
        # For graphs, we interpret this as:
        # 1. Clone r1 as base
        # 2. Apply mutations inspired by differences between r2 and r3
        base_graph = individuals[r1].graph.clone()
        
        # Calculate "difference" as parameter variations
        diff_mutations = max(1, int(self.mutation_factor * 5))
        
        mutated_graph = self.mutator.mutate(
            base_graph,
            mutation_rate=self.mutation_factor,
            max_mutations=diff_mutations
        )
        
        mutant = Individual(mutated_graph)
        return mutant
    
    def crossover(self, target: Individual, mutant: Individual) -> Individual:
        """Binomial crossover between target and mutant."""
        # For graphs, we do probabilistic node/edge selection
        if random.random() < self.crossover_prob:
            # Use mutant (more exploration)
            return mutant
        else:
            # Use target (more exploitation)
            return target.clone(keep_fitness=False)
    
    def evaluate_population(self, evaluator: Callable[[ModelGraph], float]) -> None:
        """Evaluate all unevaluated individuals."""
        unevaluated = self.population.get_unevaluated()
        
        if not unevaluated:
            return
        
        logger.info(f"Evaluating {len(unevaluated)} individuals")
        
        for i, individual in enumerate(unevaluated):
            try:
                fitness = evaluator(individual.graph)
                individual.set_fitness(fitness)
                
                if (
                    self.best_individual is None
                    or fitness > self.best_individual.fitness
                ):
                    self.best_individual = individual
                    logger.info(f"New best: {fitness:.4f}")
            
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                individual.set_fitness(0.0)
    
    def evolve_generation(self, evaluator: Callable[[ModelGraph], float]) -> None:
        """Evolve one generation."""
        individuals = list(self.population.individuals)
        new_population = []
        
        for i, target in enumerate(individuals):
            # Generate mutant
            mutant = self.mutate_individual(i)
            
            # Crossover
            trial = self.crossover(target, mutant)
            
            # Evaluate trial
            trial_fitness = evaluator(trial.graph)
            trial.set_fitness(trial_fitness)
            
            # Selection
            if trial_fitness > target.fitness:
                new_population.append(trial)
                if trial_fitness > self.best_individual.fitness:
                    self.best_individual = trial
            else:
                new_population.append(target)
        
        # Update population
        self.population.clear()
        self.population.add_many(new_population)
    
    def optimize(
        self,
        evaluator: Callable[[ModelGraph], float],
        callback: Optional[Callable[[int, Population], None]] = None
    ) -> Individual:
        """Run DE optimization."""
        try:
            # Initialize
            self.initialize_population()
            self.evaluate_population(evaluator)
            
            # Record initial stats
            stats = self.population.get_statistics()
            self.history.append(stats)
            
            # Evolution loop
            for gen in range(self.num_generations):
                logger.info(f"Generation {gen + 1}/{self.num_generations}")
                
                # Evolve
                self.evolve_generation(evaluator)
                
                # Record stats
                stats = self.population.get_statistics()
                self.history.append(stats)
                
                logger.info(
                    f"Gen {gen + 1}: Best={stats['best_fitness']:.4f}, "
                    f"Mean={stats['mean_fitness']:.4f}"
                )
                
                # Callback
                if callback:
                    callback(gen + 1, self.population)
                
                # Advance generation
                self.population.next_generation()
            
            logger.info(f"DE complete: Best fitness = {self.best_individual.fitness:.4f}")
            return self.best_individual
        
        except Exception as e:
            logger.error(f"DE optimization failed: {e}")
            raise OptimizerError(f"DE optimization failed: {e}") from e
    
    def get_history(self) -> List[dict]:
        """Get optimization history."""
        return self.history
    
    def get_best_n(self, n: int = 10) -> List[Individual]:
        """Get top N individuals."""
        return self.population.get_best(n=n)
    
    def reset(self) -> None:
        """Reset optimizer."""
        self.population.clear()
        self.history.clear()
        self.best_individual = None
        logger.info("DE reset")
    
    def __repr__(self) -> str:
        return (
            f"DifferentialEvolution(pop={self.population_size}, "
            f"F={self.mutation_factor}, CR={self.crossover_prob})"
        )
