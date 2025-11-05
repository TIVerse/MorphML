"""NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation.

NSGA-II is a state-of-the-art multi-objective evolutionary algorithm that finds
a set of Pareto-optimal solutions balancing multiple competing objectives.

Key Features:
- Fast non-dominated sorting (O(MN²))
- Crowding distance for diversity maintenance
- Elitism (parent + offspring selection)
- Multiple objective optimization

Reference:
    Deb, K., et al. "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II."
    IEEE Transactions on Evolutionary Computation, 2002.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from morphml.core.dsl import SearchSpace
from morphml.core.graph import GraphMutator, ModelGraph
from morphml.core.search import Individual
from morphml.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MultiObjectiveIndividual:
    """
    Individual with multiple fitness objectives.
    
    Attributes:
        genome: ModelGraph architecture
        objectives: Dictionary of objective values {'accuracy': 0.95, 'latency': 12.5}
        rank: Pareto rank (0 = non-dominated front)
        crowding_distance: Density measure in objective space
        domination_count: Number of solutions dominating this one
        dominated_solutions: List of solutions this one dominates
        
    Example:
        >>> ind = MultiObjectiveIndividual(
        ...     genome=graph,
        ...     objectives={'accuracy': 0.95, 'latency': 12.5, 'params': 2.1}
        ... )
        >>> ind.rank = 0  # Non-dominated
        >>> ind.crowding_distance = 1.5
    """
    
    genome: ModelGraph
    objectives: Dict[str, float] = field(default_factory=dict)
    rank: int = 0
    crowding_distance: float = 0.0
    domination_count: int = 0
    dominated_solutions: List['MultiObjectiveIndividual'] = field(default_factory=list)
    
    def dominates(self, other: 'MultiObjectiveIndividual', objective_specs: List[Dict]) -> bool:
        """
        Check if this individual Pareto-dominates another.
        
        Individual A dominates B if:
        1. A is better than or equal to B in all objectives
        2. A is strictly better than B in at least one objective
        
        Args:
            other: Other individual to compare
            objective_specs: List of objective specifications with 'maximize' flags
            
        Returns:
            True if this individual dominates the other
            
        Example:
            >>> ind1.objectives = {'acc': 0.95, 'latency': -10}
            >>> ind2.objectives = {'acc': 0.90, 'latency': -15}
            >>> ind1.dominates(ind2, objectives)  # True (better in both)
        """
        better_in_any = False
        
        for obj_spec in objective_specs:
            obj_name = obj_spec['name']
            maximize = obj_spec['maximize']
            
            my_value = self.objectives.get(obj_name, 0.0)
            other_value = other.objectives.get(obj_name, 0.0)
            
            # Compare based on optimization direction
            if maximize:
                if my_value < other_value:
                    return False  # Worse in this objective
                elif my_value > other_value:
                    better_in_any = True
            else:  # Minimize
                if my_value > other_value:
                    return False  # Worse in this objective
                elif my_value < other_value:
                    better_in_any = True
        
        return better_in_any
    
    def __repr__(self) -> str:
        """String representation."""
        obj_str = ', '.join(f"{k}={v:.4f}" for k, v in self.objectives.items())
        return f"Individual(rank={self.rank}, crowding={self.crowding_distance:.4f}, {obj_str})"


class NSGA2Optimizer:
    """
    NSGA-II: Non-dominated Sorting Genetic Algorithm II.
    
    Multi-objective evolutionary algorithm that finds a diverse set of
    Pareto-optimal solutions balancing multiple competing objectives.
    
    Algorithm:
    1. Initialize random population
    2. Evaluate all objectives for each individual
    3. Fast non-dominated sorting (rank assignment)
    4. Crowding distance calculation (diversity measure)
    5. Tournament selection based on rank and crowding
    6. Crossover and mutation to generate offspring
    7. Elitist selection from parent + offspring
    8. Repeat steps 3-7 for N generations
    
    Configuration:
        population_size: Population size (default: 100)
        num_generations: Number of generations (default: 100)
        crossover_rate: Crossover probability (default: 0.9)
        mutation_rate: Mutation probability (default: 0.1)
        tournament_size: Tournament selection size (default: 2)
        objectives: List of objective specifications
            Format: [{'name': 'accuracy', 'maximize': True}, ...]
            
    Example:
        >>> from morphml.optimizers.multi_objective import NSGA2Optimizer
        >>> optimizer = NSGA2Optimizer(
        ...     search_space=space,
        ...     config={
        ...         'population_size': 100,
        ...         'num_generations': 100,
        ...         'objectives': [
        ...             {'name': 'accuracy', 'maximize': True},
        ...             {'name': 'latency', 'maximize': False},
        ...             {'name': 'params', 'maximize': False}
        ...         ]
        ...     }
        ... )
        >>> pareto_front = optimizer.optimize(evaluator)
        >>> print(f"Found {len(pareto_front)} Pareto-optimal solutions")
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize NSGA-II optimizer.
        
        Args:
            search_space: SearchSpace defining architecture options
            config: Configuration dictionary
        """
        self.search_space = search_space
        self.config = config or {}
        
        # Algorithm parameters
        self.pop_size = self.config.get('population_size', 100)
        self.num_generations = self.config.get('num_generations', 100)
        self.crossover_rate = self.config.get('crossover_rate', 0.9)
        self.mutation_rate = self.config.get('mutation_rate', 0.1)
        self.tournament_size = self.config.get('tournament_size', 2)
        
        # Objective specifications
        self.objectives = self.config.get('objectives', [
            {'name': 'fitness', 'maximize': True},
            {'name': 'complexity', 'maximize': False}
        ])
        
        # State
        self.population: List[MultiObjectiveIndividual] = []
        self.generation = 0
        self.history: List[Dict[str, Any]] = []
        
        # Mutation operator
        self.mutator = GraphMutator()
        
        logger.info(
            f"Initialized NSGA-II with pop_size={self.pop_size}, "
            f"generations={self.num_generations}, {len(self.objectives)} objectives"
        )
    
    def initialize_population(self) -> None:
        """Initialize random population from search space."""
        self.population = []
        for i in range(self.pop_size):
            genome = self.search_space.sample()
            individual = MultiObjectiveIndividual(genome=genome)
            self.population.append(individual)
        
        logger.debug(f"Initialized population of {self.pop_size} individuals")
    
    def evaluate_individual(
        self,
        individual: MultiObjectiveIndividual,
        evaluator: Callable
    ) -> None:
        """
        Evaluate all objectives for an individual.
        
        Args:
            individual: Individual to evaluate
            evaluator: Function that returns dict of objective values
                       evaluator(graph) -> {'accuracy': 0.95, 'latency': 12.5, ...}
        """
        # Call evaluator
        results = evaluator(individual.genome)
        
        # Store objective values
        for obj_spec in self.objectives:
            obj_name = obj_spec['name']
            value = results.get(obj_name, 0.0)
            individual.objectives[obj_name] = value
    
    def fast_non_dominated_sort(
        self,
        population: List[MultiObjectiveIndividual]
    ) -> List[List[MultiObjectiveIndividual]]:
        """
        Fast non-dominated sorting algorithm.
        
        Assigns Pareto ranks to individuals and returns fronts.
        
        Complexity: O(MN²) where M = objectives, N = population size
        
        Args:
            population: Population to sort
            
        Returns:
            List of Pareto fronts [F0, F1, F2, ...]
            where F0 is the non-dominated front
            
        Example:
            >>> fronts = optimizer.fast_non_dominated_sort(population)
            >>> pareto_front = fronts[0]  # Non-dominated solutions
        """
        # Initialize
        for ind in population:
            ind.domination_count = 0
            ind.dominated_solutions = []
        
        fronts: List[List[MultiObjectiveIndividual]] = [[]]
        
        # Find dominated solutions
        for p in population:
            for q in population:
                if p is q:
                    continue
                
                if p.dominates(q, self.objectives):
                    p.dominated_solutions.append(q)
                elif q.dominates(p, self.objectives):
                    p.domination_count += 1
            
            # Non-dominated solutions (rank 0)
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        # Build subsequent fronts
        i = 0
        while fronts[i]:
            next_front = []
            
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            
            i += 1
            if next_front:
                fronts.append(next_front)
        
        return fronts
    
    def calculate_crowding_distance(
        self,
        front: List[MultiObjectiveIndividual]
    ) -> None:
        """
        Calculate crowding distance for individuals in a front.
        
        Crowding distance measures the density of solutions around an
        individual in objective space. Higher distance = more isolated.
        
        Boundary points get infinite distance to preserve diversity.
        
        Args:
            front: List of individuals in the same front
        """
        n = len(front)
        
        if n == 0:
            return
        
        # Initialize distances to 0
        for ind in front:
            ind.crowding_distance = 0.0
        
        # For each objective
        for obj_spec in self.objectives:
            obj_name = obj_spec['name']
            
            # Sort by objective value
            front_sorted = sorted(front, key=lambda x: x.objectives[obj_name])
            
            # Boundary points get infinite distance
            front_sorted[0].crowding_distance = float('inf')
            front_sorted[-1].crowding_distance = float('inf')
            
            # Objective range
            obj_min = front_sorted[0].objectives[obj_name]
            obj_max = front_sorted[-1].objectives[obj_name]
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue  # All values same, no distance contribution
            
            # Calculate crowding distance
            for i in range(1, n - 1):
                distance = (
                    front_sorted[i + 1].objectives[obj_name] -
                    front_sorted[i - 1].objectives[obj_name]
                ) / obj_range
                
                front_sorted[i].crowding_distance += distance
    
    def tournament_selection(self) -> MultiObjectiveIndividual:
        """
        Binary tournament selection.
        
        Select k random individuals and return the best based on:
        1. Lower rank (closer to Pareto front)
        2. If same rank, higher crowding distance (more isolated)
        
        Returns:
            Selected individual
        """
        contestants = random.sample(self.population, self.tournament_size)
        
        # Sort by rank (ascending), then crowding distance (descending)
        contestants.sort(key=lambda x: (x.rank, -x.crowding_distance))
        
        return contestants[0]
    
    def crossover(
        self,
        parent1: ModelGraph,
        parent2: ModelGraph
    ) -> Tuple[ModelGraph, ModelGraph]:
        """
        Graph crossover operation.
        
        Exchanges subgraphs between two parent architectures.
        
        Args:
            parent1: First parent graph
            parent2: Second parent graph
            
        Returns:
            Two offspring graphs
        """
        # Clone parents
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        # Simple crossover: swap random nodes
        if len(child1.nodes) > 2 and len(child2.nodes) > 2:
            # Get non-terminal nodes
            nodes1 = [n for n in child1.nodes.values() 
                     if n.operation not in ['input', 'output']]
            nodes2 = [n for n in child2.nodes.values() 
                     if n.operation not in ['input', 'output']]
            
            if nodes1 and nodes2:
                # Select random nodes to swap
                idx1 = random.randint(0, len(nodes1) - 1)
                idx2 = random.randint(0, len(nodes2) - 1)
                
                # Swap node operations (simplified crossover)
                nodes1[idx1].operation, nodes2[idx2].operation = \
                    nodes2[idx2].operation, nodes1[idx1].operation
        
        return child1, child2
    
    def mutate(self, graph: ModelGraph) -> ModelGraph:
        """
        Mutation operation on architecture.
        
        Args:
            graph: Graph to mutate
            
        Returns:
            Mutated graph
        """
        mutated = graph.clone()
        
        # Random mutation type
        mutation_type = random.choice(['add_node', 'remove_node', 'modify_node'])
        
        try:
            if mutation_type == 'add_node':
                self.mutator.add_node(mutated)
            elif mutation_type == 'remove_node' and len(mutated.nodes) > 3:
                self.mutator.remove_node(mutated)
            elif mutation_type == 'modify_node':
                self.mutator.mutate_node_params(mutated)
        except Exception as e:
            logger.warning(f"Mutation failed: {e}. Returning original.")
            return graph
        
        return mutated if mutated.is_valid_dag() else graph
    
    def generate_offspring(self) -> List[MultiObjectiveIndividual]:
        """
        Generate offspring population via selection, crossover, and mutation.
        
        Returns:
            Offspring population
        """
        offspring = []
        
        while len(offspring) < self.pop_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1_genome, child2_genome = self.crossover(
                    parent1.genome,
                    parent2.genome
                )
            else:
                child1_genome = parent1.genome.clone()
                child2_genome = parent2.genome.clone()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1_genome = self.mutate(child1_genome)
            if random.random() < self.mutation_rate:
                child2_genome = self.mutate(child2_genome)
            
            # Create offspring individuals
            child1 = MultiObjectiveIndividual(genome=child1_genome)
            child2 = MultiObjectiveIndividual(genome=child2_genome)
            
            offspring.extend([child1, child2])
        
        return offspring[:self.pop_size]
    
    def environmental_selection(
        self,
        combined_population: List[MultiObjectiveIndividual]
    ) -> List[MultiObjectiveIndividual]:
        """
        Select next generation from combined parent + offspring population.
        
        Elitist selection preserving best solutions based on:
        1. Pareto rank (lower is better)
        2. Crowding distance (higher is better for diversity)
        
        Args:
            combined_population: Combined parent + offspring population
            
        Returns:
            Selected population of size pop_size
        """
        # Fast non-dominated sorting
        fronts = self.fast_non_dominated_sort(combined_population)
        
        # Calculate crowding distances
        for front in fronts:
            self.calculate_crowding_distance(front)
        
        # Select individuals for next generation
        next_population = []
        
        for front in fronts:
            if len(next_population) + len(front) <= self.pop_size:
                # Add entire front
                next_population.extend(front)
            else:
                # Fill remaining slots with most isolated individuals
                remaining = self.pop_size - len(next_population)
                front_sorted = sorted(front, key=lambda x: -x.crowding_distance)
                next_population.extend(front_sorted[:remaining])
                break
        
        return next_population
    
    def optimize(self, evaluator: Callable) -> List[MultiObjectiveIndividual]:
        """
        Run NSGA-II optimization.
        
        Args:
            evaluator: Function that evaluates objectives
                       evaluator(graph) -> {'accuracy': 0.95, 'latency': 12.5, ...}
                       
        Returns:
            Pareto front (list of non-dominated solutions)
            
        Example:
            >>> def my_evaluator(graph):
            ...     return {
            ...         'accuracy': evaluate_accuracy(graph),
            ...         'latency': measure_latency(graph),
            ...         'params': count_parameters(graph)
            ...     }
            >>> pareto_front = optimizer.optimize(my_evaluator)
        """
        logger.info(f"Starting NSGA-II optimization with {self.pop_size} individuals")
        
        # Initialize population
        self.initialize_population()
        
        # Evaluate initial population
        for individual in self.population:
            self.evaluate_individual(individual, evaluator)
        
        # Evolution loop
        for generation in range(self.num_generations):
            self.generation = generation
            
            # Generate offspring
            offspring = self.generate_offspring()
            
            # Evaluate offspring
            for individual in offspring:
                self.evaluate_individual(individual, evaluator)
            
            # Combine parent + offspring
            combined = self.population + offspring
            
            # Environmental selection
            self.population = self.environmental_selection(combined)
            
            # Logging
            if generation % 10 == 0 or generation == self.num_generations - 1:
                pareto_front = [ind for ind in self.population if ind.rank == 0]
                logger.info(
                    f"Generation {generation}/{self.num_generations}: "
                    f"Pareto front size = {len(pareto_front)}"
                )
                
                # Log objective statistics
                self._log_pareto_statistics(pareto_front)
                
                # Save to history
                self.history.append({
                    'generation': generation,
                    'pareto_size': len(pareto_front),
                    'pareto_front': pareto_front
                })
        
        # Extract final Pareto front
        pareto_front = [ind for ind in self.population if ind.rank == 0]
        
        logger.info(
            f"NSGA-II complete. Final Pareto front: {len(pareto_front)} solutions"
        )
        
        return pareto_front
    
    def _log_pareto_statistics(self, pareto_front: List[MultiObjectiveIndividual]) -> None:
        """Log statistics of Pareto front objectives."""
        if not pareto_front:
            return
        
        for obj_spec in self.objectives:
            obj_name = obj_spec['name']
            values = [ind.objectives[obj_name] for ind in pareto_front]
            
            if values:
                logger.info(
                    f"  {obj_name}: "
                    f"min={min(values):.4f}, "
                    f"max={max(values):.4f}, "
                    f"mean={np.mean(values):.4f}"
                )
    
    def get_pareto_front(self) -> List[MultiObjectiveIndividual]:
        """
        Get current Pareto front (non-dominated solutions).
        
        Returns:
            List of non-dominated individuals
        """
        return [ind for ind in self.population if ind.rank == 0]
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self.history


# Convenience function
def optimize_with_nsga2(
    search_space: SearchSpace,
    evaluator: Callable,
    objectives: List[Dict[str, Any]],
    population_size: int = 100,
    num_generations: int = 100,
    verbose: bool = True
) -> List[MultiObjectiveIndividual]:
    """
    Quick NSGA-II optimization with sensible defaults.
    
    Args:
        search_space: SearchSpace to optimize over
        evaluator: Multi-objective evaluator function
        objectives: List of objective specifications
        population_size: Population size
        num_generations: Number of generations
        verbose: Print progress
        
    Returns:
        Pareto front
        
    Example:
        >>> pareto_front = optimize_with_nsga2(
        ...     search_space=space,
        ...     evaluator=my_evaluator,
        ...     objectives=[
        ...         {'name': 'accuracy', 'maximize': True},
        ...         {'name': 'latency', 'maximize': False}
        ...     ],
        ...     population_size=100,
        ...     num_generations=50
        ... )
    """
    optimizer = NSGA2Optimizer(
        search_space=search_space,
        config={
            'population_size': population_size,
            'num_generations': num_generations,
            'objectives': objectives
        }
    )
    
    pareto_front = optimizer.optimize(evaluator)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"NSGA-II Optimization Complete")
        print(f"{'='*60}")
        print(f"Pareto Front Size: {len(pareto_front)}")
        print(f"Objectives: {[obj['name'] for obj in objectives]}")
        print(f"{'='*60}\n")
    
    return pareto_front
