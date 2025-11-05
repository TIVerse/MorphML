"""Differential Evolution (DE) for neural architecture search.

Differential Evolution is a powerful evolutionary algorithm that uses vector
differences for mutation. It's particularly effective for continuous optimization
and has few parameters to tune.

Key Features:
- Vector difference-based mutation
- Multiple mutation strategies (rand/1, best/1, rand/2)
- Binomial/exponential crossover
- Greedy selection
- Self-adaptive variants

Reference:
    Storn, R., and Price, K. "Differential Evolution - A Simple and Efficient
    Heuristic for Global Optimization over Continuous Spaces." Journal of Global
    Optimization, 1997.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from morphml.core.dsl import SearchSpace
from morphml.core.search import Individual
from morphml.logging_config import get_logger
from morphml.optimizers.evolutionary.encoding import ArchitectureEncoder

logger = get_logger(__name__)


class DifferentialEvolution:
    """
    Differential Evolution optimizer for architecture search.
    
    DE uses vector differences for mutation, creating trial vectors that
    are compared against target vectors. The algorithm is simple yet
    powerful for continuous optimization.
    
    Mutation Strategies:
    1. **DE/rand/1:** mutant = x_r1 + F * (x_r2 - x_r3)
       - Random base vector with one difference
       - Good exploration
       
    2. **DE/best/1:** mutant = x_best + F * (x_r1 - x_r2)
       - Best vector as base
       - Faster convergence, risk of premature convergence
       
    3. **DE/rand/2:** mutant = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)
       - Two difference vectors
       - More exploration
       
    4. **DE/current-to-best/1:** mutant = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
       - Moves toward best
       - Balanced exploration/exploitation
    
    Crossover:
    - **Binomial:** Each parameter inherited from mutant with probability CR
    - **Exponential:** Consecutive parameters inherited from mutant
    
    Configuration:
        population_size: Population size (default: 50)
        max_generations: Maximum generations (default: 100)
        F: Mutation scaling factor (default: 0.8)
        CR: Crossover probability (default: 0.9)
        strategy: Mutation strategy (default: 'rand/1')
        
    Example:
        >>> from morphml.optimizers.evolutionary import DifferentialEvolution
        >>> optimizer = DifferentialEvolution(
        ...     search_space=space,
        ...     config={
        ...         'population_size': 50,
        ...         'F': 0.8,
        ...         'CR': 0.9,
        ...         'strategy': 'rand/1'
        ...     }
        ... )
        >>> best = optimizer.optimize(evaluator)
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Differential Evolution optimizer.
        
        Args:
            search_space: SearchSpace for architecture sampling
            config: Configuration dictionary
        """
        self.search_space = search_space
        self.config = config or {}
        
        # DE parameters
        self.population_size = self.config.get('population_size', 50)
        self.max_generations = self.config.get('max_generations', 100)
        
        # F: Differential weight (scaling factor)
        self.F = self.config.get('F', 0.8)
        self.F_min = self.config.get('F_min', 0.5)
        self.F_max = self.config.get('F_max', 1.0)
        
        # CR: Crossover probability
        self.CR = self.config.get('CR', 0.9)
        
        # Strategy
        self.strategy = self.config.get('strategy', 'rand/1')
        self.crossover_type = self.config.get('crossover_type', 'binomial')
        
        # Adaptive parameters
        self.adaptive = self.config.get('adaptive', False)
        
        # Architecture encoding
        self.max_nodes = self.config.get('max_nodes', 20)
        self.encoder = ArchitectureEncoder(search_space, self.max_nodes)
        self.dim = self.encoder.get_dimension()
        
        # Population state
        self.population: List[Individual] = []
        self.population_vectors: List[np.ndarray] = []
        self.fitnesses: List[float] = []
        
        self.best_individual: Optional[Individual] = None
        self.best_vector: Optional[np.ndarray] = None
        self.best_fitness: float = -np.inf
        
        # History
        self.generation = 0
        self.history: List[Dict[str, Any]] = []
        
        logger.info(
            f"Initialized DE: strategy={self.strategy}, "
            f"pop_size={self.population_size}, F={self.F}, CR={self.CR}"
        )
    
    def initialize_population(self, evaluator: Callable) -> None:
        """
        Initialize random population and evaluate.
        
        Args:
            evaluator: Fitness evaluation function
        """
        self.population = []
        self.population_vectors = []
        self.fitnesses = []
        
        for i in range(self.population_size):
            # Random vector in [0, 1]^dim
            vector = np.random.rand(self.dim)
            
            # Decode to architecture
            graph = self.encoder.decode(vector)
            
            # Evaluate
            fitness = evaluator(graph)
            
            # Store
            individual = Individual(graph)
            individual.fitness = fitness
            
            self.population.append(individual)
            self.population_vectors.append(vector)
            self.fitnesses.append(fitness)
            
            # Update best
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_vector = vector.copy()
                self.best_individual = individual
        
        logger.debug(
            f"Initialized population: best_fitness={self.best_fitness:.4f}"
        )
    
    def mutate(self, target_idx: int) -> np.ndarray:
        """
        Create mutant vector using selected strategy.
        
        Args:
            target_idx: Index of target individual
            
        Returns:
            Mutant vector
        """
        # Get indices excluding target
        indices = [i for i in range(self.population_size) if i != target_idx]
        
        if self.strategy == 'rand/1':
            return self._mutate_rand_1(indices)
        elif self.strategy == 'best/1':
            return self._mutate_best_1(indices)
        elif self.strategy == 'rand/2':
            return self._mutate_rand_2(indices)
        elif self.strategy == 'current-to-best/1':
            return self._mutate_current_to_best_1(target_idx, indices)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _mutate_rand_1(self, indices: List[int]) -> np.ndarray:
        """
        DE/rand/1 mutation.
        
        mutant = x_r1 + F * (x_r2 - x_r3)
        """
        r1, r2, r3 = random.sample(indices, 3)
        
        mutant = (
            self.population_vectors[r1] +
            self.F * (self.population_vectors[r2] - self.population_vectors[r3])
        )
        
        return mutant
    
    def _mutate_best_1(self, indices: List[int]) -> np.ndarray:
        """
        DE/best/1 mutation.
        
        mutant = x_best + F * (x_r1 - x_r2)
        """
        r1, r2 = random.sample(indices, 2)
        
        mutant = (
            self.best_vector +
            self.F * (self.population_vectors[r1] - self.population_vectors[r2])
        )
        
        return mutant
    
    def _mutate_rand_2(self, indices: List[int]) -> np.ndarray:
        """
        DE/rand/2 mutation.
        
        mutant = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)
        """
        r1, r2, r3, r4, r5 = random.sample(indices, 5)
        
        mutant = (
            self.population_vectors[r1] +
            self.F * (self.population_vectors[r2] - self.population_vectors[r3]) +
            self.F * (self.population_vectors[r4] - self.population_vectors[r5])
        )
        
        return mutant
    
    def _mutate_current_to_best_1(
        self,
        target_idx: int,
        indices: List[int]
    ) -> np.ndarray:
        """
        DE/current-to-best/1 mutation.
        
        mutant = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
        """
        r1, r2 = random.sample(indices, 2)
        
        mutant = (
            self.population_vectors[target_idx] +
            self.F * (self.best_vector - self.population_vectors[target_idx]) +
            self.F * (self.population_vectors[r1] - self.population_vectors[r2])
        )
        
        return mutant
    
    def crossover(
        self,
        target: np.ndarray,
        mutant: np.ndarray
    ) -> np.ndarray:
        """
        Crossover target and mutant to create trial vector.
        
        Args:
            target: Target vector
            mutant: Mutant vector
            
        Returns:
            Trial vector
        """
        if self.crossover_type == 'binomial':
            return self._crossover_binomial(target, mutant)
        elif self.crossover_type == 'exponential':
            return self._crossover_exponential(target, mutant)
        else:
            raise ValueError(f"Unknown crossover type: {self.crossover_type}")
    
    def _crossover_binomial(
        self,
        target: np.ndarray,
        mutant: np.ndarray
    ) -> np.ndarray:
        """
        Binomial crossover.
        
        Each dimension inherited from mutant with probability CR.
        """
        trial = target.copy()
        
        # Ensure at least one dimension from mutant
        j_rand = random.randint(0, self.dim - 1)
        
        for j in range(self.dim):
            if random.random() < self.CR or j == j_rand:
                trial[j] = mutant[j]
        
        return trial
    
    def _crossover_exponential(
        self,
        target: np.ndarray,
        mutant: np.ndarray
    ) -> np.ndarray:
        """
        Exponential crossover.
        
        Copy consecutive dimensions from mutant.
        """
        trial = target.copy()
        
        # Start position
        n = random.randint(0, self.dim - 1)
        L = 0
        
        # Copy consecutive dimensions
        while True:
            trial[n] = mutant[n]
            n = (n + 1) % self.dim
            L += 1
            
            if random.random() >= self.CR or L >= self.dim:
                break
        
        return trial
    
    def select(
        self,
        target_fitness: float,
        trial_fitness: float,
        target_vector: np.ndarray,
        trial_vector: np.ndarray
    ) -> tuple:
        """
        Greedy selection between target and trial.
        
        Args:
            target_fitness: Target fitness
            trial_fitness: Trial fitness
            target_vector: Target vector
            trial_vector: Trial vector
            
        Returns:
            (selected_fitness, selected_vector) tuple
        """
        if trial_fitness >= target_fitness:
            return trial_fitness, trial_vector
        else:
            return target_fitness, target_vector
    
    def optimize(self, evaluator: Callable) -> Individual:
        """
        Run Differential Evolution optimization.
        
        Args:
            evaluator: Function that evaluates ModelGraph -> fitness
            
        Returns:
            Best Individual found
            
        Example:
            >>> def my_evaluator(graph):
            ...     return train_and_evaluate(graph)
            >>> best = optimizer.optimize(my_evaluator)
            >>> print(f"Best fitness: {best.fitness:.4f}")
        """
        logger.info(
            f"Starting DE optimization for {self.max_generations} generations"
        )
        
        # Initialize population
        self.initialize_population(evaluator)
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.generation = generation
            
            new_population = []
            new_vectors = []
            new_fitnesses = []
            
            # For each individual
            for i in range(self.population_size):
                target_vector = self.population_vectors[i]
                target_fitness = self.fitnesses[i]
                
                # Mutation
                mutant_vector = self.mutate(i)
                
                # Boundary handling (clamp to [0, 1])
                mutant_vector = np.clip(mutant_vector, 0.0, 1.0)
                
                # Crossover
                trial_vector = self.crossover(target_vector, mutant_vector)
                
                # Decode and evaluate trial
                trial_graph = self.encoder.decode(trial_vector)
                trial_fitness = evaluator(trial_graph)
                
                # Selection
                selected_fitness, selected_vector = self.select(
                    target_fitness, trial_fitness,
                    target_vector, trial_vector
                )
                
                # Update population
                if selected_fitness == trial_fitness:
                    # Trial won
                    individual = Individual(trial_graph)
                    individual.fitness = trial_fitness
                else:
                    # Target won
                    individual = self.population[i]
                
                new_population.append(individual)
                new_vectors.append(selected_vector)
                new_fitnesses.append(selected_fitness)
                
                # Update global best
                if selected_fitness > self.best_fitness:
                    self.best_fitness = selected_fitness
                    self.best_vector = selected_vector.copy()
                    self.best_individual = individual
            
            # Update population
            self.population = new_population
            self.population_vectors = new_vectors
            self.fitnesses = new_fitnesses
            
            # Record history
            avg_fitness = np.mean(self.fitnesses)
            
            self.history.append({
                'generation': generation,
                'best_fitness': self.best_fitness,
                'avg_fitness': avg_fitness,
                'std_fitness': np.std(self.fitnesses)
            })
            
            # Logging
            if generation % 10 == 0 or generation == self.max_generations - 1:
                logger.info(
                    f"Generation {generation}/{self.max_generations}: "
                    f"best={self.best_fitness:.4f}, "
                    f"avg={avg_fitness:.4f}"
                )
        
        logger.info(
            f"DE complete. Best fitness: {self.best_fitness:.4f}"
        )
        
        return self.best_individual
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self.history
    
    def plot_convergence(self, save_path: Optional[str] = None) -> None:
        """
        Plot DE convergence.
        
        Args:
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, cannot plot")
            return
        
        if not self.history:
            logger.warning("No history to plot")
            return
        
        generations = [h['generation'] for h in self.history]
        best_fitness = [h['best_fitness'] for h in self.history]
        avg_fitness = [h['avg_fitness'] for h in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
        plt.plot(generations, avg_fitness, 'r--', linewidth=2, label='Average Fitness')
        
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness', fontsize=12)
        plt.title('Differential Evolution Convergence', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Convergence plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DifferentialEvolution("
            f"strategy={self.strategy}, "
            f"pop_size={self.population_size}, "
            f"F={self.F:.2f}, "
            f"CR={self.CR:.2f})"
        )


# Convenience function
def optimize_with_de(
    search_space: SearchSpace,
    evaluator: Callable,
    population_size: int = 50,
    max_generations: int = 100,
    F: float = 0.8,
    CR: float = 0.9,
    strategy: str = 'rand/1',
    verbose: bool = True
) -> Individual:
    """
    Quick Differential Evolution optimization with sensible defaults.
    
    Args:
        search_space: SearchSpace to optimize over
        evaluator: Fitness evaluation function
        population_size: Population size
        max_generations: Maximum generations
        F: Mutation scaling factor
        CR: Crossover probability
        strategy: Mutation strategy ('rand/1', 'best/1', 'rand/2')
        verbose: Print progress
        
    Returns:
        Best Individual found
        
    Example:
        >>> from morphml.optimizers.evolutionary import optimize_with_de
        >>> best = optimize_with_de(
        ...     search_space=space,
        ...     evaluator=my_evaluator,
        ...     population_size=50,
        ...     strategy='rand/1'
        ... )
    """
    optimizer = DifferentialEvolution(
        search_space=search_space,
        config={
            'population_size': population_size,
            'max_generations': max_generations,
            'F': F,
            'CR': CR,
            'strategy': strategy
        }
    )
    
    best = optimizer.optimize(evaluator)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Differential Evolution Complete")
        print(f"{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"Best Fitness: {best.fitness:.4f}")
        print(f"Generations: {max_generations}")
        print(f"Population Size: {population_size}")
        print(f"{'='*60}\n")
    
    return best
