"""Adaptive genetic operators that adjust based on population diversity.

This module provides adaptive versions of genetic operators that automatically
adjust their rates based on population diversity and search progress.

Example:
    >>> from morphml.optimizers.adaptive_operators import AdaptiveCrossoverManager
    >>> 
    >>> manager = AdaptiveCrossoverManager(
    ...     initial_rate=0.8,
    ...     min_rate=0.5,
    ...     max_rate=0.95
    ... )
    >>> 
    >>> # During optimization
    >>> crossover_rate = manager.get_rate(population, generation)
"""

from typing import List, Optional

import numpy as np

from morphml.core.search.individual import Individual
from morphml.core.search.population import Population
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class AdaptiveCrossoverManager:
    """
    Manages adaptive crossover rate based on population diversity.
    
    Increases crossover rate when diversity is low (exploitation)
    Decreases crossover rate when diversity is high (exploration)
    
    Attributes:
        initial_rate: Starting crossover rate
        min_rate: Minimum allowed rate
        max_rate: Maximum allowed rate
        adaptation_speed: How quickly to adapt (0.0-1.0)
        
    Example:
        >>> manager = AdaptiveCrossoverManager(initial_rate=0.8)
        >>> rate = manager.get_rate(population, generation=10)
        >>> print(f"Adaptive rate: {rate:.3f}")
    """
    
    def __init__(
        self,
        initial_rate: float = 0.8,
        min_rate: float = 0.5,
        max_rate: float = 0.95,
        adaptation_speed: float = 0.1,
        diversity_window: int = 10,
    ):
        """
        Initialize adaptive crossover manager.
        
        Args:
            initial_rate: Starting crossover rate (0.0-1.0)
            min_rate: Minimum crossover rate
            max_rate: Maximum crossover rate
            adaptation_speed: Speed of adaptation (0.0-1.0)
            diversity_window: Number of generations to track diversity
        """
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.adaptation_speed = adaptation_speed
        self.diversity_window = diversity_window
        
        self.current_rate = initial_rate
        self.diversity_history: List[float] = []
        
        logger.info(
            f"Initialized AdaptiveCrossoverManager: "
            f"rate={initial_rate:.2f}, range=[{min_rate:.2f}, {max_rate:.2f}]"
        )
    
    def get_rate(
        self,
        population: Population,
        generation: int,
        force_update: bool = True,
    ) -> float:
        """
        Get adaptive crossover rate for current generation.
        
        Args:
            population: Current population
            generation: Current generation number
            force_update: Whether to force rate update
            
        Returns:
            Adaptive crossover rate
        """
        if force_update:
            diversity = self._calculate_diversity(population)
            self._update_rate(diversity, generation)
        
        return self.current_rate
    
    def _calculate_diversity(self, population: Population) -> float:
        """
        Calculate population diversity.
        
        Uses multiple metrics:
        - Fitness variance
        - Graph structure diversity
        - Parameter diversity
        
        Args:
            population: Population to analyze
            
        Returns:
            Diversity score (0.0-1.0)
        """
        if len(population) < 2:
            return 0.0
        
        individuals = list(population.individuals.values())
        
        # Fitness diversity (normalized variance)
        fitnesses = [ind.fitness for ind in individuals if ind.fitness is not None]
        if len(fitnesses) < 2:
            fitness_diversity = 0.0
        else:
            fitness_std = np.std(fitnesses)
            fitness_mean = np.mean(fitnesses)
            fitness_diversity = fitness_std / (abs(fitness_mean) + 1e-8)
        
        # Structure diversity (unique node counts)
        node_counts = [len(ind.graph.nodes) for ind in individuals]
        unique_counts = len(set(node_counts))
        structure_diversity = unique_counts / len(individuals)
        
        # Edge diversity
        edge_counts = [len(ind.graph.edges) for ind in individuals]
        unique_edges = len(set(edge_counts))
        edge_diversity = unique_edges / len(individuals)
        
        # Combined diversity (weighted average)
        diversity = (
            0.5 * fitness_diversity +
            0.3 * structure_diversity +
            0.2 * edge_diversity
        )
        
        # Normalize to [0, 1]
        diversity = np.clip(diversity, 0.0, 1.0)
        
        # Track history
        self.diversity_history.append(diversity)
        if len(self.diversity_history) > self.diversity_window:
            self.diversity_history.pop(0)
        
        return diversity
    
    def _update_rate(self, diversity: float, generation: int) -> None:
        """
        Update crossover rate based on diversity.
        
        Low diversity -> Increase crossover (more exploration)
        High diversity -> Decrease crossover (more exploitation)
        
        Args:
            diversity: Current diversity score
            generation: Current generation
        """
        # Calculate target rate based on diversity
        # Inverse relationship: low diversity -> high crossover
        target_rate = self.max_rate - (diversity * (self.max_rate - self.min_rate))
        
        # Smooth adaptation
        delta = target_rate - self.current_rate
        self.current_rate += self.adaptation_speed * delta
        
        # Clamp to bounds
        self.current_rate = np.clip(self.current_rate, self.min_rate, self.max_rate)
        
        logger.debug(
            f"Gen {generation}: diversity={diversity:.3f}, "
            f"crossover_rate={self.current_rate:.3f}"
        )
    
    def get_diversity_trend(self) -> str:
        """
        Get diversity trend description.
        
        Returns:
            Trend description ("increasing", "decreasing", "stable")
        """
        if len(self.diversity_history) < 3:
            return "insufficient_data"
        
        recent = self.diversity_history[-3:]
        if recent[-1] > recent[0] + 0.1:
            return "increasing"
        elif recent[-1] < recent[0] - 0.1:
            return "decreasing"
        else:
            return "stable"
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.current_rate = self.initial_rate
        self.diversity_history.clear()
        logger.info("Reset adaptive crossover manager")


class AdaptiveMutationManager:
    """
    Manages adaptive mutation rate based on search progress.
    
    Increases mutation when stuck in local optima
    Decreases mutation when making good progress
    
    Example:
        >>> manager = AdaptiveMutationManager(initial_rate=0.2)
        >>> rate = manager.get_rate(best_fitness_history, generation)
    """
    
    def __init__(
        self,
        initial_rate: float = 0.2,
        min_rate: float = 0.05,
        max_rate: float = 0.5,
        stagnation_threshold: int = 5,
    ):
        """
        Initialize adaptive mutation manager.
        
        Args:
            initial_rate: Starting mutation rate
            min_rate: Minimum mutation rate
            max_rate: Maximum mutation rate
            stagnation_threshold: Generations without improvement to trigger increase
        """
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.stagnation_threshold = stagnation_threshold
        
        self.current_rate = initial_rate
        self.best_fitness_history: List[float] = []
        self.stagnation_counter = 0
        
        logger.info(
            f"Initialized AdaptiveMutationManager: "
            f"rate={initial_rate:.2f}, range=[{min_rate:.2f}, {max_rate:.2f}]"
        )
    
    def get_rate(
        self,
        current_best_fitness: float,
        generation: int,
    ) -> float:
        """
        Get adaptive mutation rate.
        
        Args:
            current_best_fitness: Best fitness in current generation
            generation: Current generation number
            
        Returns:
            Adaptive mutation rate
        """
        self._update_rate(current_best_fitness, generation)
        return self.current_rate
    
    def _update_rate(self, best_fitness: float, generation: int) -> None:
        """
        Update mutation rate based on progress.
        
        Args:
            best_fitness: Current best fitness
            generation: Current generation
        """
        # Track best fitness
        if not self.best_fitness_history:
            self.best_fitness_history.append(best_fitness)
            return
        
        # Check for improvement
        previous_best = max(self.best_fitness_history)
        improvement = best_fitness - previous_best
        
        if improvement > 1e-6:  # Improvement threshold
            # Making progress - decrease mutation
            self.stagnation_counter = 0
            self.current_rate *= 0.95  # Gradual decrease
        else:
            # Stagnation - increase mutation
            self.stagnation_counter += 1
            if self.stagnation_counter >= self.stagnation_threshold:
                self.current_rate *= 1.1  # Increase to escape local optimum
                logger.info(
                    f"Gen {generation}: Stagnation detected, "
                    f"increasing mutation to {self.current_rate:.3f}"
                )
        
        # Clamp to bounds
        self.current_rate = np.clip(self.current_rate, self.min_rate, self.max_rate)
        
        # Update history
        self.best_fitness_history.append(best_fitness)
        
        logger.debug(
            f"Gen {generation}: best={best_fitness:.4f}, "
            f"mutation_rate={self.current_rate:.3f}"
        )
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.current_rate = self.initial_rate
        self.best_fitness_history.clear()
        self.stagnation_counter = 0
        logger.info("Reset adaptive mutation manager")


class AdaptiveOperatorScheduler:
    """
    Coordinates multiple adaptive operators.
    
    Manages crossover and mutation rates together, ensuring they
    complement each other for effective search.
    
    Example:
        >>> scheduler = AdaptiveOperatorScheduler()
        >>> rates = scheduler.get_rates(population, best_fitness, generation)
        >>> crossover_rate, mutation_rate = rates
    """
    
    def __init__(
        self,
        initial_crossover: float = 0.8,
        initial_mutation: float = 0.2,
        balance_operators: bool = True,
    ):
        """
        Initialize adaptive operator scheduler.
        
        Args:
            initial_crossover: Initial crossover rate
            initial_mutation: Initial mutation rate
            balance_operators: Whether to balance crossover and mutation
        """
        self.crossover_manager = AdaptiveCrossoverManager(initial_rate=initial_crossover)
        self.mutation_manager = AdaptiveMutationManager(initial_rate=initial_mutation)
        self.balance_operators = balance_operators
        
        logger.info("Initialized AdaptiveOperatorScheduler")
    
    def get_rates(
        self,
        population: Population,
        current_best_fitness: float,
        generation: int,
    ) -> tuple:
        """
        Get adaptive rates for both operators.
        
        Args:
            population: Current population
            current_best_fitness: Best fitness in current generation
            generation: Current generation number
            
        Returns:
            Tuple of (crossover_rate, mutation_rate)
        """
        crossover_rate = self.crossover_manager.get_rate(population, generation)
        mutation_rate = self.mutation_manager.get_rate(current_best_fitness, generation)
        
        # Balance operators if enabled
        if self.balance_operators:
            # Ensure they sum to reasonable value
            total = crossover_rate + mutation_rate
            if total > 1.0:
                scale = 1.0 / total
                crossover_rate *= scale
                mutation_rate *= scale
        
        return crossover_rate, mutation_rate
    
    def get_statistics(self) -> dict:
        """
        Get statistics about adaptive operators.
        
        Returns:
            Dictionary with operator statistics
        """
        return {
            "crossover_rate": self.crossover_manager.current_rate,
            "mutation_rate": self.mutation_manager.current_rate,
            "diversity_trend": self.crossover_manager.get_diversity_trend(),
            "stagnation_counter": self.mutation_manager.stagnation_counter,
            "diversity_history": self.crossover_manager.diversity_history.copy(),
        }
    
    def reset(self) -> None:
        """Reset all managers."""
        self.crossover_manager.reset()
        self.mutation_manager.reset()
        logger.info("Reset adaptive operator scheduler")
