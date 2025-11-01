"""Population management for evolutionary algorithms.

Manages a collection of individuals with selection, sorting, and diversity tracking.

Example:
    >>> from morphml.core.search import Population, Individual
    >>>
    >>> population = Population(max_size=50)
    >>> population.add(individual1)
    >>> population.add(individual2)
    >>>
    >>> # Get best individuals
    >>> best = population.get_best(n=10)
    >>>
    >>> # Select for breeding
    >>> parents = population.select(n=20, method='tournament')
"""

import random
from typing import Any, Dict, List, Optional

from morphml.core.search.individual import Individual
from morphml.exceptions import SearchSpaceError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class Population:
    """
    Manages a collection of individuals in evolutionary search.

    Provides methods for:
    - Adding/removing individuals
    - Selection strategies
    - Sorting and filtering
    - Diversity metrics
    - Statistics tracking

    Attributes:
        max_size: Maximum population size
        individuals: List of Individual instances
        generation: Current generation number
        history: Historical statistics

    Example:
        >>> pop = Population(max_size=100)
        >>> pop.add_many(initial_individuals)
        >>> best = pop.get_best(n=10)
        >>> parents = pop.select(n=20, method='tournament', k=3)
    """

    def __init__(self, max_size: int = 100, elitism: int = 5):
        """
        Initialize population.

        Args:
            max_size: Maximum population size
            elitism: Number of best individuals to always keep
        """
        self.max_size = max_size
        self.elitism = elitism
        self.individuals: List[Individual] = []
        self.generation = 0
        self.history: List[Dict[str, Any]] = []

        logger.debug(f"Created Population: max_size={max_size}, elitism={elitism}")

    def add(self, individual: Individual) -> None:
        """
        Add an individual to the population.

        Args:
            individual: Individual to add
        """
        self.individuals.append(individual)
        individual.birth_generation = self.generation

    def add_many(self, individuals: List[Individual]) -> None:
        """Add multiple individuals."""
        for ind in individuals:
            self.add(ind)

    def remove(self, individual: Individual) -> None:
        """Remove an individual from the population."""
        if individual in self.individuals:
            self.individuals.remove(individual)

    def clear(self) -> None:
        """Remove all individuals."""
        self.individuals.clear()

    def size(self) -> int:
        """Get current population size."""
        return len(self.individuals)

    def is_full(self) -> bool:
        """Check if population is at maximum size."""
        return self.size() >= self.max_size

    def get_best(self, n: int = 1) -> List[Individual]:
        """
        Get the n best individuals.

        Args:
            n: Number of individuals to return

        Returns:
            List of top n individuals sorted by fitness
        """
        evaluated = [ind for ind in self.individuals if ind.is_evaluated()]
        sorted_inds = sorted(evaluated, key=lambda x: x.fitness or 0, reverse=True)
        return sorted_inds[:n]

    def get_worst(self, n: int = 1) -> List[Individual]:
        """Get the n worst individuals."""
        evaluated = [ind for ind in self.individuals if ind.is_evaluated()]
        sorted_inds = sorted(evaluated, key=lambda x: x.fitness or 0)
        return sorted_inds[:n]

    def get_unevaluated(self) -> List[Individual]:
        """Get all unevaluated individuals."""
        return [ind for ind in self.individuals if not ind.is_evaluated()]

    def select(
        self,
        n: int,
        method: str = "tournament",
        **kwargs: Any,
    ) -> List[Individual]:
        """
        Select individuals for breeding.

        Args:
            n: Number of individuals to select
            method: Selection method ('tournament', 'roulette', 'rank', 'random')
            **kwargs: Method-specific parameters

        Returns:
            List of selected individuals

        Example:
            >>> parents = pop.select(20, method='tournament', k=3)
        """
        if method == "tournament":
            return self._tournament_selection(n, k=kwargs.get("k", 3))
        elif method == "roulette":
            return self._roulette_selection(n)
        elif method == "rank":
            return self._rank_selection(n)
        elif method == "random":
            return self._random_selection(n)
        else:
            raise SearchSpaceError(f"Unknown selection method: {method}")

    def _tournament_selection(self, n: int, k: int = 3) -> List[Individual]:
        """
        Tournament selection.

        Args:
            n: Number of individuals to select
            k: Tournament size

        Returns:
            Selected individuals
        """
        evaluated = [ind for ind in self.individuals if ind.is_evaluated()]

        if len(evaluated) < k:
            logger.warning(f"Not enough evaluated individuals for tournament size {k}")
            k = max(1, len(evaluated))

        selected = []
        for _ in range(n):
            # Run tournament
            tournament = random.sample(evaluated, k)
            winner = max(tournament, key=lambda x: x.fitness or 0)
            selected.append(winner)

        return selected

    def _roulette_selection(self, n: int) -> List[Individual]:
        """
        Roulette wheel selection (fitness-proportionate).

        Args:
            n: Number of individuals to select

        Returns:
            Selected individuals
        """
        evaluated = [ind for ind in self.individuals if ind.is_evaluated()]

        if not evaluated:
            return []

        # Shift fitnesses to be positive
        min_fitness = min(ind.fitness or 0 for ind in evaluated)
        if min_fitness < 0:
            fitnesses = [(ind.fitness or 0) - min_fitness + 1e-6 for ind in evaluated]
        else:
            fitnesses = [ind.fitness or 1e-6 for ind in evaluated]

        # Normalize to probabilities
        total_fitness = sum(fitnesses)
        probabilities = [f / total_fitness for f in fitnesses]

        # Select with replacement
        selected = random.choices(evaluated, weights=probabilities, k=n)

        return selected

    def _rank_selection(self, n: int) -> List[Individual]:
        """
        Rank-based selection.

        Args:
            n: Number of individuals to select

        Returns:
            Selected individuals
        """
        evaluated = [ind for ind in self.individuals if ind.is_evaluated()]

        if not evaluated:
            return []

        # Sort by fitness
        sorted_inds = sorted(evaluated, key=lambda x: x.fitness or 0)

        # Assign ranks (linear ranking)
        ranks = list(range(1, len(sorted_inds) + 1))

        # Select based on ranks
        selected = random.choices(sorted_inds, weights=ranks, k=n)

        return selected

    def _random_selection(self, n: int) -> List[Individual]:
        """Random selection."""
        evaluated = [ind for ind in self.individuals if ind.is_evaluated()]
        return random.sample(evaluated, min(n, len(evaluated)))

    def trim(self, target_size: Optional[int] = None) -> None:
        """
        Trim population to target size, keeping best individuals.

        Args:
            target_size: Target size (defaults to max_size)
        """
        target_size = target_size or self.max_size

        if self.size() <= target_size:
            return

        # Always keep elite
        elite = self.get_best(self.elitism)
        elite_ids = {ind.id for ind in elite}

        # Get remaining individuals
        others = [ind for ind in self.individuals if ind.id not in elite_ids]

        # Sort others by fitness
        others_evaluated = [ind for ind in others if ind.is_evaluated()]
        [ind for ind in others if not ind.is_evaluated()]

        others_evaluated.sort(key=lambda x: x.fitness or 0, reverse=True)

        # Keep best from others
        remaining_slots = target_size - len(elite)
        keep_others = others_evaluated[:remaining_slots]

        # Update population
        self.individuals = elite + keep_others

        logger.debug(f"Trimmed population to {len(self.individuals)} individuals")

    def increment_ages(self) -> None:
        """Increment age of all individuals."""
        for ind in self.individuals:
            ind.increment_age()

    def next_generation(self) -> None:
        """Advance to next generation."""
        self.generation += 1
        self.increment_ages()

        # Record statistics
        stats = self.get_statistics()
        self.history.append(stats)

        logger.info(
            f"Generation {self.generation}: "
            f"size={self.size()}, "
            f"best={stats.get('best_fitness', 0):.4f}, "
            f"mean={stats.get('mean_fitness', 0):.4f}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get population statistics.

        Returns:
            Dictionary of statistics
        """
        evaluated = [ind for ind in self.individuals if ind.is_evaluated()]

        if not evaluated:
            return {
                "generation": self.generation,
                "size": self.size(),
                "evaluated": 0,
            }

        fitnesses = [ind.fitness for ind in evaluated if ind.fitness is not None]

        stats = {
            "generation": self.generation,
            "size": self.size(),
            "evaluated": len(evaluated),
            "best_fitness": max(fitnesses) if fitnesses else 0,
            "worst_fitness": min(fitnesses) if fitnesses else 0,
            "mean_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            "median_fitness": sorted(fitnesses)[len(fitnesses) // 2] if fitnesses else 0,
        }

        return stats

    def get_diversity(self, method: str = "hash") -> float:
        """
        Calculate population diversity.

        Args:
            method: Diversity metric ('hash', 'hamming', 'depth')

        Returns:
            Diversity score (0-1, higher = more diverse)
        """
        if self.size() <= 1:
            return 0.0

        if method == "hash":
            # Count unique graph hashes
            hashes = {ind.graph.hash() for ind in self.individuals}
            return len(hashes) / self.size()

        elif method == "depth":
            # Variance in graph depths
            depths = [ind.graph.get_depth() for ind in self.individuals]
            if len(set(depths)) == 1:
                return 0.0
            mean_depth = sum(depths) / len(depths)
            variance = sum((d - mean_depth) ** 2 for d in depths) / len(depths)
            # Normalize
            return min(1.0, variance / (mean_depth + 1))

        else:
            return 0.0

    def __len__(self) -> int:
        """Return population size."""
        return self.size()

    def __iter__(self) -> Any:
        """Iterate over individuals."""
        return iter(self.individuals)

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"Population(generation={self.generation}, "
            f"size={self.size()}/{self.max_size}, "
            f"best_fitness={stats.get('best_fitness', 0):.4f})"
        )
