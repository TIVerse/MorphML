"""Individual representation for evolutionary algorithms.

An Individual wraps a neural architecture (ModelGraph) with fitness tracking
and metadata for use in population-based optimization.

Example:
    >>> from morphml.core.graph import ModelGraph
    >>> from morphml.core.search import Individual
    >>>
    >>> graph = ModelGraph()
    >>> # ... build graph ...
    >>>
    >>> individual = Individual(graph)
    >>> individual.fitness = 0.95
    >>> individual.metadata['accuracy'] = 0.95
    >>> individual.metadata['latency'] = 12.3
"""

import time
from typing import Any, Dict, Optional

from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class Individual:
    """
    Represents a single architecture in the population.

    An Individual consists of:
    - A neural architecture (ModelGraph)
    - Fitness score (objective value)
    - Metadata (metrics, evaluation info, etc.)
    - Age (number of generations alive)

    Attributes:
        graph: The neural architecture
        fitness: Fitness score (higher is better)
        metadata: Additional information
        age: Number of generations this individual has survived
        parent_ids: IDs of parent individuals (for genealogy)
        birth_generation: Generation when created

    Example:
        >>> individual = Individual(graph, fitness=0.92)
        >>> individual.metadata['accuracy'] = 0.92
        >>> individual.metadata['params'] = 1000000
    """

    def __init__(
        self,
        graph: ModelGraph,
        fitness: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_ids: Optional[list] = None,
    ):
        """
        Initialize individual.

        Args:
            graph: Neural architecture
            fitness: Initial fitness score
            metadata: Additional metadata
            parent_ids: List of parent individual IDs
        """
        self.graph = graph
        self.fitness = fitness
        self.metadata = metadata or {}
        self.age = 0
        self.parent_ids = parent_ids or []
        self.birth_generation = 0

        # Generate unique ID (with microseconds for better uniqueness)
        import random

        self.id = f"{graph.hash()[:16]}_{int(time.time() * 1000000)}_{random.randint(0, 9999)}"

        # Track evaluation status
        self._evaluated = fitness is not None

        logger.debug(f"Created Individual: {self.id[:12]}")

    def is_evaluated(self) -> bool:
        """
        Check if individual has been evaluated.

        Returns:
            True if fitness has been set
        """
        return self._evaluated

    def set_fitness(self, fitness: float, **metrics: Any) -> None:
        """
        Set fitness and optional metrics.

        Args:
            fitness: Fitness score
            **metrics: Additional metrics to store in metadata

        Example:
            >>> individual.set_fitness(0.95, accuracy=0.95, loss=0.05)
        """
        self.fitness = fitness
        self._evaluated = True

        # Store metrics
        for key, value in metrics.items():
            self.metadata[key] = value

        logger.debug(f"Individual {self.id[:12]} fitness set to {fitness:.4f}")

    def increment_age(self) -> None:
        """Increment age by 1 generation."""
        self.age += 1

    def clone(self, keep_fitness: bool = False) -> "Individual":
        """
        Create a copy of this individual.

        Args:
            keep_fitness: Whether to copy fitness score

        Returns:
            New Individual instance
        """
        new_graph = self.graph.clone()
        new_fitness = self.fitness if keep_fitness else None
        new_metadata = self.metadata.copy()

        new_individual = Individual(
            graph=new_graph,
            fitness=new_fitness,
            metadata=new_metadata,
            parent_ids=[self.id],
        )

        new_individual.age = 0  # Reset age

        return new_individual

    def get_metric(self, key: str, default: Any = None) -> Any:
        """
        Get a metric from metadata.

        Args:
            key: Metric key
            default: Default value if not found

        Returns:
            Metric value
        """
        return self.metadata.get(key, default)

    def dominates(self, other: "Individual", objectives: list) -> bool:
        """
        Check if this individual dominates another (for multi-objective).

        Args:
            other: Other individual
            objectives: List of objective names to compare

        Returns:
            True if this individual dominates the other
        """
        if not self.is_evaluated() or not other.is_evaluated():
            return False

        better_in_any = False

        for obj in objectives:
            self_val = self.get_metric(obj, 0.0)
            other_val = other.get_metric(obj, 0.0)

            if self_val < other_val:
                return False  # Worse in this objective
            if self_val > other_val:
                better_in_any = True

        return better_in_any

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "graph": self.graph.to_dict(),
            "fitness": self.fitness,
            "metadata": self.metadata,
            "age": self.age,
            "parent_ids": self.parent_ids,
            "birth_generation": self.birth_generation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Individual":
        """
        Deserialize from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Individual instance
        """
        graph = ModelGraph.from_dict(data["graph"])

        individual = cls(
            graph=graph,
            fitness=data.get("fitness"),
            metadata=data.get("metadata", {}),
            parent_ids=data.get("parent_ids", []),
        )

        individual.id = data["id"]
        individual.age = data.get("age", 0)
        individual.birth_generation = data.get("birth_generation", 0)

        return individual

    def __repr__(self) -> str:
        """String representation."""
        fitness_str = f"{self.fitness:.4f}" if self.fitness is not None else "N/A"
        return (
            f"Individual(id={self.id[:12]}, "
            f"fitness={fitness_str}, "
            f"age={self.age}, "
            f"nodes={len(self.graph.nodes)})"
        )

    def __lt__(self, other: "Individual") -> bool:
        """Less than comparison based on fitness (for sorting)."""
        if self.fitness is None:
            return True
        if other.fitness is None:
            return False
        return self.fitness < other.fitness

    def __eq__(self, other: object) -> bool:
        """Equality based on ID."""
        if not isinstance(other, Individual):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)
