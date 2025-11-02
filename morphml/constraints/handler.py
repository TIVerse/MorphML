"""Constraint handler for managing architecture constraints."""

from typing import Dict, List

from morphml.constraints.predicates import Constraint
from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class ConstraintHandler:
    """
    Manages and evaluates constraints on architectures.

    Example:
        >>> handler = ConstraintHandler()
        >>> handler.add_constraint(MaxParametersConstraint(1000000))
        >>> handler.add_constraint(DepthConstraint(min_depth=5, max_depth=20))
        >>>
        >>> if handler.check(graph):
        ...     print("Valid!")
        >>>
        >>> penalty = handler.total_penalty(graph)
    """

    def __init__(self):
        """Initialize constraint handler."""
        self.constraints: List[Constraint] = []
        logger.debug("Created ConstraintHandler")

    def add_constraint(self, constraint: Constraint) -> None:
        """
        Add a constraint.

        Args:
            constraint: Constraint to add
        """
        self.constraints.append(constraint)
        logger.debug(f"Added constraint: {constraint}")

    def remove_constraint(self, name: str) -> None:
        """
        Remove a constraint by name.

        Args:
            name: Constraint name
        """
        self.constraints = [c for c in self.constraints if c.name != name]
        logger.debug(f"Removed constraint: {name}")

    def check(self, graph: ModelGraph) -> bool:
        """
        Check if graph satisfies all constraints.

        Args:
            graph: Graph to check

        Returns:
            True if all constraints satisfied
        """
        for constraint in self.constraints:
            if not constraint.check(graph):
                logger.debug(f"Constraint violated: {constraint.name}")
                return False
        return True

    def get_violations(self, graph: ModelGraph) -> List[str]:
        """
        Get list of violated constraints.

        Args:
            graph: Graph to check

        Returns:
            List of violated constraint names
        """
        violations = []
        for constraint in self.constraints:
            if not constraint.check(graph):
                violations.append(constraint.name)
        return violations

    def total_penalty(self, graph: ModelGraph) -> float:
        """
        Calculate total penalty for all constraints.

        Args:
            graph: Graph to evaluate

        Returns:
            Total penalty (0 = all constraints satisfied)
        """
        if not self.constraints:
            return 0.0

        total = sum(c.penalty(graph) for c in self.constraints)
        return total / len(self.constraints)

    def get_penalties(self, graph: ModelGraph) -> Dict[str, float]:
        """
        Get penalties for each constraint.

        Args:
            graph: Graph to evaluate

        Returns:
            Dictionary mapping constraint names to penalties
        """
        return {c.name: c.penalty(graph) for c in self.constraints}

    def apply_penalty_to_fitness(
        self, fitness: float, graph: ModelGraph, penalty_weight: float = 0.5
    ) -> float:
        """
        Apply constraint penalties to fitness score.

        Args:
            fitness: Original fitness
            graph: Architecture graph
            penalty_weight: Weight for penalty term

        Returns:
            Penalized fitness
        """
        penalty = self.total_penalty(graph)
        return fitness * (1 - penalty_weight * penalty)

    def clear(self) -> None:
        """Remove all constraints."""
        self.constraints.clear()
        logger.debug("Cleared all constraints")

    def __len__(self) -> int:
        """Return number of constraints."""
        return len(self.constraints)

    def __repr__(self) -> str:
        return f"ConstraintHandler(constraints={len(self.constraints)})"
