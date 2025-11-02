"""Constraint predicates for architecture validation."""

from typing import List, Optional, Set

from morphml.core.graph import ModelGraph


class Constraint:
    """Base class for constraints."""

    def __init__(self, name: str):
        """Initialize constraint."""
        self.name = name

    def check(self, graph: ModelGraph) -> bool:
        """Check if graph satisfies constraint."""
        raise NotImplementedError

    def penalty(self, graph: ModelGraph) -> float:
        """Calculate penalty for constraint violation."""
        return 0.0 if self.check(graph) else 1.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class MaxParametersConstraint(Constraint):
    """Constraint on maximum parameters."""

    def __init__(self, max_params: int, name: str = "max_parameters"):
        """
        Initialize max parameters constraint.

        Args:
            max_params: Maximum allowed parameters
            name: Constraint name
        """
        super().__init__(name)
        self.max_params = max_params

    def check(self, graph: ModelGraph) -> bool:
        """Check if within parameter limit."""
        return graph.estimate_parameters() <= self.max_params

    def penalty(self, graph: ModelGraph) -> float:
        """Penalty based on excess parameters."""
        params = graph.estimate_parameters()
        if params <= self.max_params:
            return 0.0
        excess_ratio = (params - self.max_params) / self.max_params
        return min(1.0, excess_ratio)

    def __repr__(self) -> str:
        return f"MaxParametersConstraint(max={self.max_params:,})"


class MinParametersConstraint(Constraint):
    """Constraint on minimum parameters."""

    def __init__(self, min_params: int, name: str = "min_parameters"):
        """Initialize min parameters constraint."""
        super().__init__(name)
        self.min_params = min_params

    def check(self, graph: ModelGraph) -> bool:
        """Check if meets minimum parameters."""
        return graph.estimate_parameters() >= self.min_params

    def penalty(self, graph: ModelGraph) -> float:
        """Penalty based on parameter deficit."""
        params = graph.estimate_parameters()
        if params >= self.min_params:
            return 0.0
        deficit_ratio = (self.min_params - params) / self.min_params
        return min(1.0, deficit_ratio)


class DepthConstraint(Constraint):
    """Constraint on network depth."""

    def __init__(self, min_depth: int = 1, max_depth: int = 100, name: str = "depth"):
        """Initialize depth constraint."""
        super().__init__(name)
        self.min_depth = min_depth
        self.max_depth = max_depth

    def check(self, graph: ModelGraph) -> bool:
        """Check if depth is within range."""
        depth = graph.get_depth()
        return self.min_depth <= depth <= self.max_depth

    def penalty(self, graph: ModelGraph) -> float:
        """Penalty based on depth violation."""
        depth = graph.get_depth()

        if depth < self.min_depth:
            deficit = (self.min_depth - depth) / self.min_depth
            return min(1.0, deficit)
        elif depth > self.max_depth:
            excess = (depth - self.max_depth) / self.max_depth
            return min(1.0, excess)
        else:
            return 0.0


class WidthConstraint(Constraint):
    """Constraint on network width."""

    def __init__(self, min_width: int = 1, max_width: int = 100, name: str = "width"):
        """Initialize width constraint."""
        super().__init__(name)
        self.min_width = min_width
        self.max_width = max_width

    def check(self, graph: ModelGraph) -> bool:
        """Check if width is within range."""
        width = graph.get_max_width()
        return self.min_width <= width <= self.max_width

    def penalty(self, graph: ModelGraph) -> float:
        """Penalty based on width violation."""
        width = graph.get_max_width()

        if width < self.min_width:
            return (self.min_width - width) / self.min_width
        elif width > self.max_width:
            return (width - self.max_width) / self.max_width
        else:
            return 0.0


class OperationConstraint(Constraint):
    """Constraint on required/forbidden operations."""

    def __init__(
        self,
        required_ops: Optional[Set[str]] = None,
        forbidden_ops: Optional[Set[str]] = None,
        name: str = "operations",
    ):
        """
        Initialize operation constraint.

        Args:
            required_ops: Set of required operation types
            forbidden_ops: Set of forbidden operation types
            name: Constraint name
        """
        super().__init__(name)
        self.required_ops = required_ops or set()
        self.forbidden_ops = forbidden_ops or set()

    def check(self, graph: ModelGraph) -> bool:
        """Check if operations satisfy constraints."""
        graph_ops = {node.operation for node in graph.nodes.values()}

        # Check required operations
        if self.required_ops and not self.required_ops.issubset(graph_ops):
            return False

        # Check forbidden operations
        if self.forbidden_ops and self.forbidden_ops.intersection(graph_ops):
            return False

        return True

    def penalty(self, graph: ModelGraph) -> float:
        """Penalty based on operation violations."""
        graph_ops = {node.operation for node in graph.nodes.values()}

        penalty = 0.0

        # Penalty for missing required ops
        if self.required_ops:
            missing = len(self.required_ops - graph_ops)
            penalty += missing / max(1, len(self.required_ops))

        # Penalty for forbidden ops
        if self.forbidden_ops:
            forbidden_count = len(self.forbidden_ops.intersection(graph_ops))
            penalty += forbidden_count / max(1, len(self.forbidden_ops))

        return min(1.0, penalty)


class ConnectivityConstraint(Constraint):
    """Constraint on graph connectivity."""

    def __init__(
        self,
        min_edges: Optional[int] = None,
        max_edges: Optional[int] = None,
        name: str = "connectivity",
    ):
        """Initialize connectivity constraint."""
        super().__init__(name)
        self.min_edges = min_edges
        self.max_edges = max_edges

    def check(self, graph: ModelGraph) -> bool:
        """Check if connectivity is within range."""
        num_edges = len(graph.edges)

        if self.min_edges is not None and num_edges < self.min_edges:
            return False
        if self.max_edges is not None and num_edges > self.max_edges:
            return False

        return True

    def penalty(self, graph: ModelGraph) -> float:
        """Penalty based on connectivity violation."""
        num_edges = len(graph.edges)

        if self.min_edges and num_edges < self.min_edges:
            return (self.min_edges - num_edges) / self.min_edges
        elif self.max_edges and num_edges > self.max_edges:
            return (num_edges - self.max_edges) / self.max_edges
        else:
            return 0.0


class CompositeConstraint(Constraint):
    """Composite constraint combining multiple constraints."""

    def __init__(self, constraints: List[Constraint], mode: str = "all", name: str = "composite"):
        """
        Initialize composite constraint.

        Args:
            constraints: List of constraints
            mode: 'all' (AND) or 'any' (OR)
            name: Constraint name
        """
        super().__init__(name)
        self.constraints = constraints
        self.mode = mode

    def check(self, graph: ModelGraph) -> bool:
        """Check if graph satisfies composite constraint."""
        if self.mode == "all":
            return all(c.check(graph) for c in self.constraints)
        elif self.mode == "any":
            return any(c.check(graph) for c in self.constraints)
        else:
            return False

    def penalty(self, graph: ModelGraph) -> float:
        """Calculate composite penalty."""
        penalties = [c.penalty(graph) for c in self.constraints]

        if self.mode == "all":
            return sum(penalties) / len(penalties) if penalties else 0.0
        elif self.mode == "any":
            return min(penalties) if penalties else 0.0
        else:
            return 0.0
