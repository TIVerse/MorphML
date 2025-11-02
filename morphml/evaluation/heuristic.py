"""Heuristic evaluators for fast architecture assessment.

These evaluators estimate architecture quality without training,
enabling rapid architecture search and development.

Example:
    >>> from morphml.evaluation import HeuristicEvaluator
    >>> 
    >>> evaluator = HeuristicEvaluator()
    >>> score = evaluator.combined_score(graph)
"""

from typing import Dict

from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class HeuristicEvaluator:
    """
    Fast heuristic evaluation without training.

    Provides multiple proxy metrics that correlate with actual
    performance, enabling rapid architecture assessment.

    Metrics:
    - Parameter count
    - FLOPs estimation
    - Graph depth
    - Graph width
    - Connectivity ratio

    Example:
        >>> evaluator = HeuristicEvaluator(
        ...     param_weight=0.3,
        ...     depth_weight=0.3,
        ...     width_weight=0.2
        ... )
        >>> score = evaluator(graph)
    """

    def __init__(
        self,
        param_weight: float = 0.3,
        depth_weight: float = 0.3,
        width_weight: float = 0.2,
        connectivity_weight: float = 0.2,
        target_params: int = 1000000,
        target_depth: int = 20,
    ):
        """
        Initialize heuristic evaluator.

        Args:
            param_weight: Weight for parameter penalty
            depth_weight: Weight for depth score
            width_weight: Weight for width score
            connectivity_weight: Weight for connectivity
            target_params: Target parameter count
            target_depth: Target network depth
        """
        self.param_weight = param_weight
        self.depth_weight = depth_weight
        self.width_weight = width_weight
        self.connectivity_weight = connectivity_weight
        self.target_params = target_params
        self.target_depth = target_depth

    def __call__(self, graph: ModelGraph) -> float:
        """Evaluate using combined score."""
        return self.combined_score(graph)

    def parameter_score(self, graph: ModelGraph) -> float:
        """
        Score based on parameter count.

        Penalizes large models, rewards compact ones.

        Args:
            graph: Architecture to evaluate

        Returns:
            Score in [0, 1]
        """
        params = graph.estimate_parameters()

        # Sigmoid-like scoring
        ratio = params / self.target_params
        if ratio < 0.5:
            score = 1.0
        elif ratio < 1.0:
            score = 1.0 - 0.5 * (ratio - 0.5)
        elif ratio < 2.0:
            score = 0.75 - 0.5 * (ratio - 1.0)
        else:
            score = max(0.1, 0.25 - 0.1 * (ratio - 2.0))

        return score

    def depth_score(self, graph: ModelGraph) -> float:
        """
        Score based on network depth.

        Moderate depth is preferred.

        Args:
            graph: Architecture to evaluate

        Returns:
            Score in [0, 1]
        """
        depth = graph.get_depth()

        if depth < 5:
            score = 0.5 + 0.1 * depth  # Too shallow
        elif depth <= self.target_depth:
            score = 0.8 + 0.01 * depth  # Good range
        else:
            score = max(0.3, 1.0 - 0.02 * (depth - self.target_depth))  # Too deep

        return min(1.0, score)

    def width_score(self, graph: ModelGraph) -> float:
        """
        Score based on network width.

        Moderate width is preferred.

        Args:
            graph: Architecture to evaluate

        Returns:
            Score in [0, 1]
        """
        width = graph.get_max_width()

        if width < 3:
            score = 0.5 + 0.1 * width
        elif width <= 10:
            score = 0.8 + 0.02 * width
        else:
            score = max(0.5, 1.0 - 0.02 * (width - 10))

        return min(1.0, score)

    def connectivity_score(self, graph: ModelGraph) -> float:
        """
        Score based on connectivity ratio.

        Measures how well-connected the graph is.

        Args:
            graph: Architecture to evaluate

        Returns:
            Score in [0, 1]
        """
        num_nodes = len(graph.nodes)
        num_edges = len(graph.edges)

        if num_nodes <= 1:
            return 0.5

        # Expected edges for DAG: between n-1 (linear) and n*(n-1)/2 (complete)
        min_edges = num_nodes - 1
        max_edges = num_nodes * (num_nodes - 1) / 2

        if num_edges < min_edges:
            return 0.0  # Disconnected

        # Normalize
        connectivity_ratio = (num_edges - min_edges) / (max_edges - min_edges + 1)

        # Prefer moderate connectivity
        if connectivity_ratio < 0.3:
            score = 0.6 + connectivity_ratio
        elif connectivity_ratio < 0.7:
            score = 0.9
        else:
            score = 0.9 - 0.5 * (connectivity_ratio - 0.7)

        return score

    def combined_score(self, graph: ModelGraph) -> float:
        """
        Combined heuristic score.

        Weighted combination of all metrics.

        Args:
            graph: Architecture to evaluate

        Returns:
            Combined score in [0, 1]
        """
        param_score = self.parameter_score(graph)
        depth_score = self.depth_score(graph)
        width_score = self.width_score(graph)
        connectivity_score = self.connectivity_score(graph)

        combined = (
            self.param_weight * param_score
            + self.depth_weight * depth_score
            + self.width_weight * width_score
            + self.connectivity_weight * connectivity_score
        )

        # Ensure in [0, 1]
        combined = max(0.0, min(1.0, combined))

        logger.debug(
            f"Heuristic scores: param={param_score:.3f}, depth={depth_score:.3f}, "
            f"width={width_score:.3f}, connectivity={connectivity_score:.3f}, "
            f"combined={combined:.3f}"
        )

        return combined

    def get_all_scores(self, graph: ModelGraph) -> Dict[str, float]:
        """
        Get all individual scores.

        Args:
            graph: Architecture to evaluate

        Returns:
            Dictionary of all scores
        """
        return {
            "parameter": self.parameter_score(graph),
            "depth": self.depth_score(graph),
            "width": self.width_score(graph),
            "connectivity": self.connectivity_score(graph),
            "combined": self.combined_score(graph),
        }
