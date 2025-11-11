"""Example custom evaluator plugin.

Demonstrates how to create a custom evaluator plugin for MorphML.

Usage:
    >>> from morphml.plugins import PluginManager
    >>> manager = PluginManager()
    >>> plugin = manager.load_plugin_from_file(
    ...     'morphml/plugins/custom_evaluator_example.py',
    ...     config={'latency_weight': 0.5}
    ... )
    >>> fitness = plugin.evaluate(architecture)
"""

from morphml.core.graph import ModelGraph
from morphml.plugins import EvaluatorPlugin


class Plugin(EvaluatorPlugin):
    """
    Example plugin: Multi-objective evaluator combining accuracy and latency.

    This plugin demonstrates how to:
    1. Inherit from EvaluatorPlugin
    2. Implement custom evaluation logic
    3. Combine multiple objectives
    """

    def initialize(self, config):
        """
        Initialize plugin with configuration.

        Args:
            config: Configuration dictionary with keys:
                - latency_weight: Weight for latency objective (default: 0.3)
                - accuracy_weight: Weight for accuracy objective (default: 0.7)
        """
        self.latency_weight = config.get("latency_weight", 0.3)
        self.accuracy_weight = config.get("accuracy_weight", 0.7)

        # Normalize weights
        total = self.latency_weight + self.accuracy_weight
        self.latency_weight /= total
        self.accuracy_weight /= total

        print("Initialized Multi-Objective Evaluator plugin:")
        print(f"  Accuracy weight: {self.accuracy_weight:.2f}")
        print(f"  Latency weight: {self.latency_weight:.2f}")

    def evaluate(self, architecture: ModelGraph) -> float:
        """
        Evaluate architecture with multi-objective fitness.

        Combines estimated accuracy and latency into single fitness score.

        Args:
            architecture: ModelGraph to evaluate

        Returns:
            Combined fitness score (0-1, higher is better)
        """
        # Estimate accuracy (simplified)
        depth = architecture.estimate_depth()
        width = architecture.estimate_width()
        params = architecture.estimate_parameters()

        # Heuristic accuracy estimate
        accuracy_score = min(1.0, (depth * width) / 1000.0)

        # Estimate latency (inverse of parameters, normalized)
        # Lower parameters = lower latency = higher score
        latency_score = 1.0 / (1.0 + params / 1000000.0)

        # Combine objectives
        fitness = self.accuracy_weight * accuracy_score + self.latency_weight * latency_score

        return fitness

    def cleanup(self):
        """Cleanup when plugin is unloaded."""
        print("Multi-Objective Evaluator plugin unloaded")
