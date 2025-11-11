"""Example custom optimizer plugin.

Demonstrates how to create a custom optimizer plugin for MorphML.

Usage:
    >>> from morphml.plugins import PluginManager
    >>> manager = PluginManager()
    >>> plugin = manager.load_plugin_from_file(
    ...     'morphml/plugins/custom_optimizer_example.py',
    ...     config={'temperature': 100.0, 'cooling_rate': 0.95}
    ... )
    >>> optimizer = plugin.get_optimizer()
"""

from morphml.plugins import OptimizerPlugin
from morphml.optimizers.simulated_annealing import SimulatedAnnealing


class Plugin(OptimizerPlugin):
    """
    Example plugin: Simulated Annealing optimizer with custom configuration.

    This plugin demonstrates how to:
    1. Inherit from OptimizerPlugin
    2. Initialize with custom configuration
    3. Return a configured optimizer instance
    """

    def initialize(self, config):
        """
        Initialize plugin with configuration.

        Args:
            config: Configuration dictionary with keys:
                - temperature: Initial temperature (default: 100.0)
                - cooling_rate: Cooling rate (default: 0.95)
                - max_iterations: Maximum iterations (default: 1000)
        """
        self.temperature = config.get("temperature", 100.0)
        self.cooling_rate = config.get("cooling_rate", 0.95)
        self.max_iterations = config.get("max_iterations", 1000)

        print(f"Initialized SimulatedAnnealing plugin:")
        print(f"  Temperature: {self.temperature}")
        print(f"  Cooling rate: {self.cooling_rate}")
        print(f"  Max iterations: {self.max_iterations}")

    def get_optimizer(self):
        """
        Return configured SimulatedAnnealing optimizer.

        Returns:
            SimulatedAnnealing instance
        """
        return SimulatedAnnealing(
            temperature=self.temperature,
            cooling_rate=self.cooling_rate,
            max_iterations=self.max_iterations,
        )

    def cleanup(self):
        """Cleanup when plugin is unloaded."""
        print("SimulatedAnnealing plugin unloaded")
