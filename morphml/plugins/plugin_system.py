"""Plugin system implementation for MorphML.

Provides base classes and plugin manager for extending MorphML functionality.

Example:
    >>> from morphml.plugins import PluginManager, OptimizerPlugin
    >>> 
    >>> class MyPlugin(OptimizerPlugin):
    ...     def initialize(self, config):
    ...         self.config = config
    ...     
    ...     def get_optimizer(self):
    ...         return MyCustomOptimizer(self.config)
    >>> 
    >>> manager = PluginManager()
    >>> manager.register_plugin('my_optimizer', MyPlugin)
    >>> plugin = manager.load_plugin('my_optimizer', config={})
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
import importlib
import importlib.util
from pathlib import Path

from morphml.logging_config import get_logger

logger = get_logger(__name__)


class Plugin(ABC):
    """
    Base class for all MorphML plugins.

    All plugins must inherit from this class and implement
    the required methods.

    Example:
        >>> class MyPlugin(Plugin):
        ...     def initialize(self, config):
        ...         self.config = config
        ...
        ...     def execute(self, context):
        ...         return self.config['value'] * 2
    """

    @abstractmethod
    def initialize(self, config: Dict[str, Any]):
        """
        Initialize plugin with configuration.

        Args:
            config: Plugin configuration dictionary
        """
        pass

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Any:
        """
        Execute plugin logic.

        Args:
            context: Execution context with relevant data

        Returns:
            Plugin execution result
        """
        pass

    def cleanup(self):
        """Optional cleanup method called when plugin is unloaded."""
        pass


class OptimizerPlugin(Plugin):
    """
    Plugin for custom optimizers.

    Allows users to add custom optimization algorithms.

    Example:
        >>> class SimulatedAnnealingPlugin(OptimizerPlugin):
        ...     def initialize(self, config):
        ...         self.temperature = config.get('temperature', 100.0)
        ...
        ...     def get_optimizer(self):
        ...         return SimulatedAnnealing(temperature=self.temperature)
    """

    @abstractmethod
    def get_optimizer(self):
        """
        Return optimizer instance.

        Returns:
            Optimizer instance (subclass of BaseOptimizer)
        """
        pass

    def execute(self, context: Dict[str, Any]):
        """Execute returns the optimizer."""
        return self.get_optimizer()


class EvaluatorPlugin(Plugin):
    """
    Plugin for custom evaluation metrics.

    Allows users to define custom architecture evaluation functions.

    Example:
        >>> class LatencyEvaluatorPlugin(EvaluatorPlugin):
        ...     def evaluate(self, architecture):
        ...         return estimate_latency(architecture)
    """

    @abstractmethod
    def evaluate(self, architecture) -> float:
        """
        Evaluate architecture.

        Args:
            architecture: ModelGraph to evaluate

        Returns:
            Fitness score
        """
        pass

    def execute(self, context: Dict[str, Any]) -> float:
        """Execute evaluation."""
        return self.evaluate(context.get("architecture"))


class MutationPlugin(Plugin):
    """
    Plugin for custom mutation operators.

    Allows users to define custom mutation strategies.

    Example:
        >>> class LayerSwapMutationPlugin(MutationPlugin):
        ...     def mutate(self, architecture):
        ...         # Swap two random layers
        ...         return mutated_architecture
    """

    @abstractmethod
    def mutate(self, architecture):
        """
        Mutate architecture.

        Args:
            architecture: ModelGraph to mutate

        Returns:
            Mutated ModelGraph
        """
        pass

    def execute(self, context: Dict[str, Any]):
        """Execute mutation."""
        return self.mutate(context.get("architecture"))


class ObjectivePlugin(Plugin):
    """
    Plugin for custom objectives in multi-objective optimization.

    Example:
        >>> class EnergyObjectivePlugin(ObjectivePlugin):
        ...     def compute(self, architecture):
        ...         return estimate_energy_consumption(architecture)
    """

    @abstractmethod
    def compute(self, architecture) -> float:
        """
        Compute objective value.

        Args:
            architecture: ModelGraph

        Returns:
            Objective value
        """
        pass

    def execute(self, context: Dict[str, Any]) -> float:
        """Execute objective computation."""
        return self.compute(context.get("architecture"))


class VisualizationPlugin(Plugin):
    """
    Plugin for custom visualizations.

    Example:
        >>> class CustomPlotPlugin(VisualizationPlugin):
        ...     def visualize(self, data):
        ...         # Create custom plot
        ...         return figure
    """

    @abstractmethod
    def visualize(self, data: Any):
        """
        Create visualization.

        Args:
            data: Data to visualize

        Returns:
            Visualization object (e.g., matplotlib Figure)
        """
        pass

    def execute(self, context: Dict[str, Any]):
        """Execute visualization."""
        return self.visualize(context.get("data"))


class PluginManager:
    """
    Manage and load plugins.

    Supports loading plugins from:
    - Built-in morphml.plugins module
    - External Python files
    - Installed packages with entry points

    Example:
        >>> manager = PluginManager()
        >>> manager.register_plugin('my_optimizer', MyOptimizerPlugin)
        >>> plugin = manager.load_plugin('my_optimizer', config={'lr': 0.01})
        >>> optimizer = plugin.get_optimizer()
    """

    def __init__(self):
        """Initialize plugin manager."""
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_classes: Dict[str, Type[Plugin]] = {}
        logger.info("Initialized PluginManager")

    def register_plugin(self, name: str, plugin_class: Type[Plugin]):
        """
        Register a plugin class.

        Args:
            name: Plugin name
            plugin_class: Plugin class (subclass of Plugin)

        Example:
            >>> manager.register_plugin('my_plugin', MyPlugin)
        """
        if not issubclass(plugin_class, Plugin):
            raise TypeError(f"{plugin_class} must be a subclass of Plugin")

        self.plugin_classes[name] = plugin_class
        logger.info(f"Registered plugin class: {name}")

    def load_plugin(self, name: str, config: Optional[Dict[str, Any]] = None) -> Plugin:
        """
        Load and initialize a plugin.

        Args:
            name: Plugin name
            config: Plugin configuration

        Returns:
            Initialized plugin instance

        Example:
            >>> plugin = manager.load_plugin('my_plugin', {'param': 'value'})
        """
        if name in self.plugins:
            logger.warning(f"Plugin {name} already loaded, returning existing instance")
            return self.plugins[name]

        # Try registered plugins first
        if name in self.plugin_classes:
            plugin_class = self.plugin_classes[name]
        else:
            # Try loading from morphml.plugins module
            plugin_class = self._load_plugin_from_module(name)

        if plugin_class is None:
            raise ValueError(f"Plugin {name} not found")

        # Initialize plugin
        plugin = plugin_class()
        plugin.initialize(config or {})

        self.plugins[name] = plugin
        logger.info(f"Loaded plugin: {name}")

        return plugin

    def _load_plugin_from_module(self, name: str) -> Optional[Type[Plugin]]:
        """
        Load plugin from morphml.plugins module.

        Args:
            name: Plugin name

        Returns:
            Plugin class or None
        """
        try:
            module = importlib.import_module(f"morphml.plugins.{name}")

            # Look for Plugin class
            if hasattr(module, "Plugin"):
                return getattr(module, "Plugin")

            # Look for class with matching name
            class_name = "".join(word.capitalize() for word in name.split("_"))
            if hasattr(module, class_name):
                return getattr(module, class_name)

            logger.warning(f"No Plugin class found in morphml.plugins.{name}")
            return None

        except ImportError as e:
            logger.debug(f"Could not import morphml.plugins.{name}: {e}")
            return None

    def load_plugin_from_file(
        self, filepath: str, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ) -> Plugin:
        """
        Load plugin from Python file.

        Args:
            filepath: Path to Python file
            name: Optional plugin name (defaults to filename)
            config: Plugin configuration

        Returns:
            Initialized plugin instance

        Example:
            >>> plugin = manager.load_plugin_from_file(
            ...     'my_plugin.py',
            ...     name='my_plugin',
            ...     config={'param': 'value'}
            ... )
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Plugin file not found: {filepath}")

        # Use filename as plugin name if not provided
        if name is None:
            name = path.stem

        # Load module from file
        spec = importlib.util.spec_from_file_location(name, filepath)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {filepath}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find Plugin class
        if hasattr(module, "Plugin"):
            plugin_class = getattr(module, "Plugin")
        else:
            raise AttributeError(f"No Plugin class found in {filepath}")

        # Initialize and register
        plugin = plugin_class()
        plugin.initialize(config or {})

        self.plugins[name] = plugin
        logger.info(f"Loaded plugin from file: {filepath}")

        return plugin

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Get loaded plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None

        Example:
            >>> plugin = manager.get_plugin('my_plugin')
        """
        return self.plugins.get(name)

    def unload_plugin(self, name: str):
        """
        Unload a plugin.

        Args:
            name: Plugin name

        Example:
            >>> manager.unload_plugin('my_plugin')
        """
        if name in self.plugins:
            plugin = self.plugins[name]
            plugin.cleanup()
            del self.plugins[name]
            logger.info(f"Unloaded plugin: {name}")

    def list_plugins(self) -> List[str]:
        """
        List all loaded plugins.

        Returns:
            List of plugin names

        Example:
            >>> plugins = manager.list_plugins()
            >>> print(plugins)
            ['optimizer_plugin', 'evaluator_plugin']
        """
        return list(self.plugins.keys())

    def list_registered_classes(self) -> List[str]:
        """
        List all registered plugin classes.

        Returns:
            List of registered class names
        """
        return list(self.plugin_classes.keys())


# Global plugin manager instance
_global_manager = None


def get_plugin_manager() -> PluginManager:
    """
    Get global plugin manager instance.

    Returns:
        Global PluginManager instance

    Example:
        >>> manager = get_plugin_manager()
        >>> manager.load_plugin('my_plugin')
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = PluginManager()
    return _global_manager
