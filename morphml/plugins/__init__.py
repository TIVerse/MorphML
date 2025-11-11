"""Plugin system for MorphML.

Provides extensibility through plugins for:
- Custom optimizers
- Custom evaluators
- Custom mutation operators
- Custom objectives
- Custom visualizations

Example:
    >>> from morphml.plugins import PluginManager
    >>> manager = PluginManager()
    >>> manager.load_plugin('custom_optimizer')
    >>> plugin = manager.get_plugin('custom_optimizer')
"""

from morphml.plugins.plugin_system import (
    EvaluatorPlugin,
    MutationPlugin,
    ObjectivePlugin,
    OptimizerPlugin,
    Plugin,
    PluginManager,
    VisualizationPlugin,
)

__all__ = [
    "Plugin",
    "OptimizerPlugin",
    "EvaluatorPlugin",
    "MutationPlugin",
    "ObjectivePlugin",
    "VisualizationPlugin",
    "PluginManager",
]
