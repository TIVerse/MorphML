"""Visualization tools for architectures and optimization."""

from morphml.visualization.architecture_plot import (
    plot_architecture,
    plot_architecture_hierarchy,
    plot_architecture_stats,
)
from morphml.visualization.convergence_plot import (
    plot_convergence,
    plot_convergence_comparison,
    plot_fitness_distribution,
)
from morphml.visualization.graph_viz import GraphVisualizer

# Phase 2: Advanced visualization
from morphml.visualization.pareto_plot import (
    plot_parallel_coordinates,
    plot_pareto_front_2d,
    plot_pareto_front_3d,
)
from morphml.visualization.population import PopulationVisualizer
from morphml.visualization.progress import ProgressPlotter

__all__ = [
    # Phase 1
    "GraphVisualizer",
    "ProgressPlotter",
    "PopulationVisualizer",
    # Phase 2: Pareto
    "plot_pareto_front_2d",
    "plot_pareto_front_3d",
    "plot_parallel_coordinates",
    # Phase 2: Convergence
    "plot_convergence",
    "plot_convergence_comparison",
    "plot_fitness_distribution",
    # Phase 2: Architecture
    "plot_architecture",
    "plot_architecture_hierarchy",
    "plot_architecture_stats",
]
