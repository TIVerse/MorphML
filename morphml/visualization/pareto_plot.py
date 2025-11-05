"""Pareto front visualization utilities.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import List, Optional

from morphml.logging_config import get_logger

logger = get_logger(__name__)


def plot_pareto_front_2d(
    pareto_front: List,
    objective_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Pareto Front",
) -> None:
    """
    Plot 2D Pareto front.

    Args:
        pareto_front: List of MultiObjectiveIndividual objects
        objective_names: Names of the two objectives to plot
        save_path: Path to save plot (displays if None)
        title: Plot title

    Example:
        >>> from morphml.visualization.pareto_plot import plot_pareto_front_2d
        >>> plot_pareto_front_2d(pareto_front, ['accuracy', 'latency'])
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for plotting. Install with: pip install matplotlib")
        return

    if len(objective_names) != 2:
        logger.error("2D plot requires exactly 2 objectives")
        return

    # Extract objectives
    obj1_name, obj2_name = objective_names
    obj1_values = [ind.objectives[obj1_name] for ind in pareto_front]
    obj2_values = [ind.objectives[obj2_name] for ind in pareto_front]

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(obj1_values, obj2_values, c="blue", s=100, alpha=0.6, edgecolors="black")
    plt.xlabel(obj1_name.capitalize(), fontsize=12)
    plt.ylabel(obj2_name.capitalize(), fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Pareto front plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_pareto_front_3d(
    pareto_front: List,
    objective_names: List[str],
    save_path: Optional[str] = None,
    title: str = "3D Pareto Front",
) -> None:
    """
    Plot 3D Pareto front.

    Args:
        pareto_front: List of MultiObjectiveIndividual objects
        objective_names: Names of the three objectives to plot
        save_path: Path to save plot (displays if None)
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for plotting")
        return

    if len(objective_names) != 3:
        logger.error("3D plot requires exactly 3 objectives")
        return

    # Extract objectives
    obj1_name, obj2_name, obj3_name = objective_names
    obj1_values = [ind.objectives[obj1_name] for ind in pareto_front]
    obj2_values = [ind.objectives[obj2_name] for ind in pareto_front]
    obj3_values = [ind.objectives[obj3_name] for ind in pareto_front]

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        obj1_values, obj2_values, obj3_values, c="blue", s=100, alpha=0.6, edgecolors="black"
    )

    ax.set_xlabel(obj1_name.capitalize(), fontsize=12)
    ax.set_ylabel(obj2_name.capitalize(), fontsize=12)
    ax.set_zlabel(obj3_name.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"3D Pareto front plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_parallel_coordinates(
    pareto_front: List,
    objective_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Parallel Coordinates Plot",
) -> None:
    """
    Plot Pareto front as parallel coordinates.

    Useful for visualizing many objectives simultaneously.

    Args:
        pareto_front: List of MultiObjectiveIndividual objects
        objective_names: Names of objectives to plot
        save_path: Path to save plot (displays if None)
        title: Plot title
    """
    try:
        from morphml.optimizers.multi_objective.visualization import (
            plot_parallel_coordinates as pc_plot,
        )

        pc_plot(pareto_front, objective_names, save_path, title)
    except ImportError:
        logger.error("Multi-objective visualization module required")


# Re-export from multi_objective module for convenience
try:
    pass
except ImportError:
    pass
