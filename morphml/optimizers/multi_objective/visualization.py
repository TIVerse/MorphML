"""Visualization tools for multi-objective optimization results.

Provides plotting functions for Pareto fronts, objective trade-offs,
and optimization convergence.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import List, Optional

import numpy as np

from morphml.logging_config import get_logger
from morphml.optimizers.multi_objective.nsga2 import MultiObjectiveIndividual

logger = get_logger(__name__)


class ParetoVisualizer:
    """
    Visualization tools for Pareto fronts.

    Provides various plotting methods for multi-objective optimization results.

    Example:
        >>> visualizer = ParetoVisualizer()
        >>> visualizer.plot_2d(pareto_front, 'accuracy', 'latency')
        >>> visualizer.plot_3d(pareto_front, 'accuracy', 'latency', 'params')
    """

    @staticmethod
    def plot_2d(
        pareto_front: List[MultiObjectiveIndividual],
        obj1_name: str,
        obj2_name: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_labels: bool = False,
    ) -> None:
        """
        Plot 2D Pareto front.

        Args:
            pareto_front: List of Pareto-optimal individuals
            obj1_name: Name of first objective (x-axis)
            obj2_name: Name of second objective (y-axis)
            title: Optional plot title
            save_path: Optional path to save figure
            show_labels: Whether to show point labels

        Example:
            >>> ParetoVisualizer.plot_2d(
            ...     pareto_front,
            ...     obj1_name='accuracy',
            ...     obj2_name='latency',
            ...     save_path='pareto_front.png'
            ... )
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, cannot plot")
            return

        # Extract objective values
        x = [ind.objectives[obj1_name] for ind in pareto_front]
        y = [ind.objectives[obj2_name] for ind in pareto_front]

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, s=100, alpha=0.6, c="blue", edgecolors="black", linewidth=1.5)

        # Connect points to show Pareto front
        sorted_indices = np.argsort(x)
        x_sorted = [x[i] for i in sorted_indices]
        y_sorted = [y[i] for i in sorted_indices]
        plt.plot(x_sorted, y_sorted, "r--", alpha=0.3, linewidth=1)

        # Labels
        plt.xlabel(obj1_name.replace("_", " ").title(), fontsize=14, fontweight="bold")
        plt.ylabel(obj2_name.replace("_", " ").title(), fontsize=14, fontweight="bold")

        if title is None:
            title = f"Pareto Front: {obj1_name} vs {obj2_name}"
        plt.title(title, fontsize=16, fontweight="bold")

        # Add labels if requested
        if show_labels:
            for i, (xi, yi) in enumerate(zip(x, y)):
                plt.annotate(f"{i}", (xi, yi), fontsize=8, alpha=0.7)

        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_3d(
        pareto_front: List[MultiObjectiveIndividual],
        obj1_name: str,
        obj2_name: str,
        obj3_name: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot 3D Pareto front.

        Args:
            pareto_front: List of Pareto-optimal individuals
            obj1_name: Name of first objective (x-axis)
            obj2_name: Name of second objective (y-axis)
            obj3_name: Name of third objective (z-axis)
            title: Optional plot title
            save_path: Optional path to save figure

        Example:
            >>> ParetoVisualizer.plot_3d(
            ...     pareto_front,
            ...     'accuracy', 'latency', 'params'
            ... )
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, cannot plot")
            return

        # Extract objective values
        x = [ind.objectives[obj1_name] for ind in pareto_front]
        y = [ind.objectives[obj2_name] for ind in pareto_front]
        z = [ind.objectives[obj3_name] for ind in pareto_front]

        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        # Scatter plot colored by third objective
        scatter = ax.scatter(
            x, y, z, s=100, c=z, cmap="viridis", alpha=0.7, edgecolors="black", linewidth=1
        )

        # Labels
        ax.set_xlabel(obj1_name.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_ylabel(obj2_name.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_zlabel(obj3_name.replace("_", " ").title(), fontsize=12, fontweight="bold")

        if title is None:
            title = f"3D Pareto Front: {obj1_name}, {obj2_name}, {obj3_name}"
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label(obj3_name.replace("_", " ").title(), fontsize=11)

        # Grid
        ax.grid(True, alpha=0.3)

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_parallel_coordinates(
        pareto_front: List[MultiObjectiveIndividual],
        objective_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Parallel coordinates plot for multiple objectives.

        Each line represents a solution, each axis represents an objective.
        Useful for visualizing high-dimensional Pareto fronts.

        Args:
            pareto_front: List of Pareto-optimal individuals
            objective_names: List of objectives to plot (None = all)
            title: Optional plot title
            save_path: Optional path to save figure

        Example:
            >>> ParetoVisualizer.plot_parallel_coordinates(
            ...     pareto_front,
            ...     objective_names=['accuracy', 'latency', 'params', 'flops']
            ... )
        """
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            from pandas.plotting import parallel_coordinates
        except ImportError:
            logger.warning("matplotlib/pandas not available, cannot plot")
            return

        # Extract objective values
        if objective_names is None:
            objective_names = list(pareto_front[0].objectives.keys())

        data = []
        for i, ind in enumerate(pareto_front):
            row = {name: ind.objectives[name] for name in objective_names}
            row["solution"] = i
            data.append(row)

        df = pd.DataFrame(data)

        # Normalize objectives to [0, 1] for better visualization
        for col in objective_names:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)

        # Create plot
        plt.figure(figsize=(14, 7))
        parallel_coordinates(df, "solution", colormap="viridis", alpha=0.6, linewidth=1.5)

        # Labels
        if title is None:
            title = "Pareto Front - Parallel Coordinates"
        plt.title(title, fontsize=16, fontweight="bold")

        plt.ylabel("Normalized Objective Value", fontsize=12)
        plt.legend().remove()  # Remove legend (too many solutions)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_convergence(history: List[dict], save_path: Optional[str] = None) -> None:
        """
        Plot optimization convergence over generations.

        Shows how Pareto front size and quality evolve.

        Args:
            history: Optimization history from NSGA2Optimizer
            save_path: Optional path to save figure

        Example:
            >>> history = optimizer.get_history()
            >>> ParetoVisualizer.plot_convergence(history)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, cannot plot")
            return

        if not history:
            logger.warning("No history to plot")
            return

        generations = [h["generation"] for h in history]
        pareto_sizes = [h["pareto_size"] for h in history]

        plt.figure(figsize=(10, 6))
        plt.plot(generations, pareto_sizes, "b-o", linewidth=2, markersize=6)

        plt.xlabel("Generation", fontsize=14, fontweight="bold")
        plt.ylabel("Pareto Front Size", fontsize=14, fontweight="bold")
        plt.title("NSGA-II Convergence", fontsize=16, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved convergence plot to {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_objective_distribution(
        pareto_front: List[MultiObjectiveIndividual],
        objective_name: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot distribution of a single objective across Pareto front.

        Args:
            pareto_front: List of Pareto-optimal individuals
            objective_name: Objective to visualize
            title: Optional plot title
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, cannot plot")
            return

        values = [ind.objectives[objective_name] for ind in pareto_front]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(values, bins=20, alpha=0.7, color="blue", edgecolor="black")
        ax1.set_xlabel(objective_name.replace("_", " ").title(), fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_title("Distribution", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")

        # Box plot
        ax2.boxplot(
            values, vert=True, patch_artist=True, boxprops={"facecolor": "lightblue", "alpha": 0.7}
        )
        ax2.set_ylabel(objective_name.replace("_", " ").title(), fontsize=12)
        ax2.set_title("Box Plot", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        if title:
            fig.suptitle(title, fontsize=16, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved distribution plot to {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_tradeoff_matrix(
        pareto_front: List[MultiObjectiveIndividual],
        objective_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot pairwise trade-off matrix for all objectives.

        Creates a grid of scatter plots showing relationships between
        all pairs of objectives.

        Args:
            pareto_front: List of Pareto-optimal individuals
            objective_names: List of objectives (None = all)
            save_path: Optional path to save figure

        Example:
            >>> ParetoVisualizer.plot_tradeoff_matrix(pareto_front)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, cannot plot")
            return

        if objective_names is None:
            objective_names = list(pareto_front[0].objectives.keys())

        n = len(objective_names)
        fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))

        for i, obj1 in enumerate(objective_names):
            for j, obj2 in enumerate(objective_names):
                ax = axes[i, j] if n > 1 else axes

                if i == j:
                    # Diagonal: histogram
                    values = [ind.objectives[obj1] for ind in pareto_front]
                    ax.hist(values, bins=15, alpha=0.7, color="blue", edgecolor="black")
                    ax.set_ylabel("Count", fontsize=10)
                else:
                    # Off-diagonal: scatter plot
                    x = [ind.objectives[obj2] for ind in pareto_front]
                    y = [ind.objectives[obj1] for ind in pareto_front]
                    ax.scatter(x, y, s=50, alpha=0.6, c="blue", edgecolors="black")

                # Labels
                if i == n - 1:
                    ax.set_xlabel(obj2.replace("_", " ").title(), fontsize=10)
                if j == 0:
                    ax.set_ylabel(obj1.replace("_", " ").title(), fontsize=10)

                ax.grid(True, alpha=0.3)

        plt.suptitle("Objective Trade-off Matrix", fontsize=18, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved trade-off matrix to {save_path}")
        else:
            plt.show()

        plt.close()


# Convenience functions
def quick_visualize_2d(pareto_front: List[MultiObjectiveIndividual], obj1: str, obj2: str) -> None:
    """Quick 2D visualization."""
    ParetoVisualizer.plot_2d(pareto_front, obj1, obj2)


def quick_visualize_3d(
    pareto_front: List[MultiObjectiveIndividual], obj1: str, obj2: str, obj3: str
) -> None:
    """Quick 3D visualization."""
    ParetoVisualizer.plot_3d(pareto_front, obj1, obj2, obj3)
