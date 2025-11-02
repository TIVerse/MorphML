"""Population visualization tools."""

from typing import Optional

from morphml.core.search import Population
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class PopulationVisualizer:
    """
    Visualize population statistics and distributions.

    Example:
        >>> viz = PopulationVisualizer()
        >>> viz.plot_fitness_distribution(population)
        >>> viz.plot_age_distribution(population)
    """

    def __init__(self):
        """Initialize visualizer."""
        pass

    def plot_fitness_distribution(
        self, population: Population, output_path: Optional[str] = None
    ) -> None:
        """
        Plot fitness distribution histogram.

        Args:
            population: Population to visualize
            output_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return

        # Get fitness values
        fitnesses = [
            ind.fitness
            for ind in population.individuals
            if ind.is_evaluated() and ind.fitness is not None
        ]

        if not fitnesses:
            logger.warning("No evaluated individuals to plot")
            return

        plt.figure(figsize=(10, 6))
        plt.hist(fitnesses, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
        plt.axvline(
            sum(fitnesses) / len(fitnesses),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {sum(fitnesses)/len(fitnesses):.3f}",
        )
        plt.xlabel("Fitness", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title(f"Fitness Distribution (N={len(fitnesses)})", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Fitness distribution plot saved to {output_path}")
        else:
            plt.show()

    def plot_age_distribution(
        self, population: Population, output_path: Optional[str] = None
    ) -> None:
        """
        Plot age distribution.

        Args:
            population: Population to visualize
            output_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return

        ages = [ind.age for ind in population.individuals]

        if not ages:
            return

        plt.figure(figsize=(10, 6))
        plt.hist(ages, bins=range(max(ages) + 2), color="lightgreen", edgecolor="black", alpha=0.7)
        plt.xlabel("Age (Generations)", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title(
            f"Age Distribution (Mean: {sum(ages)/len(ages):.1f})", fontsize=14, fontweight="bold"
        )
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Age distribution plot saved to {output_path}")
        else:
            plt.show()

    def plot_complexity_distribution(
        self, population: Population, output_path: Optional[str] = None
    ) -> None:
        """
        Plot architecture complexity distribution.

        Args:
            population: Population to visualize
            output_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return

        node_counts = [len(ind.graph.nodes) for ind in population.individuals]
        param_counts = [ind.graph.estimate_parameters() for ind in population.individuals]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Node count distribution
        ax1.hist(node_counts, bins=20, color="coral", edgecolor="black", alpha=0.7)
        ax1.set_xlabel("Number of Nodes", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_title("Node Count Distribution", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")

        # Parameter count distribution
        ax2.hist(param_counts, bins=20, color="lightblue", edgecolor="black", alpha=0.7)
        ax2.set_xlabel("Parameters", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_title("Parameter Count Distribution", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Complexity distribution plot saved to {output_path}")
        else:
            plt.show()

    def plot_fitness_vs_complexity(
        self, population: Population, output_path: Optional[str] = None
    ) -> None:
        """
        Plot fitness vs complexity scatter.

        Args:
            population: Population to visualize
            output_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return

        # Get data
        data = []
        for ind in population.individuals:
            if ind.is_evaluated() and ind.fitness is not None:
                data.append(
                    {
                        "fitness": ind.fitness,
                        "nodes": len(ind.graph.nodes),
                        "params": ind.graph.estimate_parameters(),
                    }
                )

        if not data:
            return

        fitnesses = [d["fitness"] for d in data]
        nodes = [d["nodes"] for d in data]
        params = [d["params"] for d in data]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Fitness vs nodes
        ax1.scatter(nodes, fitnesses, c="blue", s=100, alpha=0.6, edgecolors="black")
        ax1.set_xlabel("Number of Nodes", fontsize=12)
        ax1.set_ylabel("Fitness", fontsize=12)
        ax1.set_title("Fitness vs Node Count", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Fitness vs parameters
        ax2.scatter(params, fitnesses, c="red", s=100, alpha=0.6, edgecolors="black")
        ax2.set_xlabel("Parameters", fontsize=12)
        ax2.set_ylabel("Fitness", fontsize=12)
        ax2.set_title("Fitness vs Parameters", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Fitness vs complexity plot saved to {output_path}")
        else:
            plt.show()

    def plot_population_summary(
        self, population: Population, output_path: Optional[str] = None
    ) -> None:
        """
        Plot comprehensive population summary.

        Args:
            population: Population to visualize
            output_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Fitness distribution
        ax1 = fig.add_subplot(gs[0, 0])
        fitnesses = [
            ind.fitness for ind in population.individuals if ind.is_evaluated() and ind.fitness
        ]
        if fitnesses:
            ax1.hist(fitnesses, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
            ax1.set_xlabel("Fitness")
            ax1.set_ylabel("Count")
            ax1.set_title("Fitness Distribution")
            ax1.grid(True, alpha=0.3, axis="y")

        # Age distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ages = [ind.age for ind in population.individuals]
        ax2.hist(ages, bins=range(max(ages) + 2), color="lightgreen", edgecolor="black", alpha=0.7)
        ax2.set_xlabel("Age (Generations)")
        ax2.set_ylabel("Count")
        ax2.set_title("Age Distribution")
        ax2.grid(True, alpha=0.3, axis="y")

        # Node count distribution
        ax3 = fig.add_subplot(gs[0, 2])
        node_counts = [len(ind.graph.nodes) for ind in population.individuals]
        ax3.hist(node_counts, bins=20, color="coral", edgecolor="black", alpha=0.7)
        ax3.set_xlabel("Number of Nodes")
        ax3.set_ylabel("Count")
        ax3.set_title("Complexity Distribution")
        ax3.grid(True, alpha=0.3, axis="y")

        # Fitness vs nodes scatter
        ax4 = fig.add_subplot(gs[1, :2])
        data = [
            (ind.fitness, len(ind.graph.nodes))
            for ind in population.individuals
            if ind.is_evaluated() and ind.fitness
        ]
        if data:
            fit_vals, node_vals = zip(*data)
            ax4.scatter(node_vals, fit_vals, c="purple", s=100, alpha=0.6, edgecolors="black")
            ax4.set_xlabel("Number of Nodes")
            ax4.set_ylabel("Fitness")
            ax4.set_title("Fitness vs Complexity")
            ax4.grid(True, alpha=0.3)

        # Statistics text
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis("off")
        stats = population.get_statistics()
        stats_text = [
            f"Population Size: {stats.get('size', 0)}",
            f"Generation: {stats.get('generation', 0)}",
            f"Evaluated: {stats.get('evaluated', 0)}",
            "",
            f"Best Fitness: {stats.get('best_fitness', 0):.4f}",
            f"Mean Fitness: {stats.get('mean_fitness', 0):.4f}",
            f"Worst Fitness: {stats.get('worst_fitness', 0):.4f}",
            "",
            f"Diversity: {population.get_diversity():.3f}",
        ]
        ax5.text(
            0.1,
            0.9,
            "\n".join(stats_text),
            verticalalignment="top",
            fontsize=11,
            family="monospace",
        )

        fig.suptitle(
            f'Population Summary (Gen {stats.get("generation", 0)})', fontsize=16, fontweight="bold"
        )

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Population summary saved to {output_path}")
        else:
            plt.show()
