"""Optimizer comparison utilities."""

import statistics
from typing import Any, Dict, List, Optional

from morphml.logging_config import get_logger

logger = get_logger(__name__)


class OptimizerComparator:
    """
    Compare multiple optimizers on benchmark problems.

    Provides statistical analysis and visualization support.

    Example:
        >>> comparator = OptimizerComparator()
        >>> comparator.add_result("GA", "Problem1", [0.9, 0.92, 0.91])
        >>> comparator.add_result("RS", "Problem1", [0.85, 0.87, 0.86])
        >>> comparator.print_comparison()
    """

    def __init__(self):
        """Initialize comparator."""
        self.results: Dict[tuple, List[float]] = {}
        self.metadata: Dict[tuple, Dict[str, Any]] = {}

    def add_result(
        self, optimizer_name: str, problem_name: str, fitnesses: List[float], **metadata
    ) -> None:
        """
        Add results for an optimizer on a problem.

        Args:
            optimizer_name: Name of optimizer
            problem_name: Name of problem
            fitnesses: List of fitness values from multiple runs
            **metadata: Additional metadata (time, evaluations, etc.)
        """
        key = (optimizer_name, problem_name)
        self.results[key] = fitnesses
        self.metadata[key] = metadata

        logger.debug(f"Added results: {optimizer_name} on {problem_name} " f"(n={len(fitnesses)})")

    def get_statistics(self, optimizer_name: str, problem_name: str) -> Dict[str, float]:
        """
        Get statistics for an optimizer-problem pair.

        Args:
            optimizer_name: Optimizer name
            problem_name: Problem name

        Returns:
            Dictionary of statistics
        """
        key = (optimizer_name, problem_name)

        if key not in self.results:
            return {}

        fitnesses = self.results[key]

        if not fitnesses:
            return {}

        stats = {
            "mean": statistics.mean(fitnesses),
            "median": statistics.median(fitnesses),
            "std": statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0,
            "min": min(fitnesses),
            "max": max(fitnesses),
            "count": len(fitnesses),
        }

        return stats

    def rank_optimizers(self, problem_name: str) -> List[tuple]:
        """
        Rank optimizers for a specific problem.

        Args:
            problem_name: Problem name

        Returns:
            List of (optimizer_name, mean_fitness) tuples, sorted
        """
        rankings = []

        for (opt_name, prob_name), fitnesses in self.results.items():
            if prob_name == problem_name and fitnesses:
                mean_fitness = statistics.mean(fitnesses)
                rankings.append((opt_name, mean_fitness))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def compare_pair(self, optimizer1: str, optimizer2: str, problem_name: str) -> Dict[str, Any]:
        """
        Statistical comparison of two optimizers.

        Args:
            optimizer1: First optimizer name
            optimizer2: Second optimizer name
            problem_name: Problem name

        Returns:
            Comparison results
        """
        key1 = (optimizer1, problem_name)
        key2 = (optimizer2, problem_name)

        if key1 not in self.results or key2 not in self.results:
            return {}

        fitnesses1 = self.results[key1]
        fitnesses2 = self.results[key2]

        stats1 = self.get_statistics(optimizer1, problem_name)
        stats2 = self.get_statistics(optimizer2, problem_name)

        # Perform t-test if enough samples
        p_value = None
        if len(fitnesses1) > 1 and len(fitnesses2) > 1:
            try:
                from scipy import stats as scipy_stats

                t_stat, p_value = scipy_stats.ttest_ind(fitnesses1, fitnesses2)
            except ImportError:
                logger.warning("scipy not available for statistical tests")

        comparison = {
            "optimizer1": optimizer1,
            "optimizer2": optimizer2,
            "problem": problem_name,
            "mean_diff": stats1["mean"] - stats2["mean"],
            "median_diff": stats1["median"] - stats2["median"],
            "winner": optimizer1 if stats1["mean"] > stats2["mean"] else optimizer2,
            "p_value": p_value,
            "significant": p_value < 0.05 if p_value else None,
        }

        return comparison

    def get_dominance_matrix(self) -> Dict[tuple, int]:
        """
        Get dominance matrix showing which optimizer beats which.

        Returns:
            Dictionary mapping (opt1, opt2) to number of problems opt1 beats opt2
        """
        optimizers = {opt for opt, _ in self.results.keys()}
        problems = {prob for _, prob in self.results.keys()}

        dominance = {}

        for opt1 in optimizers:
            for opt2 in optimizers:
                if opt1 == opt2:
                    continue

                wins = 0
                for problem in problems:
                    stats1 = self.get_statistics(opt1, problem)
                    stats2 = self.get_statistics(opt2, problem)

                    if stats1 and stats2:
                        if stats1["mean"] > stats2["mean"]:
                            wins += 1

                dominance[(opt1, opt2)] = wins

        return dominance

    def print_comparison(self) -> None:
        """Print detailed comparison of all optimizers."""
        problems = {prob for _, prob in self.results.keys()}
        optimizers = {opt for opt, _ in self.results.keys()}

        print("\n" + "=" * 100)
        print("OPTIMIZER COMPARISON")
        print("=" * 100)

        for problem in sorted(problems):
            print(f"\n{problem}")
            print("-" * 100)
            print(
                f"{'Optimizer':<20} {'Mean':<12} {'Median':<12} {'Std':<12} {'Min':<12} {'Max':<12}"
            )
            print("-" * 100)

            rankings = self.rank_optimizers(problem)

            for opt_name, _mean_fitness in rankings:
                stats = self.get_statistics(opt_name, problem)

                print(
                    f"{opt_name:<20} "
                    f"{stats['mean']:<12.4f} "
                    f"{stats['median']:<12.4f} "
                    f"{stats['std']:<12.4f} "
                    f"{stats['min']:<12.4f} "
                    f"{stats['max']:<12.4f}"
                )

        # Overall rankings
        print("\n" + "=" * 100)
        print("OVERALL RANKINGS (by mean fitness across all problems)")
        print("=" * 100)

        overall_scores = {}
        for opt in optimizers:
            scores = []
            for prob in problems:
                stats = self.get_statistics(opt, prob)
                if stats:
                    scores.append(stats["mean"])

            if scores:
                overall_scores[opt] = statistics.mean(scores)

        overall_rankings = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)

        for rank, (opt, score) in enumerate(overall_rankings, 1):
            print(f"{rank}. {opt:<20} Average: {score:.4f}")

        print("\n" + "=" * 100)

    def get_best_optimizer(self) -> Optional[str]:
        """Get overall best optimizer."""
        problems = {prob for _, prob in self.results.keys()}
        optimizers = {opt for opt, _ in self.results.keys()}

        overall_scores = {}
        for opt in optimizers:
            scores = []
            for prob in problems:
                stats = self.get_statistics(opt, prob)
                if stats:
                    scores.append(stats["mean"])

            if scores:
                overall_scores[opt] = statistics.mean(scores)

        if not overall_scores:
            return None

        return max(overall_scores.items(), key=lambda x: x[1])[0]

    def export_latex_table(self, filename: str) -> None:
        """Export comparison as LaTeX table."""
        problems = sorted({prob for _, prob in self.results.keys()})
        optimizers = sorted({opt for opt, _ in self.results.keys()})

        with open(filename, "w") as f:
            # Header
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Optimizer Comparison}\n")
            f.write("\\begin{tabular}{l" + "c" * len(optimizers) + "}\n")
            f.write("\\hline\n")

            # Column headers
            f.write("Problem & " + " & ".join(optimizers) + " \\\\\n")
            f.write("\\hline\n")

            # Data rows
            for problem in problems:
                row = [problem]
                for opt in optimizers:
                    stats = self.get_statistics(opt, problem)
                    if stats:
                        row.append(f"{stats['mean']:.3f}$\\pm${stats['std']:.3f}")
                    else:
                        row.append("--")

                f.write(" & ".join(row) + " \\\\\n")

            # Footer
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        logger.info(f"LaTeX table exported to {filename}")

    def plot_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Plot comparison of optimizers.

        Args:
            save_path: Path to save plot (displays if None)
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return

        problems = sorted({prob for _, prob in self.results.keys()})
        optimizers = sorted({opt for opt, _ in self.results.keys()})

        fig, axes = plt.subplots(1, len(problems), figsize=(5 * len(problems), 5))

        if len(problems) == 1:
            axes = [axes]

        for ax, problem in zip(axes, problems):
            means = []
            stds = []
            labels = []

            for opt in optimizers:
                stats = self.get_statistics(opt, problem)
                if stats:
                    means.append(stats["mean"])
                    stds.append(stats["std"])
                    labels.append(opt)

            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Fitness")
            ax.set_title(problem)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()


class ConvergenceAnalyzer:
    """Analyze convergence behavior of optimizers."""

    def __init__(self):
        """Initialize analyzer."""
        self.histories: Dict[str, List[List[float]]] = {}

    def add_history(self, optimizer_name: str, history: List[float]) -> None:
        """Add optimization history."""
        if optimizer_name not in self.histories:
            self.histories[optimizer_name] = []

        self.histories[optimizer_name].append(history)

    def get_mean_convergence(self, optimizer_name: str) -> List[float]:
        """Get mean convergence curve."""
        if optimizer_name not in self.histories:
            return []

        histories = self.histories[optimizer_name]

        if not histories:
            return []

        # Find minimum length
        min_len = min(len(h) for h in histories)

        # Calculate mean at each generation
        mean_curve = []
        for i in range(min_len):
            values = [h[i] for h in histories]
            mean_curve.append(statistics.mean(values))

        return mean_curve

    def calculate_auc(self, optimizer_name: str) -> float:
        """Calculate area under convergence curve."""
        curve = self.get_mean_convergence(optimizer_name)

        if not curve:
            return 0.0

        # Simple trapezoidal integration
        auc = sum(curve) / len(curve)
        return auc

    def plot_convergence(self, save_path: Optional[str] = None) -> None:
        """Plot convergence curves."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return

        plt.figure(figsize=(10, 6))

        for opt_name in sorted(self.histories.keys()):
            curve = self.get_mean_convergence(opt_name)
            if curve:
                plt.plot(curve, label=opt_name, linewidth=2)

        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Convergence Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Convergence plot saved to {save_path}")
        else:
            plt.show()
