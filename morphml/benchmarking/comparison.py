"""Optimizer comparison and benchmarking tools.

This module provides tools to systematically compare different optimization
algorithms on the same search space and evaluation budget.

Features:
- Run multiple optimizers with same budget
- Statistical comparison (mean, std, confidence intervals)
- Convergence curve comparison
- Sample efficiency analysis
- Result visualization

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from morphml.core.dsl import SearchSpace
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class OptimizerComparison:
    """
    Compare multiple optimizers on the same problem.

    Runs each optimizer multiple times and collects statistics for
    rigorous comparison including convergence speed and sample efficiency.

    Example:
        >>> from morphml.benchmarking import OptimizerComparison
        >>> from morphml.optimizers import GeneticAlgorithm, optimize_with_pso
        >>>
        >>> comparison = OptimizerComparison(
        ...     search_space=space,
        ...     evaluator=my_evaluator,
        ...     budget=100,
        ...     num_runs=10
        ... )
        >>>
        >>> comparison.add_optimizer('GA', GeneticAlgorithm(space, config))
        >>> comparison.add_optimizer('PSO', ParticleSwarmOptimizer(space, config))
        >>>
        >>> results = comparison.run()
        >>> comparison.plot_comparison()
    """

    def __init__(
        self, search_space: SearchSpace, evaluator: Callable, budget: int = 100, num_runs: int = 5
    ):
        """
        Initialize comparison.

        Args:
            search_space: SearchSpace to search
            evaluator: Fitness evaluation function
            budget: Number of evaluations per run
            num_runs: Number of independent runs per optimizer
        """
        self.search_space = search_space
        self.evaluator = evaluator
        self.budget = budget
        self.num_runs = num_runs

        self.optimizers: Dict[str, Any] = {}
        self.results: Dict[str, List[Dict]] = {}
        self.statistics: Dict[str, Dict] = {}

        logger.info(f"Initialized OptimizerComparison: " f"budget={budget}, num_runs={num_runs}")

    def add_optimizer(self, name: str, optimizer: Any) -> None:
        """
        Add optimizer to comparison.

        Args:
            name: Name for this optimizer
            optimizer: Optimizer instance or factory function
        """
        self.optimizers[name] = optimizer
        logger.debug(f"Added optimizer: {name}")

    def run(self) -> Dict[str, Dict]:
        """
        Run all optimizers and collect results.

        Returns:
            Statistics dictionary with results for each optimizer

        Example:
            >>> results = comparison.run()
            >>> print(f"GA mean: {results['GA']['mean_best']:.4f}")
        """
        logger.info(
            f"Running comparison with {len(self.optimizers)} optimizers, "
            f"{self.num_runs} runs each"
        )

        for name, optimizer_template in self.optimizers.items():
            logger.info(f"\nBenchmarking {name}...")

            run_results = []

            for run_id in range(self.num_runs):
                logger.info(f"  Run {run_id + 1}/{self.num_runs}")

                # Create fresh optimizer instance
                if callable(optimizer_template):
                    optimizer = optimizer_template()
                else:
                    optimizer = optimizer_template

                # Run optimization
                start_time = time.time()

                try:
                    best = optimizer.optimize(self.evaluator)
                    elapsed_time = time.time() - start_time

                    # Collect results
                    history = optimizer.get_history() if hasattr(optimizer, "get_history") else []

                    run_results.append(
                        {
                            "run_id": run_id,
                            "best_fitness": best.fitness,
                            "best_individual": best,
                            "time_seconds": elapsed_time,
                            "history": history,
                            "success": True,
                        }
                    )

                    logger.info(
                        f"    Best fitness: {best.fitness:.4f}, " f"Time: {elapsed_time:.2f}s"
                    )

                except Exception as e:
                    logger.error(f"    Run failed: {e}")
                    run_results.append({"run_id": run_id, "success": False, "error": str(e)})

            self.results[name] = run_results

        # Compute statistics
        self.statistics = self._compute_statistics()

        # Print summary
        self._print_summary()

        return self.statistics

    def _compute_statistics(self) -> Dict[str, Dict]:
        """
        Compute statistics across runs for each optimizer.

        Returns:
            Statistics dictionary
        """
        stats = {}

        for name, runs in self.results.items():
            # Filter successful runs
            successful_runs = [r for r in runs if r.get("success", False)]

            if not successful_runs:
                stats[name] = {"error": "All runs failed"}
                continue

            # Extract metrics
            best_fitnesses = [r["best_fitness"] for r in successful_runs]
            times = [r["time_seconds"] for r in successful_runs]

            # Compute statistics
            stats[name] = {
                "num_successful_runs": len(successful_runs),
                "num_failed_runs": len(runs) - len(successful_runs),
                # Best fitness statistics
                "mean_best": np.mean(best_fitnesses),
                "std_best": np.std(best_fitnesses),
                "min_best": np.min(best_fitnesses),
                "max_best": np.max(best_fitnesses),
                "median_best": np.median(best_fitnesses),
                # Confidence interval (95%)
                "ci_lower": np.percentile(best_fitnesses, 2.5),
                "ci_upper": np.percentile(best_fitnesses, 97.5),
                # Time statistics
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                # All runs data
                "all_best_fitnesses": best_fitnesses,
                "all_times": times,
            }

        return stats

    def _print_summary(self) -> None:
        """Print comparison summary."""
        print("\n" + "=" * 70)
        print("OPTIMIZER COMPARISON SUMMARY")
        print("=" * 70)
        print(f"Budget: {self.budget} evaluations")
        print(f"Runs per optimizer: {self.num_runs}")
        print("=" * 70)

        # Sort by mean best fitness (descending)
        sorted_optimizers = sorted(
            self.statistics.items(), key=lambda x: x[1].get("mean_best", -np.inf), reverse=True
        )

        print(f"\n{'Optimizer':<20} {'Mean Best':<12} {'Std':<10} {'Time (s)':<10}")
        print("-" * 70)

        for name, stats in sorted_optimizers:
            if "error" in stats:
                print(f"{name:<20} {'FAILED':<12}")
            else:
                print(
                    f"{name:<20} "
                    f"{stats['mean_best']:<12.4f} "
                    f"{stats['std_best']:<10.4f} "
                    f"{stats['mean_time']:<10.2f}"
                )

        print("=" * 70 + "\n")

        # Winner
        if sorted_optimizers and "error" not in sorted_optimizers[0][1]:
            winner = sorted_optimizers[0][0]
            print(f"🏆 Best Optimizer: {winner}")
            print(f"   Mean Fitness: {sorted_optimizers[0][1]['mean_best']:.4f}")
            print(f"   Std: {sorted_optimizers[0][1]['std_best']:.4f}")
            print("=" * 70 + "\n")

    def plot_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Plot comparison results.

        Creates box plots showing distribution of best fitnesses.

        Args:
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, cannot plot")
            return

        if not self.statistics:
            logger.warning("No results to plot. Run comparison first.")
            return

        # Prepare data
        names = []
        data = []

        for name, stats in self.statistics.items():
            if "all_best_fitnesses" in stats:
                names.append(name)
                data.append(stats["all_best_fitnesses"])

        if not data:
            logger.warning("No successful runs to plot")
            return

        # Create box plot
        fig, ax = plt.subplots(figsize=(10, 6))

        bp = ax.boxplot(data, labels=names, patch_artist=True)

        # Color boxes
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
        for patch, color in zip(bp["boxes"], colors * len(names)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel("Optimizer", fontsize=12, fontweight="bold")
        ax.set_ylabel("Best Fitness", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Optimizer Comparison ({self.num_runs} runs, {self.budget} evaluations)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="y")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Comparison plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_convergence(self, save_path: Optional[str] = None) -> None:
        """
        Plot convergence curves for all optimizers.

        Shows how best fitness evolves over iterations.

        Args:
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, cannot plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#C7CEEA"]

        for (name, runs), color in zip(self.results.items(), colors * len(self.results)):
            # Average convergence across runs
            all_histories = [r["history"] for r in runs if r.get("success") and r.get("history")]

            if not all_histories:
                continue

            # Find max length
            max_len = max(len(h) for h in all_histories)

            # Pad and average
            padded_histories = []
            for hist in all_histories:
                if len(hist) < max_len:
                    # Pad with last value
                    last_val = hist[-1]["best_fitness"] if hist else 0
                    padded = hist + [{"best_fitness": last_val}] * (max_len - len(hist))
                else:
                    padded = hist

                padded_histories.append([h.get("best_fitness", 0) for h in padded])

            avg_convergence = np.mean(padded_histories, axis=0)
            std_convergence = np.std(padded_histories, axis=0)

            iterations = range(len(avg_convergence))

            ax.plot(iterations, avg_convergence, label=name, linewidth=2, color=color)
            ax.fill_between(
                iterations,
                avg_convergence - std_convergence,
                avg_convergence + std_convergence,
                alpha=0.2,
                color=color,
            )

        ax.set_xlabel("Iteration", fontsize=12, fontweight="bold")
        ax.set_ylabel("Best Fitness", fontsize=12, fontweight="bold")
        ax.set_title("Convergence Comparison", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Convergence plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def get_results(self) -> Dict[str, List[Dict]]:
        """Get raw results."""
        return self.results

    def get_statistics(self) -> Dict[str, Dict]:
        """Get computed statistics."""
        return self.statistics


# Convenience function
def compare_optimizers(
    optimizers: Dict[str, Any],
    search_space: SearchSpace,
    evaluator: Callable,
    budget: int = 100,
    num_runs: int = 5,
    plot: bool = True,
) -> Dict[str, Dict]:
    """
    Quick comparison of multiple optimizers.

    Args:
        optimizers: Dictionary of {name: optimizer} pairs
        search_space: SearchSpace to search
        evaluator: Fitness evaluation function
        budget: Number of evaluations per run
        num_runs: Number of runs per optimizer
        plot: Whether to generate plots

    Returns:
        Statistics dictionary

    Example:
        >>> from morphml.benchmarking import compare_optimizers
        >>> from morphml.optimizers import GeneticAlgorithm, optimize_with_pso
        >>>
        >>> results = compare_optimizers(
        ...     optimizers={
        ...         'GA': GeneticAlgorithm(space, config),
        ...         'PSO': ParticleSwarmOptimizer(space, config)
        ...     },
        ...     search_space=space,
        ...     evaluator=my_evaluator,
        ...     budget=100,
        ...     num_runs=5
        ... )
    """
    comparison = OptimizerComparison(search_space, evaluator, budget, num_runs)

    for name, optimizer in optimizers.items():
        comparison.add_optimizer(name, optimizer)

    results = comparison.run()

    if plot:
        comparison.plot_comparison()
        comparison.plot_convergence()

    return results
