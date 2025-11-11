"""Performance analytics for MorphML experiments.

Statistical analysis and reporting of NAS results.

Example:
    >>> from morphml.visualization.analytics import PerformanceAnalytics
    >>> analytics = PerformanceAnalytics()
    >>> report = analytics.analyze_experiment(history)
    >>> print(report['summary'])
"""

from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from morphml.logging_config import get_logger

logger = get_logger(__name__)


class PerformanceAnalytics:
    """
    Statistical analysis of NAS experiment results.

    Provides comprehensive analytics including:
    - Descriptive statistics
    - Convergence analysis
    - Diversity metrics
    - Statistical significance tests

    Example:
        >>> analytics = PerformanceAnalytics()
        >>> report = analytics.analyze_experiment(optimizer.history)
        >>> print(f"Best: {report['best']:.4f}")
        >>> print(f"Mean: {report['mean']:.4f}")
    """

    def __init__(self):
        """Initialize analytics engine."""
        logger.info("Initialized PerformanceAnalytics")

    def analyze_experiment(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive experiment analysis.

        Args:
            history: Optimization history with keys:
                - best_fitness: List of best fitness per generation
                - mean_fitness: List of mean fitness per generation
                - population_fitness: List of lists of fitness values
                - diversity: List of diversity scores
                - architectures: List of architecture dicts

        Returns:
            Analysis report dictionary

        Example:
            >>> report = analytics.analyze_experiment(history)
            >>> print(report['summary'])
        """
        best_fitness = history.get("best_fitness", [])
        mean_fitness = history.get("mean_fitness", [])
        all_fitness = []

        for pop in history.get("population_fitness", []):
            all_fitness.extend(pop)

        if not all_fitness:
            all_fitness = best_fitness

        report = {
            "summary": self._compute_summary_statistics(all_fitness),
            "convergence": self._analyze_convergence(best_fitness),
            "diversity": self._analyze_diversity(history.get("diversity", [])),
            "efficiency": self._analyze_efficiency(best_fitness, mean_fitness),
            "architectures": self._analyze_architectures(history.get("architectures", [])),
        }

        return report

    def _compute_summary_statistics(self, fitness_values: List[float]) -> Dict[str, float]:
        """
        Compute summary statistics.

        Args:
            fitness_values: List of fitness values

        Returns:
            Dictionary of statistics
        """
        if not fitness_values:
            return {}

        fitness_array = np.array(fitness_values)

        return {
            "best": float(np.max(fitness_array)),
            "worst": float(np.min(fitness_array)),
            "mean": float(np.mean(fitness_array)),
            "median": float(np.median(fitness_array)),
            "std": float(np.std(fitness_array)),
            "variance": float(np.var(fitness_array)),
            "q25": float(np.percentile(fitness_array, 25)),
            "q75": float(np.percentile(fitness_array, 75)),
            "iqr": float(np.percentile(fitness_array, 75) - np.percentile(fitness_array, 25)),
            "count": len(fitness_values),
        }

    def _analyze_convergence(self, best_fitness: List[float]) -> Dict[str, Any]:
        """
        Analyze convergence behavior.

        Args:
            best_fitness: List of best fitness per generation

        Returns:
            Convergence analysis
        """
        if len(best_fitness) < 2:
            return {}

        fitness_array = np.array(best_fitness)
        generations = np.arange(len(best_fitness))

        # Compute improvement rate
        improvements = np.diff(fitness_array)
        improvement_rate = (
            float(np.mean(improvements[improvements > 0]))
            if len(improvements[improvements > 0]) > 0
            else 0.0
        )

        # Detect convergence point (when improvement < threshold)
        threshold = 0.001
        converged_at = None
        for i in range(len(improvements)):
            if all(abs(improvements[i : i + 5]) < threshold):
                converged_at = i
                break

        # Fit linear trend
        if len(best_fitness) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                generations, fitness_array
            )
        else:
            slope, r_value = 0.0, 0.0

        return {
            "converged": converged_at is not None,
            "converged_at_generation": converged_at,
            "improvement_rate": improvement_rate,
            "total_improvement": float(fitness_array[-1] - fitness_array[0]),
            "trend_slope": float(slope),
            "trend_r_squared": float(r_value**2),
            "final_fitness": float(fitness_array[-1]),
            "generations": len(best_fitness),
        }

    def _analyze_diversity(self, diversity_scores: List[float]) -> Dict[str, Any]:
        """
        Analyze population diversity.

        Args:
            diversity_scores: List of diversity scores per generation

        Returns:
            Diversity analysis
        """
        if not diversity_scores:
            return {}

        diversity_array = np.array(diversity_scores)

        # Detect diversity collapse (when diversity drops below threshold)
        threshold = 0.1
        collapsed = diversity_array[-1] < threshold if len(diversity_array) > 0 else False

        return {
            "initial_diversity": float(diversity_array[0]) if len(diversity_array) > 0 else 0.0,
            "final_diversity": float(diversity_array[-1]) if len(diversity_array) > 0 else 0.0,
            "mean_diversity": float(np.mean(diversity_array)),
            "min_diversity": float(np.min(diversity_array)),
            "max_diversity": float(np.max(diversity_array)),
            "diversity_collapsed": collapsed,
            "diversity_trend": "decreasing"
            if len(diversity_array) > 1 and diversity_array[-1] < diversity_array[0]
            else "stable",
        }

    def _analyze_efficiency(
        self, best_fitness: List[float], mean_fitness: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze search efficiency.

        Args:
            best_fitness: Best fitness per generation
            mean_fitness: Mean fitness per generation

        Returns:
            Efficiency metrics
        """
        if not best_fitness or not mean_fitness:
            return {}

        best_array = np.array(best_fitness)
        mean_array = np.array(mean_fitness)

        # Compute selection pressure (best vs mean)
        selection_pressure = best_array - mean_array

        # Compute exploitation vs exploration balance
        # High selection pressure = more exploitation
        avg_pressure = float(np.mean(selection_pressure))

        return {
            "average_selection_pressure": avg_pressure,
            "exploitation_score": min(1.0, avg_pressure / 0.5),  # Normalized
            "exploration_score": max(0.0, 1.0 - avg_pressure / 0.5),
            "efficiency_ratio": float(best_array[-1] / (len(best_fitness) * 0.01))
            if len(best_fitness) > 0
            else 0.0,
        }

    def _analyze_architectures(self, architectures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze architecture characteristics.

        Args:
            architectures: List of architecture dicts

        Returns:
            Architecture analysis
        """
        if not architectures:
            return {}

        # Extract metrics
        parameters = [a.get("parameters", 0) for a in architectures]
        depths = [a.get("depth", 0) for a in architectures]
        widths = [a.get("width", 0) for a in architectures]
        fitness = [a.get("fitness", 0) for a in architectures]

        # Compute correlations
        param_fitness_corr = (
            float(np.corrcoef(parameters, fitness)[0, 1]) if len(parameters) > 1 else 0.0
        )
        depth_fitness_corr = float(np.corrcoef(depths, fitness)[0, 1]) if len(depths) > 1 else 0.0

        return {
            "count": len(architectures),
            "parameters": {
                "mean": float(np.mean(parameters)),
                "min": float(np.min(parameters)),
                "max": float(np.max(parameters)),
                "std": float(np.std(parameters)),
            },
            "depth": {
                "mean": float(np.mean(depths)),
                "min": float(np.min(depths)),
                "max": float(np.max(depths)),
            },
            "width": {
                "mean": float(np.mean(widths)),
                "min": float(np.min(widths)),
                "max": float(np.max(widths)),
            },
            "correlations": {
                "parameters_fitness": param_fitness_corr,
                "depth_fitness": depth_fitness_corr,
            },
        }

    def compare_experiments(self, experiments: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple experiments statistically.

        Args:
            experiments: Dict mapping experiment names to history dicts

        Returns:
            Comparison report

        Example:
            >>> comparison = analytics.compare_experiments({
            ...     'GA': ga_history,
            ...     'Random': random_history
            ... })
        """
        results = {}

        for name, history in experiments.items():
            best_fitness = history.get("best_fitness", [])
            if best_fitness:
                results[name] = {
                    "final_fitness": best_fitness[-1],
                    "mean_fitness": np.mean(best_fitness),
                    "convergence_speed": self._compute_convergence_speed(best_fitness),
                }

        # Statistical tests
        if len(results) >= 2:
            names = list(results.keys())
            fitness_lists = [experiments[name].get("best_fitness", []) for name in names]

            # Perform t-test between first two
            if len(fitness_lists[0]) > 1 and len(fitness_lists[1]) > 1:
                t_stat, p_value = stats.ttest_ind(fitness_lists[0], fitness_lists[1])
                results["statistical_test"] = {
                    "test": "t-test",
                    "comparison": f"{names[0]} vs {names[1]}",
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }

        return results

    def _compute_convergence_speed(self, best_fitness: List[float]) -> float:
        """
        Compute convergence speed metric.

        Args:
            best_fitness: Best fitness per generation

        Returns:
            Convergence speed score
        """
        if len(best_fitness) < 2:
            return 0.0

        # Compute area under curve (higher = faster convergence)
        fitness_array = np.array(best_fitness)
        normalized = (fitness_array - fitness_array[0]) / (
            fitness_array[-1] - fitness_array[0] + 1e-10
        )
        auc = float(np.trapz(normalized))

        return auc / len(best_fitness)

    def generate_report(self, history: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate human-readable analysis report.

        Args:
            history: Optimization history
            output_path: Optional file path to save report

        Returns:
            Report string

        Example:
            >>> report = analytics.generate_report(history, "report.txt")
            >>> print(report)
        """
        analysis = self.analyze_experiment(history)

        report_lines = [
            "=" * 60,
            "MORPHML EXPERIMENT ANALYSIS REPORT",
            "=" * 60,
            "",
            "SUMMARY STATISTICS",
            "-" * 60,
        ]

        summary = analysis.get("summary", {})
        if summary:
            report_lines.extend(
                [
                    f"Best Fitness:    {summary.get('best', 0):.6f}",
                    f"Worst Fitness:   {summary.get('worst', 0):.6f}",
                    f"Mean Fitness:    {summary.get('mean', 0):.6f}",
                    f"Median Fitness:  {summary.get('median', 0):.6f}",
                    f"Std Deviation:   {summary.get('std', 0):.6f}",
                    f"IQR:             {summary.get('iqr', 0):.6f}",
                    f"Total Evaluated: {summary.get('count', 0)}",
                    "",
                ]
            )

        convergence = analysis.get("convergence", {})
        if convergence:
            report_lines.extend(
                [
                    "CONVERGENCE ANALYSIS",
                    "-" * 60,
                    f"Converged:           {convergence.get('converged', False)}",
                    f"Converged at Gen:    {convergence.get('converged_at_generation', 'N/A')}",
                    f"Total Improvement:   {convergence.get('total_improvement', 0):.6f}",
                    f"Improvement Rate:    {convergence.get('improvement_rate', 0):.6f}",
                    f"Trend R²:            {convergence.get('trend_r_squared', 0):.4f}",
                    "",
                ]
            )

        diversity = analysis.get("diversity", {})
        if diversity:
            report_lines.extend(
                [
                    "DIVERSITY ANALYSIS",
                    "-" * 60,
                    f"Initial Diversity:   {diversity.get('initial_diversity', 0):.4f}",
                    f"Final Diversity:     {diversity.get('final_diversity', 0):.4f}",
                    f"Mean Diversity:      {diversity.get('mean_diversity', 0):.4f}",
                    f"Diversity Trend:     {diversity.get('diversity_trend', 'N/A')}",
                    f"Collapsed:           {diversity.get('diversity_collapsed', False)}",
                    "",
                ]
            )

        report_lines.append("=" * 60)

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Saved report to {output_path}")

        return report
