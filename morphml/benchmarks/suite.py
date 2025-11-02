"""Benchmark suite for running comprehensive optimizer comparisons."""

import time
from typing import Any, Callable, Dict, List, Optional

from morphml.benchmarks.problems import BenchmarkProblem
from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class BenchmarkResult:
    """Results from a single benchmark run."""

    def __init__(
        self,
        problem_name: str,
        optimizer_name: str,
        best_fitness: float,
        mean_fitness: float,
        evaluations: int,
        time_seconds: float,
        convergence_generation: Optional[int] = None,
        final_diversity: Optional[float] = None,
    ):
        """Initialize benchmark result."""
        self.problem_name = problem_name
        self.optimizer_name = optimizer_name
        self.best_fitness = best_fitness
        self.mean_fitness = mean_fitness
        self.evaluations = evaluations
        self.time_seconds = time_seconds
        self.convergence_generation = convergence_generation
        self.final_diversity = final_diversity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "problem": self.problem_name,
            "optimizer": self.optimizer_name,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "evaluations": self.evaluations,
            "time_seconds": self.time_seconds,
            "convergence_generation": self.convergence_generation,
            "final_diversity": self.final_diversity,
        }

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult({self.optimizer_name} on {self.problem_name}: "
            f"best={self.best_fitness:.4f}, time={self.time_seconds:.2f}s)"
        )


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for NAS optimizers.

    Example:
        >>> suite = BenchmarkSuite()
        >>> suite.add_optimizer("GA", GeneticAlgorithm, {...})
        >>> suite.add_optimizer("RS", RandomSearch, {...})
        >>> results = suite.run(problems, num_runs=5)
        >>> suite.print_summary(results)
    """

    def __init__(self):
        """Initialize benchmark suite."""
        self.optimizers: Dict[str, tuple] = {}
        self.results: List[BenchmarkResult] = []

    def add_optimizer(self, name: str, optimizer_class: type, config: Dict[str, Any]) -> None:
        """
        Add an optimizer to benchmark.

        Args:
            name: Optimizer name
            optimizer_class: Optimizer class
            config: Configuration dict
        """
        self.optimizers[name] = (optimizer_class, config)
        logger.info(f"Added optimizer: {name}")

    def run_single(
        self,
        problem: BenchmarkProblem,
        optimizer_name: str,
        optimizer_class: type,
        config: Dict[str, Any],
    ) -> BenchmarkResult:
        """Run single benchmark."""
        logger.info(f"Running {optimizer_name} on {problem.name}...")

        start_time = time.time()

        # Create optimizer
        optimizer = optimizer_class(search_space=problem.search_space, **config)

        # Run optimization
        try:
            if hasattr(optimizer, "optimize"):
                best = optimizer.optimize(problem.evaluate)
            else:
                raise ValueError(f"Optimizer {optimizer_name} has no optimize method")

            end_time = time.time()
            elapsed = end_time - start_time

            # Collect metrics
            best_fitness = best.fitness if best else 0.0

            # Get population stats if available
            mean_fitness = 0.0
            final_diversity = None
            convergence_gen = None

            if hasattr(optimizer, "population"):
                pop = optimizer.population
                if pop.size() > 0:
                    stats = pop.get_statistics()
                    mean_fitness = stats.get("mean_fitness", 0.0)
                    final_diversity = pop.get_diversity()
                    convergence_gen = pop.generation

            # Count evaluations
            evaluations = config.get("num_generations", 0) * config.get("population_size", 1)
            if hasattr(optimizer, "num_samples"):
                evaluations = config.get("num_samples", 0)

            result = BenchmarkResult(
                problem_name=problem.name,
                optimizer_name=optimizer_name,
                best_fitness=best_fitness,
                mean_fitness=mean_fitness,
                evaluations=evaluations,
                time_seconds=elapsed,
                convergence_generation=convergence_gen,
                final_diversity=final_diversity,
            )

            logger.info(f"Completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            # Return failure result
            return BenchmarkResult(
                problem_name=problem.name,
                optimizer_name=optimizer_name,
                best_fitness=0.0,
                mean_fitness=0.0,
                evaluations=0,
                time_seconds=time.time() - start_time,
            )

    def run(self, problems: List[BenchmarkProblem], num_runs: int = 5) -> List[BenchmarkResult]:
        """
        Run full benchmark suite.

        Args:
            problems: List of benchmark problems
            num_runs: Number of runs per optimizer-problem pair

        Returns:
            List of benchmark results
        """
        all_results = []

        total_runs = len(self.optimizers) * len(problems) * num_runs
        current_run = 0

        logger.info(f"Starting benchmark: {total_runs} total runs")

        for problem in problems:
            for opt_name, (opt_class, config) in self.optimizers.items():
                for run in range(num_runs):
                    current_run += 1
                    logger.info(
                        f"Run {current_run}/{total_runs}: "
                        f"{opt_name} on {problem.name} (run {run + 1}/{num_runs})"
                    )

                    result = self.run_single(problem, opt_name, opt_class, config)
                    all_results.append(result)

        self.results = all_results
        logger.info("Benchmark complete!")

        return all_results

    def get_summary(self, results: Optional[List[BenchmarkResult]] = None) -> Dict[str, Any]:
        """
        Get summary statistics.

        Args:
            results: Results to summarize (uses self.results if None)

        Returns:
            Summary dictionary
        """
        if results is None:
            results = self.results

        if not results:
            return {}

        summary = {}

        # Group by optimizer and problem
        for result in results:
            key = (result.optimizer_name, result.problem_name)

            if key not in summary:
                summary[key] = {
                    "optimizer": result.optimizer_name,
                    "problem": result.problem_name,
                    "best_fitnesses": [],
                    "mean_fitnesses": [],
                    "times": [],
                    "evaluations": [],
                }

            summary[key]["best_fitnesses"].append(result.best_fitness)
            summary[key]["mean_fitnesses"].append(result.mean_fitness)
            summary[key]["times"].append(result.time_seconds)
            summary[key]["evaluations"].append(result.evaluations)

        # Compute statistics
        import statistics

        for key, data in summary.items():
            data["mean_best_fitness"] = statistics.mean(data["best_fitnesses"])
            data["std_best_fitness"] = (
                statistics.stdev(data["best_fitnesses"]) if len(data["best_fitnesses"]) > 1 else 0.0
            )
            data["median_best_fitness"] = statistics.median(data["best_fitnesses"])
            data["mean_time"] = statistics.mean(data["times"])
            data["mean_evaluations"] = statistics.mean(data["evaluations"])

        return summary

    def print_summary(self, results: Optional[List[BenchmarkResult]] = None) -> None:
        """Print summary of results."""
        summary = self.get_summary(results)

        if not summary:
            print("No results to summarize")
            return

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Group by problem
        problems = set(data["problem"] for data in summary.values())

        for problem in sorted(problems):
            print(f"\n{problem}")
            print("-" * 80)
            print(f"{'Optimizer':<20} {'Best (Mean±Std)':<25} {'Median':<10} {'Time (s)':<12}")
            print("-" * 80)

            problem_results = [
                (data["optimizer"], data)
                for key, data in summary.items()
                if data["problem"] == problem
            ]

            # Sort by mean best fitness (descending)
            problem_results.sort(key=lambda x: x[1]["mean_best_fitness"], reverse=True)

            for opt_name, data in problem_results:
                mean_best = data["mean_best_fitness"]
                std_best = data["std_best_fitness"]
                median_best = data["median_best_fitness"]
                mean_time = data["mean_time"]

                print(
                    f"{opt_name:<20} "
                    f"{mean_best:.4f}±{std_best:.4f}          "
                    f"{median_best:.4f}     "
                    f"{mean_time:>10.2f}"
                )

        print("\n" + "=" * 80)

    def get_winner(
        self, problem_name: str, results: Optional[List[BenchmarkResult]] = None
    ) -> Optional[str]:
        """Get best optimizer for a problem."""
        summary = self.get_summary(results)

        problem_results = [
            (data["optimizer"], data["mean_best_fitness"])
            for key, data in summary.items()
            if data["problem"] == problem_name
        ]

        if not problem_results:
            return None

        problem_results.sort(key=lambda x: x[1], reverse=True)
        return problem_results[0][0]

    def export_results(self, filename: str) -> None:
        """Export results to JSON."""
        import json

        data = {
            "results": [r.to_dict() for r in self.results],
            "summary": {str(k): v for k, v in self.get_summary().items()},
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results exported to {filename}")
