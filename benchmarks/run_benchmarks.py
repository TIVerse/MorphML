#!/usr/bin/env python3
"""Comprehensive benchmark suite for MorphML optimizers.

Benchmarks all optimizers on standard datasets and generates comparison reports.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from morphml.core.dsl import Layer, SearchSpace
from morphml.core.graph import ModelGraph
from morphml.optimizers import (
    GeneticAlgorithm,
    RandomSearch,
    HillClimbing,
)

# Try to import optional optimizers
try:
    from morphml.optimizers.bayesian import GaussianProcessOptimizer, TPEOptimizer
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

try:
    from morphml.optimizers.multi_objective import NSGA2
    MULTI_OBJ_AVAILABLE = True
except ImportError:
    MULTI_OBJ_AVAILABLE = False

console = Console()


class BenchmarkConfig:
    """Benchmark configuration."""
    
    def __init__(
        self,
        max_evaluations: int = 100,
        num_runs: int = 3,
        output_dir: str = "benchmark_results",
        save_results: bool = True
    ):
        self.max_evaluations = max_evaluations
        self.num_runs = num_runs
        self.output_dir = Path(output_dir)
        self.save_results = save_results
        
        if save_results:
            self.output_dir.mkdir(exist_ok=True, parents=True)


class MockEvaluator:
    """Mock evaluator for benchmarking without actual training.
    
    Simulates evaluation based on architecture characteristics.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.eval_count = 0
        self.eval_times = []
    
    def evaluate(self, graph: ModelGraph) -> Dict[str, float]:
        """Evaluate architecture (mock)."""
        start_time = time.time()
        
        # Simulate evaluation time (0.1 - 0.5 seconds)
        time.sleep(self.rng.uniform(0.1, 0.5))
        
        # Score based on architecture characteristics
        depth = graph.get_depth()
        params = graph.estimate_parameters()
        
        # Deeper networks up to a point (7-12 layers optimal)
        depth_score = 1.0 - abs(10 - depth) / 20.0
        depth_score = max(0.3, min(1.0, depth_score))
        
        # Prefer moderate parameter count (1M - 10M optimal)
        param_score = 1.0 - abs(np.log10(max(params, 1)) - 6.5) / 2.0
        param_score = max(0.3, min(1.0, param_score))
        
        # Add some noise
        noise = self.rng.normal(0, 0.05)
        
        accuracy = 0.6 + 0.3 * depth_score + 0.1 * param_score + noise
        accuracy = max(0.5, min(0.99, accuracy))
        
        # Latency correlates with depth and params
        latency = depth * 5 + params / 1e6 * 10 + self.rng.uniform(-5, 5)
        
        elapsed = time.time() - start_time
        self.eval_times.append(elapsed)
        self.eval_count += 1
        
        return {
            "accuracy": accuracy,
            "latency": latency,
            "params": params,
            "depth": depth
        }
    
    def get_stats(self) -> Dict[str, float]:
        """Get evaluator statistics."""
        return {
            "total_evaluations": self.eval_count,
            "avg_eval_time": np.mean(self.eval_times) if self.eval_times else 0,
            "total_eval_time": sum(self.eval_times)
        }


class BenchmarkResult:
    """Results from a single benchmark run."""
    
    def __init__(self, optimizer_name: str, run_id: int):
        self.optimizer_name = optimizer_name
        self.run_id = run_id
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.evaluations = []
        self.start_time = None
        self.end_time = None
        self.final_best_fitness = 0.0
        self.convergence_iteration = None
        self.num_evaluations = 0
    
    def add_evaluation(self, iteration: int, best_fitness: float, mean_fitness: float):
        """Add evaluation result."""
        self.best_fitness_history.append(best_fitness)
        self.mean_fitness_history.append(mean_fitness)
        self.evaluations.append(iteration)
        self.num_evaluations = iteration
    
    def finalize(self, best_fitness: float):
        """Finalize benchmark."""
        self.end_time = time.time()
        self.final_best_fitness = best_fitness
        
        # Detect convergence (when improvement < 0.001 for 10 iterations)
        if len(self.best_fitness_history) >= 10:
            for i in range(10, len(self.best_fitness_history)):
                improvement = self.best_fitness_history[i] - self.best_fitness_history[i-10]
                if abs(improvement) < 0.001:
                    self.convergence_iteration = i
                    break
    
    def get_metrics(self) -> Dict[str, float]:
        """Get benchmark metrics."""
        elapsed_time = self.end_time - self.start_time if self.end_time else 0
        
        return {
            "final_best_fitness": self.final_best_fitness,
            "convergence_iteration": self.convergence_iteration or self.num_evaluations,
            "elapsed_time": elapsed_time,
            "evaluations_per_second": self.num_evaluations / elapsed_time if elapsed_time > 0 else 0,
            "mean_final_fitness": np.mean(self.best_fitness_history[-10:]) if len(self.best_fitness_history) >= 10 else self.final_best_fitness
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "optimizer": self.optimizer_name,
            "run_id": self.run_id,
            "metrics": self.get_metrics(),
            "best_fitness_history": self.best_fitness_history,
            "mean_fitness_history": self.mean_fitness_history,
            "evaluations": self.evaluations
        }


def create_search_space(name: str = "benchmark") -> SearchSpace:
    """Create benchmark search space."""
    space = SearchSpace(name)
    
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5]),
        Layer.relu(),
        Layer.batchnorm(),
        Layer.maxpool(pool_size=2),
        Layer.conv2d(filters=[64, 128, 256], kernel_size=[3, 5]),
        Layer.relu(),
        Layer.batchnorm(),
        Layer.maxpool(pool_size=2),
        Layer.conv2d(filters=[128, 256, 512], kernel_size=3),
        Layer.relu(),
        Layer.flatten(),
        Layer.dense(units=[256, 512, 1024]),
        Layer.relu(),
        Layer.dropout(rate=[0.3, 0.5]),
        Layer.dense(units=[128, 256, 512]),
        Layer.relu(),
        Layer.output(units=10)
    )
    
    return space


def benchmark_optimizer(
    optimizer_class,
    optimizer_config: Dict,
    search_space: SearchSpace,
    evaluator: MockEvaluator,
    max_evaluations: int,
    run_id: int
) -> BenchmarkResult:
    """Benchmark a single optimizer."""
    optimizer_name = optimizer_class.__name__
    result = BenchmarkResult(optimizer_name, run_id)
    result.start_time = time.time()
    
    # Initialize optimizer
    optimizer = optimizer_class(search_space=search_space, **optimizer_config)
    
    # Run optimization
    best_fitness = 0.0
    for iteration in range(max_evaluations):
        # Generate candidate
        if hasattr(optimizer, 'ask'):
            candidates = optimizer.ask()
        else:
            # For optimizers without ask/tell interface
            candidates = [search_space.sample()]
        
        # Evaluate
        fitnesses = []
        for candidate in candidates:
            eval_result = evaluator.evaluate(candidate)
            fitness = eval_result['accuracy']
            fitnesses.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
        
        # Update optimizer
        if hasattr(optimizer, 'tell'):
            optimizer.tell(list(zip(candidates, fitnesses)))
        
        # Record metrics
        mean_fitness = np.mean(fitnesses) if fitnesses else 0
        result.add_evaluation(iteration + 1, best_fitness, mean_fitness)
    
    result.finalize(best_fitness)
    return result


def run_benchmark_suite(config: BenchmarkConfig) -> Dict[str, List[BenchmarkResult]]:
    """Run complete benchmark suite."""
    console.print("\n[bold cyan]🧬 MorphML Benchmark Suite[/bold cyan]")
    console.print(f"Max evaluations: {config.max_evaluations}")
    console.print(f"Runs per optimizer: {config.num_runs}")
    console.print()
    
    # Create search space
    search_space = create_search_space()
    
    # Define optimizers to benchmark
    optimizers = [
        (RandomSearch, {"num_samples": config.max_evaluations}),
        (GeneticAlgorithm, {
            "population_size": 20,
            "num_generations": config.max_evaluations // 20,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7
        }),
        (HillClimbing, {
            "max_iterations": config.max_evaluations,
            "step_size": 0.1
        }),
    ]
    
    if BAYESIAN_AVAILABLE:
        optimizers.extend([
            (GaussianProcessOptimizer, {
                "n_initial_points": 10,
                "acquisition": "ei"
            }),
            (TPEOptimizer, {
                "n_initial_points": 10
            })
        ])
    
    if MULTI_OBJ_AVAILABLE:
        optimizers.append((NSGA2, {
            "population_size": 20,
            "num_generations": config.max_evaluations // 20
        }))
    
    # Run benchmarks
    all_results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for optimizer_class, optimizer_config in optimizers:
            optimizer_name = optimizer_class.__name__
            task = progress.add_task(
                f"[cyan]Benchmarking {optimizer_name}...",
                total=config.num_runs
            )
            
            results = []
            for run_id in range(config.num_runs):
                evaluator = MockEvaluator(seed=42 + run_id)
                result = benchmark_optimizer(
                    optimizer_class,
                    optimizer_config,
                    search_space,
                    evaluator,
                    config.max_evaluations,
                    run_id
                )
                results.append(result)
                progress.update(task, advance=1)
            
            all_results[optimizer_name] = results
            progress.update(task, description=f"[green]✓ {optimizer_name}")
    
    return all_results


def print_results_table(all_results: Dict[str, List[BenchmarkResult]]):
    """Print results in a table."""
    table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
    
    table.add_column("Optimizer", style="cyan")
    table.add_column("Avg Best Fitness", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_column("Avg Convergence", justify="right")
    table.add_column("Avg Time (s)", justify="right")
    table.add_column("Evals/sec", justify="right")
    
    for optimizer_name, results in all_results.items():
        metrics = [r.get_metrics() for r in results]
        
        avg_fitness = np.mean([m['final_best_fitness'] for m in metrics])
        std_fitness = np.std([m['final_best_fitness'] for m in metrics])
        avg_convergence = np.mean([m['convergence_iteration'] for m in metrics])
        avg_time = np.mean([m['elapsed_time'] for m in metrics])
        avg_evals_per_sec = np.mean([m['evaluations_per_second'] for m in metrics])
        
        table.add_row(
            optimizer_name,
            f"{avg_fitness:.4f}",
            f"±{std_fitness:.4f}",
            f"{avg_convergence:.0f}",
            f"{avg_time:.2f}",
            f"{avg_evals_per_sec:.2f}"
        )
    
    console.print("\n")
    console.print(table)
    console.print()


def save_results(all_results: Dict[str, List[BenchmarkResult]], config: BenchmarkConfig):
    """Save results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_data = {}
    for optimizer_name, results in all_results.items():
        results_data[optimizer_name] = [r.to_dict() for r in results]
    
    results_file = config.output_dir / f"benchmark_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    console.print(f"[green]✓ Detailed results saved to: {results_file}[/green]")
    
    # Save summary
    summary = {}
    for optimizer_name, results in all_results.items():
        metrics = [r.get_metrics() for r in results]
        summary[optimizer_name] = {
            "avg_best_fitness": float(np.mean([m['final_best_fitness'] for m in metrics])),
            "std_best_fitness": float(np.std([m['final_best_fitness'] for m in metrics])),
            "avg_convergence_iteration": float(np.mean([m['convergence_iteration'] for m in metrics])),
            "avg_elapsed_time": float(np.mean([m['elapsed_time'] for m in metrics])),
            "avg_evaluations_per_second": float(np.mean([m['evaluations_per_second'] for m in metrics]))
        }
    
    summary_file = config.output_dir / f"benchmark_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    console.print(f"[green]✓ Summary saved to: {summary_file}[/green]")


def main():
    """Main benchmark runner."""
    console.print("[bold cyan]" + "="*70 + "[/bold cyan]")
    console.print("[bold cyan]  MorphML Benchmark Suite  [/bold cyan]")
    console.print("[bold cyan]  Testing All Optimizers on Standard Search Space  [/bold cyan]")
    console.print("[bold cyan]" + "="*70 + "[/bold cyan]")
    
    # Configuration
    config = BenchmarkConfig(
        max_evaluations=100,
        num_runs=3,
        output_dir="benchmark_results",
        save_results=True
    )
    
    # Run benchmarks
    all_results = run_benchmark_suite(config)
    
    # Print results
    print_results_table(all_results)
    
    # Save results
    if config.save_results:
        save_results(all_results, config)
    
    console.print("[bold green]✓ Benchmark suite completed![/bold green]\n")
    
    # Print winner
    best_optimizer = None
    best_fitness = 0.0
    for optimizer_name, results in all_results.items():
        avg_fitness = np.mean([r.final_best_fitness for r in results])
        if avg_fitness > best_fitness:
            best_fitness = avg_fitness
            best_optimizer = optimizer_name
    
    console.print(f"[bold yellow]🏆 Best Optimizer: {best_optimizer} (fitness: {best_fitness:.4f})[/bold yellow]\n")


if __name__ == "__main__":
    main()
