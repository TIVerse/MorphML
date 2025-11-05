"""Metrics tracking and reporting for MorphML experiments.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ExperimentMetrics:
    """Metrics for a single experiment."""
    
    experiment_id: str
    optimizer: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Fitness metrics
    best_fitness: float = 0.0
    mean_fitness: float = 0.0
    fitness_std: float = 0.0
    fitness_history: List[float] = field(default_factory=list)
    
    # Performance metrics
    total_evaluations: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    
    # Timing metrics
    evaluation_times: List[float] = field(default_factory=list)
    
    # Architecture metrics
    best_architecture: Optional[Dict] = None
    architecture_depths: List[int] = field(default_factory=list)
    architecture_params: List[int] = field(default_factory=list)
    
    # Convergence metrics
    convergence_iteration: Optional[int] = None
    convergence_threshold: float = 0.001
    
    # Population metrics (for population-based algorithms)
    population_diversity: List[float] = field(default_factory=list)
    
    def update(self, iteration: int, fitness: float, architecture: Optional[Dict] = None):
        """Update metrics with new evaluation."""
        self.fitness_history.append(fitness)
        self.total_evaluations = iteration
        
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            if architecture:
                self.best_architecture = architecture
        
        # Update mean and std
        if self.fitness_history:
            self.mean_fitness = float(np.mean(self.fitness_history))
            self.fitness_std = float(np.std(self.fitness_history))
        
        # Detect convergence
        if not self.convergence_iteration and len(self.fitness_history) >= 10:
            recent_improvement = max(self.fitness_history[-10:]) - min(self.fitness_history[-10:])
            if recent_improvement < self.convergence_threshold:
                self.convergence_iteration = iteration
    
    def add_evaluation_time(self, eval_time: float):
        """Add evaluation timing."""
        self.evaluation_times.append(eval_time)
    
    def add_architecture_stats(self, depth: int, params: int):
        """Add architecture statistics."""
        self.architecture_depths.append(depth)
        self.architecture_params.append(params)
    
    def finalize(self):
        """Finalize metrics."""
        self.end_time = time.time()
    
    def get_summary(self) -> Dict:
        """Get summary metrics."""
        elapsed = self.end_time - self.start_time if self.end_time else time.time() - self.start_time
        
        return {
            "experiment_id": self.experiment_id,
            "optimizer": self.optimizer,
            "elapsed_time": elapsed,
            "total_evaluations": self.total_evaluations,
            "successful_evaluations": self.successful_evaluations,
            "failed_evaluations": self.failed_evaluations,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "fitness_std": self.fitness_std,
            "convergence_iteration": self.convergence_iteration,
            "evaluations_per_second": self.total_evaluations / elapsed if elapsed > 0 else 0,
            "avg_evaluation_time": np.mean(self.evaluation_times) if self.evaluation_times else 0,
            "median_evaluation_time": np.median(self.evaluation_times) if self.evaluation_times else 0,
            "avg_architecture_depth": np.mean(self.architecture_depths) if self.architecture_depths else 0,
            "avg_architecture_params": np.mean(self.architecture_params) if self.architecture_params else 0,
            "best_architecture": self.best_architecture
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "optimizer": self.optimizer,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metrics": self.get_summary(),
            "fitness_history": self.fitness_history,
            "evaluation_times": self.evaluation_times,
            "architecture_depths": self.architecture_depths,
            "architecture_params": self.architecture_params,
            "population_diversity": self.population_diversity
        }


class MetricsTracker:
    """Track metrics across multiple experiments."""
    
    def __init__(self, output_dir: str = "metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.experiments: Dict[str, ExperimentMetrics] = {}
    
    def create_experiment(self, experiment_id: str, optimizer: str) -> ExperimentMetrics:
        """Create new experiment tracking."""
        metrics = ExperimentMetrics(experiment_id=experiment_id, optimizer=optimizer)
        self.experiments[experiment_id] = metrics
        return metrics
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentMetrics]:
        """Get experiment metrics."""
        return self.experiments.get(experiment_id)
    
    def save_experiment(self, experiment_id: str):
        """Save experiment metrics to file."""
        if experiment_id not in self.experiments:
            return
        
        metrics = self.experiments[experiment_id]
        metrics.finalize()
        
        filename = self.output_dir / f"{experiment_id}_metrics.json"
        with open(filename, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
    
    def save_all(self):
        """Save all experiments."""
        for exp_id in self.experiments:
            self.save_experiment(exp_id)
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict:
        """Compare multiple experiments."""
        comparison = {}
        
        for exp_id in experiment_ids:
            if exp_id in self.experiments:
                metrics = self.experiments[exp_id]
                comparison[exp_id] = metrics.get_summary()
        
        return comparison
    
    def get_best_experiment(self) -> Optional[str]:
        """Get experiment with best fitness."""
        if not self.experiments:
            return None
        
        best_id = None
        best_fitness = -float('inf')
        
        for exp_id, metrics in self.experiments.items():
            if metrics.best_fitness > best_fitness:
                best_fitness = metrics.best_fitness
                best_id = exp_id
        
        return best_id
    
    def generate_report(self, output_file: str = "report.json"):
        """Generate comprehensive report."""
        report = {
            "total_experiments": len(self.experiments),
            "experiments": {},
            "comparison": {}
        }
        
        # Individual experiment summaries
        for exp_id, metrics in self.experiments.items():
            report["experiments"][exp_id] = metrics.get_summary()
        
        # Aggregate by optimizer
        by_optimizer: Dict[str, List[ExperimentMetrics]] = {}
        for metrics in self.experiments.values():
            if metrics.optimizer not in by_optimizer:
                by_optimizer[metrics.optimizer] = []
            by_optimizer[metrics.optimizer].append(metrics)
        
        for optimizer, metrics_list in by_optimizer.items():
            fitnesses = [m.best_fitness for m in metrics_list]
            convergences = [m.convergence_iteration for m in metrics_list if m.convergence_iteration]
            times = [m.end_time - m.start_time for m in metrics_list if m.end_time]
            
            report["comparison"][optimizer] = {
                "num_runs": len(metrics_list),
                "avg_best_fitness": float(np.mean(fitnesses)),
                "std_best_fitness": float(np.std(fitnesses)),
                "max_fitness": float(np.max(fitnesses)),
                "min_fitness": float(np.min(fitnesses)),
                "avg_convergence": float(np.mean(convergences)) if convergences else None,
                "avg_time": float(np.mean(times)) if times else None
            }
        
        # Save report
        report_path = self.output_dir / output_file
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


class PerformanceMetrics:
    """System performance metrics."""
    
    @staticmethod
    def calculate_sample_efficiency(
        fitness_history: List[float],
        target_fitness: float = 0.9
    ) -> Optional[int]:
        """Calculate number of evaluations to reach target fitness."""
        for i, fitness in enumerate(fitness_history):
            if fitness >= target_fitness:
                return i + 1
        return None
    
    @staticmethod
    def calculate_convergence_rate(fitness_history: List[float], window: int = 10) -> float:
        """Calculate convergence rate (improvement per evaluation)."""
        if len(fitness_history) < window:
            return 0.0
        
        recent = fitness_history[-window:]
        improvements = [recent[i] - recent[i-1] for i in range(1, len(recent))]
        return float(np.mean([imp for imp in improvements if imp > 0]))
    
    @staticmethod
    def calculate_exploration_ratio(
        unique_architectures: int,
        total_evaluations: int
    ) -> float:
        """Calculate ratio of unique architectures explored."""
        if total_evaluations == 0:
            return 0.0
        return unique_architectures / total_evaluations
    
    @staticmethod
    def calculate_regret(
        fitness_history: List[float],
        optimal_fitness: float = 1.0
    ) -> List[float]:
        """Calculate cumulative regret over time."""
        regrets = []
        cumulative_regret = 0.0
        
        for fitness in fitness_history:
            instant_regret = optimal_fitness - fitness
            cumulative_regret += instant_regret
            regrets.append(cumulative_regret)
        
        return regrets
    
    @staticmethod
    def calculate_hypervolume(
        pareto_front: List[tuple],
        reference_point: Optional[tuple] = None
    ) -> float:
        """Calculate hypervolume indicator for multi-objective optimization.
        
        Args:
            pareto_front: List of (obj1, obj2, ...) tuples
            reference_point: Reference point for hypervolume calculation
        
        Returns:
            Hypervolume value
        """
        if not pareto_front:
            return 0.0
        
        # Simple 2D hypervolume calculation
        if len(pareto_front[0]) == 2:
            if reference_point is None:
                reference_point = (0.0, float('inf'))
            
            # Sort by first objective
            sorted_front = sorted(pareto_front, key=lambda x: x[0], reverse=True)
            
            hypervolume = 0.0
            for i, point in enumerate(sorted_front):
                width = point[0] - (sorted_front[i+1][0] if i+1 < len(sorted_front) else reference_point[0])
                height = reference_point[1] - point[1]
                hypervolume += width * height
            
            return hypervolume
        
        # For higher dimensions, use approximation
        return 0.0


def create_benchmark_metrics() -> Dict[str, Dict]:
    """Create standard benchmark metrics template."""
    return {
        "sample_efficiency": {
            "description": "Number of evaluations to reach target fitness",
            "target_fitness": 0.9,
            "lower_is_better": True
        },
        "convergence_rate": {
            "description": "Average improvement per evaluation",
            "higher_is_better": True
        },
        "final_fitness": {
            "description": "Best fitness achieved",
            "higher_is_better": True
        },
        "time_to_convergence": {
            "description": "Time (seconds) to reach convergence",
            "lower_is_better": True
        },
        "evaluations_per_second": {
            "description": "Throughput of evaluations",
            "higher_is_better": True
        },
        "exploration_ratio": {
            "description": "Ratio of unique architectures",
            "higher_is_better": True
        },
        "cumulative_regret": {
            "description": "Total regret accumulated",
            "lower_is_better": True
        }
    }


if __name__ == "__main__":
    # Example usage
    tracker = MetricsTracker("example_metrics")
    
    # Create experiment
    metrics = tracker.create_experiment("test_exp_1", "GeneticAlgorithm")
    
    # Simulate updates
    for i in range(100):
        fitness = 0.5 + i * 0.005 + np.random.normal(0, 0.01)
        metrics.update(i + 1, fitness)
        metrics.add_evaluation_time(np.random.uniform(0.1, 0.5))
        metrics.add_architecture_stats(
            depth=np.random.randint(5, 15),
            params=np.random.randint(1000000, 10000000)
        )
    
    # Save and report
    tracker.save_experiment("test_exp_1")
    report = tracker.generate_report()
    
    print(json.dumps(report, indent=2))
