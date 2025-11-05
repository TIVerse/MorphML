"""Benchmark different task schedulers.

Compares FIFO, Priority, Load Balancing, Work Stealing, and Adaptive schedulers.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import time
from typing import Any, Dict, List

from morphml.core.dsl import Layer, SearchSpace
from morphml.distributed import (
    Task,
    WorkerInfo,
    create_scheduler,
)
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class SchedulerBenchmark:
    """
    Benchmark task schedulers.
    
    Measures:
    - Task assignment time
    - Load distribution fairness
    - Throughput (tasks/second)
    - Overhead per task
    
    Example:
        >>> benchmark = SchedulerBenchmark()
        >>> results = benchmark.run_all()
        >>> print(results['fifo']['throughput'])
    """
    
    def __init__(self, num_tasks: int = 1000, num_workers: int = 10):
        """
        Initialize benchmark.
        
        Args:
            num_tasks: Number of tasks to schedule
            num_workers: Number of simulated workers
        """
        self.num_tasks = num_tasks
        self.num_workers = num_workers
        
        # Create test tasks
        self.tasks = self._create_tasks()
        self.workers = self._create_workers()
        
        logger.info(
            f"Initialized SchedulerBenchmark: {num_tasks} tasks, {num_workers} workers"
        )
    
    def _create_tasks(self) -> List[Task]:
        """Create test tasks."""
        space = SearchSpace("benchmark")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64),
            Layer.output(units=10),
        )
        
        tasks = []
        for i in range(self.num_tasks):
            graph = space.sample()
            tasks.append(Task(f"task-{i}", graph))
        
        return tasks
    
    def _create_workers(self) -> List[WorkerInfo]:
        """Create simulated workers."""
        workers = []
        for i in range(self.num_workers):
            worker = WorkerInfo(
                worker_id=f"worker-{i}",
                host=f"host-{i}",
                port=50052 + i,
                num_gpus=1 + (i % 4),  # 1-4 GPUs
                status="idle",
            )
            workers.append(worker)
        
        return workers
    
    def benchmark_scheduler(self, scheduler_type: str) -> Dict[str, Any]:
        """
        Benchmark a specific scheduler.
        
        Args:
            scheduler_type: Scheduler name
        
        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Benchmarking {scheduler_type} scheduler...")
        
        # Create scheduler
        scheduler = create_scheduler(scheduler_type)
        
        # Reset workers
        for worker in self.workers:
            worker.status = "idle"
            worker.tasks_completed = 0
        
        # Measure assignment time
        assignments = []
        start_time = time.time()
        
        for task in self.tasks:
            # Find available worker
            available = [w for w in self.workers if w.status == "idle"]
            
            if available:
                assign_start = time.time()
                worker = scheduler.assign_task(task, available)
                assign_time = time.time() - assign_start
                
                if worker:
                    assignments.append({
                        "task_id": task.task_id,
                        "worker_id": worker.worker_id,
                        "assignment_time": assign_time,
                    })
                    
                    # Simulate task execution
                    worker.status = "busy"
                    worker.tasks_completed += 1
                    
                    # Make worker available again (simulate completion)
                    if worker.tasks_completed % 10 == 0:
                        worker.status = "idle"
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        assignment_times = [a["assignment_time"] for a in assignments]
        worker_loads = {w.worker_id: w.tasks_completed for w in self.workers}
        
        results = {
            "scheduler": scheduler_type,
            "total_time": total_time,
            "tasks_assigned": len(assignments),
            "throughput": len(assignments) / total_time if total_time > 0 else 0,
            "avg_assignment_time": sum(assignment_times) / len(assignment_times) if assignment_times else 0,
            "min_assignment_time": min(assignment_times) if assignment_times else 0,
            "max_assignment_time": max(assignment_times) if assignment_times else 0,
            "worker_loads": worker_loads,
            "load_std_dev": self._calculate_std_dev(list(worker_loads.values())),
            "load_balance_score": self._calculate_load_balance(worker_loads),
        }
        
        logger.info(
            f"{scheduler_type}: {results['throughput']:.1f} tasks/s, "
            f"avg={results['avg_assignment_time']*1000:.2f}ms"
        )
        
        return results
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_load_balance(self, worker_loads: Dict[str, int]) -> float:
        """
        Calculate load balance score (0-1, 1=perfect balance).
        
        Uses coefficient of variation (inverse).
        """
        loads = list(worker_loads.values())
        if not loads or max(loads) == 0:
            return 1.0
        
        mean = sum(loads) / len(loads)
        std_dev = self._calculate_std_dev(loads)
        
        # Coefficient of variation
        cv = std_dev / mean if mean > 0 else 0
        
        # Convert to 0-1 score (lower CV = better balance)
        score = 1.0 / (1.0 + cv)
        
        return score
    
    def run_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Run benchmarks for all schedulers.
        
        Returns:
            Dictionary mapping scheduler name to results
        """
        schedulers = [
            "fifo",
            "priority",
            "load_balancing",
            "work_stealing",
            "adaptive",
            "round_robin",
        ]
        
        results = {}
        
        for scheduler_type in schedulers:
            try:
                results[scheduler_type] = self.benchmark_scheduler(scheduler_type)
            except Exception as e:
                logger.error(f"Failed to benchmark {scheduler_type}: {e}")
                results[scheduler_type] = {"error": str(e)}
        
        return results
    
    def print_summary(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Print benchmark summary."""
        print("\n" + "="*80)
        print("SCHEDULER BENCHMARK RESULTS")
        print("="*80)
        print(f"Tasks: {self.num_tasks}, Workers: {self.num_workers}\n")
        
        # Sort by throughput
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if "error" not in v],
            key=lambda x: x[1].get("throughput", 0),
            reverse=True,
        )
        
        print(f"{'Scheduler':<20} {'Throughput':<15} {'Avg Time':<15} {'Balance':<10}")
        print("-"*80)
        
        for scheduler, data in sorted_results:
            throughput = data.get("throughput", 0)
            avg_time = data.get("avg_assignment_time", 0) * 1000  # ms
            balance = data.get("load_balance_score", 0)
            
            print(f"{scheduler:<20} {throughput:>10.1f} t/s  {avg_time:>10.2f} ms   {balance:>7.3f}")
        
        print("="*80 + "\n")


def main():
    """Run scheduler benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark MorphML schedulers")
    parser.add_argument("--tasks", type=int, default=1000, help="Number of tasks")
    parser.add_argument("--workers", type=int, default=10, help="Number of workers")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = SchedulerBenchmark(num_tasks=args.tasks, num_workers=args.workers)
    results = benchmark.run_all()
    benchmark.print_summary(results)
    
    # Save results
    import json
    with open("scheduler_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to scheduler_benchmark_results.json")


if __name__ == "__main__":
    main()
