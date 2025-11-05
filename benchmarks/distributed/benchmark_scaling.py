"""Benchmark distributed system scaling characteristics.

Tests how MorphML performs as workers are added.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import time
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from morphml.core.dsl import Layer, SearchSpace
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class ScalingBenchmark:
    """
    Benchmark scaling characteristics.
    
    Measures:
    - Throughput vs number of workers
    - Speedup and efficiency
    - Overhead of coordination
    - Optimal worker count
    
    Example:
        >>> benchmark = ScalingBenchmark()
        >>> results = benchmark.run_scaling_test()
        >>> benchmark.plot_results(results)
    """
    
    def __init__(self, num_tasks: int = 1000):
        """
        Initialize scaling benchmark.
        
        Args:
            num_tasks: Number of tasks to process
        """
        self.num_tasks = num_tasks
        self.space = self._create_search_space()
        
        logger.info(f"Initialized ScalingBenchmark: {num_tasks} tasks")
    
    def _create_search_space(self) -> SearchSpace:
        """Create test search space."""
        space = SearchSpace("scaling_benchmark")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=32),
            Layer.conv2d(filters=64),
            Layer.flatten(),
            Layer.dense(units=128),
            Layer.output(units=10),
        )
        return space
    
    def _simulate_evaluation(self, task_id: int) -> Dict:
        """Simulate architecture evaluation."""
        # Sample architecture
        graph = self.space.sample()
        
        # Simulate evaluation time (0.1-0.5s)
        eval_time = 0.1 + (hash(str(task_id)) % 100) / 250.0
        time.sleep(eval_time)
        
        # Simulated fitness
        fitness = 0.5 + (hash(str(task_id)) % 500) / 1000.0
        
        return {
            "task_id": task_id,
            "fitness": fitness,
            "eval_time": eval_time,
        }
    
    def benchmark_workers(self, num_workers: int) -> Dict:
        """
        Benchmark with specific number of workers.
        
        Args:
            num_workers: Number of parallel workers
        
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking with {num_workers} workers...")
        
        start_time = time.time()
        results = []
        
        # Use ThreadPoolExecutor to simulate workers
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._simulate_evaluation, i): i
                for i in range(self.num_tasks)
            }
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Task failed: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        throughput = len(results) / total_time
        avg_eval_time = sum(r["eval_time"] for r in results) / len(results)
        
        return {
            "num_workers": num_workers,
            "total_time": total_time,
            "tasks_completed": len(results),
            "throughput": throughput,
            "avg_eval_time": avg_eval_time,
        }
    
    def run_scaling_test(
        self, worker_counts: List[int] = None
    ) -> List[Dict]:
        """
        Run scaling test with different worker counts.
        
        Args:
            worker_counts: List of worker counts to test
        
        Returns:
            List of results for each worker count
        """
        if worker_counts is None:
            worker_counts = [1, 2, 4, 8, 16, 32, 64]
        
        results = []
        baseline_time = None
        
        for num_workers in worker_counts:
            result = self.benchmark_workers(num_workers)
            
            # Calculate speedup and efficiency
            if baseline_time is None:
                baseline_time = result["total_time"]
                result["speedup"] = 1.0
                result["efficiency"] = 1.0
            else:
                speedup = baseline_time / result["total_time"]
                efficiency = speedup / num_workers
                result["speedup"] = speedup
                result["efficiency"] = efficiency
            
            results.append(result)
            
            logger.info(
                f"Workers: {num_workers:3d} | "
                f"Time: {result['total_time']:6.2f}s | "
                f"Throughput: {result['throughput']:6.1f} t/s | "
                f"Speedup: {result.get('speedup', 0):5.2f}x | "
                f"Efficiency: {result.get('efficiency', 0):5.2%}"
            )
        
        return results
    
    def print_summary(self, results: List[Dict]) -> None:
        """Print scaling benchmark summary."""
        print("\n" + "="*90)
        print("SCALING BENCHMARK RESULTS")
        print("="*90)
        print(f"Total tasks: {self.num_tasks}\n")
        
        print(f"{'Workers':<10} {'Time (s)':<12} {'Throughput':<15} {'Speedup':<12} {'Efficiency':<12}")
        print("-"*90)
        
        for result in results:
            workers = result["num_workers"]
            total_time = result["total_time"]
            throughput = result["throughput"]
            speedup = result.get("speedup", 0)
            efficiency = result.get("efficiency", 0)
            
            print(
                f"{workers:<10} {total_time:<12.2f} {throughput:<15.1f} "
                f"{speedup:<12.2f}x {efficiency:<12.1%}"
            )
        
        print("="*90 + "\n")
        
        # Find optimal worker count
        best_efficiency = max(results, key=lambda x: x.get("efficiency", 0))
        best_throughput = max(results, key=lambda x: x["throughput"])
        
        print(f"Best efficiency: {best_efficiency['num_workers']} workers "
              f"({best_efficiency['efficiency']:.1%})")
        print(f"Best throughput: {best_throughput['num_workers']} workers "
              f"({best_throughput['throughput']:.1f} tasks/s)\n")


def main():
    """Run scaling benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark MorphML scaling")
    parser.add_argument("--tasks", type=int, default=100, help="Number of tasks")
    parser.add_argument(
        "--workers",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated worker counts",
    )
    
    args = parser.parse_args()
    
    worker_counts = [int(x) for x in args.workers.split(",")]
    
    # Run benchmark
    benchmark = ScalingBenchmark(num_tasks=args.tasks)
    results = benchmark.run_scaling_test(worker_counts)
    benchmark.print_summary(results)
    
    # Save results
    import json
    with open("scaling_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to scaling_benchmark_results.json")


if __name__ == "__main__":
    main()
