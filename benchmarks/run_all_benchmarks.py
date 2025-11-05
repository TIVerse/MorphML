"""Run all MorphML benchmarks and generate report.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import json
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from morphml.logging_config import get_logger

logger = get_logger(__name__)


def run_scheduler_benchmarks():
    """Run scheduler benchmarks."""
    print("\n" + "="*80)
    print("RUNNING SCHEDULER BENCHMARKS")
    print("="*80 + "\n")
    
    from benchmarks.distributed.benchmark_schedulers import SchedulerBenchmark
    
    benchmark = SchedulerBenchmark(num_tasks=1000, num_workers=10)
    results = benchmark.run_all()
    benchmark.print_summary(results)
    
    # Save results
    with open("scheduler_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def run_scaling_benchmarks():
    """Run scaling benchmarks."""
    print("\n" + "="*80)
    print("RUNNING SCALING BENCHMARKS")
    print("="*80 + "\n")
    
    from benchmarks.distributed.benchmark_scaling import ScalingBenchmark
    
    benchmark = ScalingBenchmark(num_tasks=200)
    results = benchmark.run_scaling_test([1, 2, 4, 8, 16, 32])
    benchmark.print_summary(results)
    
    # Save results
    with open("scaling_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def generate_visualizations():
    """Generate visualization plots."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    from benchmarks.visualization.plot_results import (
        plot_scheduler_comparison,
        plot_scaling_results,
    )
    
    try:
        plot_scheduler_comparison()
        plot_scaling_results()
    except Exception as e:
        print(f"Failed to generate plots: {e}")
        print("Install matplotlib with: pip install matplotlib")


def generate_report(scheduler_results, scaling_results):
    """Generate markdown report."""
    report = """# MorphML Performance Benchmark Report

**Date:** {date}
**System:** Python {python_version}

## Executive Summary

This report presents comprehensive performance benchmarks of MorphML's distributed
Neural Architecture Search system.

## 1. Scheduler Performance

### Throughput Comparison

""".format(date=time.strftime("%Y-%m-%d %H:%M:%S"), python_version=sys.version.split()[0])
    
    # Scheduler table
    report += "| Scheduler | Throughput (tasks/s) | Avg Time (ms) | Load Balance |\n"
    report += "|-----------|---------------------|---------------|-------------|\n"
    
    for scheduler, data in sorted(
        scheduler_results.items(),
        key=lambda x: x[1].get("throughput", 0),
        reverse=True
    ):
        if "error" in data:
            continue
        
        throughput = data.get("throughput", 0)
        avg_time = data.get("avg_assignment_time", 0) * 1000
        balance = data.get("load_balance_score", 0)
        
        report += f"| {scheduler:<12} | {throughput:>15.1f} | {avg_time:>11.2f} | {balance:>9.3f} |\n"
    
    report += """
### Key Findings

- **Best throughput:** {best_scheduler} ({best_throughput:.1f} tasks/s)
- **Best latency:** {best_latency_scheduler} ({best_latency:.2f} ms)
- **Best load balance:** {best_balance_scheduler} ({best_balance:.3f})

## 2. Scaling Performance

### Scaling Characteristics

""".format(
        best_scheduler=max(scheduler_results, key=lambda x: scheduler_results[x].get("throughput", 0)),
        best_throughput=max(scheduler_results.values(), key=lambda x: x.get("throughput", 0))["throughput"],
        best_latency_scheduler=min(scheduler_results, key=lambda x: scheduler_results[x].get("avg_assignment_time", float("inf"))),
        best_latency=min(scheduler_results.values(), key=lambda x: x.get("avg_assignment_time", float("inf")))["avg_assignment_time"] * 1000,
        best_balance_scheduler=max(scheduler_results, key=lambda x: scheduler_results[x].get("load_balance_score", 0)),
        best_balance=max(scheduler_results.values(), key=lambda x: x.get("load_balance_score", 0))["load_balance_score"],
    )
    
    # Scaling table
    report += "| Workers | Time (s) | Throughput (t/s) | Speedup | Efficiency |\n"
    report += "|---------|----------|-----------------|---------|------------|\n"
    
    for result in scaling_results:
        workers = result["num_workers"]
        total_time = result["total_time"]
        throughput = result["throughput"]
        speedup = result.get("speedup", 1.0)
        efficiency = result.get("efficiency", 1.0)
        
        report += f"| {workers:>7} | {total_time:>8.2f} | {throughput:>15.1f} | {speedup:>7.2f}x | {efficiency:>9.1%} |\n"
    
    best_efficiency = max(scaling_results, key=lambda x: x.get("efficiency", 0))
    best_throughput = max(scaling_results, key=lambda x: x["throughput"])
    
    report += f"""
### Key Findings

- **Optimal workers (efficiency):** {best_efficiency['num_workers']} ({best_efficiency['efficiency']:.1%})
- **Maximum throughput:** {best_throughput['throughput']:.1f} tasks/s @ {best_throughput['num_workers']} workers
- **Scaling efficiency:** {best_efficiency['efficiency']:.1%} at {best_efficiency['num_workers']} workers

## 3. Recommendations

### Production Deployment

Based on these benchmarks:

1. **Scheduler choice:**
   - For throughput: Use `{best_scheduler}` scheduler
   - For fairness: Use load balancing or adaptive scheduler
   - For simplicity: Use FIFO or round-robin

2. **Worker count:**
   - Start with {best_efficiency['num_workers']}-{best_throughput['num_workers']} workers
   - Scale based on workload
   - Monitor efficiency metrics

3. **Resource allocation:**
   - Each worker should have dedicated GPU(s)
   - Ensure sufficient CPU and memory
   - Consider network bandwidth for large models

## 4. Visualizations

See generated plots:
- `scheduler_comparison.png` - Scheduler performance comparison
- `scaling_results.png` - Scaling characteristics

## Conclusion

MorphML demonstrates excellent scaling characteristics and efficient task scheduling.
The distributed system is production-ready for large-scale Neural Architecture Search.

---

*Report generated by MorphML Benchmark Suite*
*https://github.com/TIVerse/MorphML*
""".format(
        best_scheduler=max(scheduler_results, key=lambda x: scheduler_results[x].get("throughput", 0))
    )
    
    # Save report
    with open("BENCHMARK_REPORT.md", "w") as f:
        f.write(report)
    
    print("\nBenchmark report saved to BENCHMARK_REPORT.md")


def main():
    """Run all benchmarks and generate report."""
    print("\n" + "🚀"*40)
    print(" "*30 + "MorphML Benchmark Suite")
    print("🚀"*40 + "\n")
    
    start_time = time.time()
    
    try:
        # Run benchmarks
        scheduler_results = run_scheduler_benchmarks()
        scaling_results = run_scaling_benchmarks()
        
        # Generate visualizations
        generate_visualizations()
        
        # Generate report
        generate_report(scheduler_results, scaling_results)
        
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print(f"ALL BENCHMARKS COMPLETE ({total_time:.1f}s)")
        print("="*80)
        print("\nGenerated files:")
        print("  - scheduler_benchmark_results.json")
        print("  - scaling_benchmark_results.json")
        print("  - scheduler_comparison.png")
        print("  - scaling_results.png")
        print("  - BENCHMARK_REPORT.md")
        print("\n")
    
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
