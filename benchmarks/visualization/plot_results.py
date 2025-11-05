"""Visualization tools for benchmark results.

Creates plots and charts for performance analysis.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import json
from typing import Any, Dict, List

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_scheduler_comparison(results_file: str = "scheduler_benchmark_results.json"):
    """
    Plot scheduler comparison charts.
    
    Args:
        results_file: Path to JSON results file
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available. Install with: pip install matplotlib")
        return
    
    # Load results
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # Extract data
    schedulers = []
    throughputs = []
    avg_times = []
    balance_scores = []
    
    for scheduler, data in results.items():
        if "error" in data:
            continue
        
        schedulers.append(scheduler)
        throughputs.append(data.get("throughput", 0))
        avg_times.append(data.get("avg_assignment_time", 0) * 1000)  # ms
        balance_scores.append(data.get("load_balance_score", 0))
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Throughput comparison
    axes[0].bar(schedulers, throughputs, color="steelblue")
    axes[0].set_ylabel("Throughput (tasks/s)")
    axes[0].set_title("Scheduler Throughput")
    axes[0].tick_params(axis="x", rotation=45)
    
    # Assignment time comparison
    axes[1].bar(schedulers, avg_times, color="coral")
    axes[1].set_ylabel("Avg Assignment Time (ms)")
    axes[1].set_title("Assignment Latency")
    axes[1].tick_params(axis="x", rotation=45)
    
    # Load balance comparison
    axes[2].bar(schedulers, balance_scores, color="seagreen")
    axes[2].set_ylabel("Load Balance Score (0-1)")
    axes[2].set_ylim(0, 1.0)
    axes[2].set_title("Load Balance")
    axes[2].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    plt.savefig("scheduler_comparison.png", dpi=150)
    print("Saved scheduler_comparison.png")
    plt.show()


def plot_scaling_results(results_file: str = "scaling_benchmark_results.json"):
    """
    Plot scaling benchmark results.
    
    Args:
        results_file: Path to JSON results file
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available. Install with: pip install matplotlib")
        return
    
    # Load results
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # Extract data
    workers = [r["num_workers"] for r in results]
    times = [r["total_time"] for r in results]
    throughputs = [r["throughput"] for r in results]
    speedups = [r.get("speedup", 1.0) for r in results]
    efficiencies = [r.get("efficiency", 1.0) for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Execution time vs workers
    axes[0, 0].plot(workers, times, marker="o", linewidth=2)
    axes[0, 0].set_xlabel("Number of Workers")
    axes[0, 0].set_ylabel("Execution Time (s)")
    axes[0, 0].set_title("Execution Time vs Workers")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale("log", base=2)
    
    # Throughput vs workers
    axes[0, 1].plot(workers, throughputs, marker="s", color="coral", linewidth=2)
    axes[0, 1].set_xlabel("Number of Workers")
    axes[0, 1].set_ylabel("Throughput (tasks/s)")
    axes[0, 1].set_title("Throughput vs Workers")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale("log", base=2)
    
    # Speedup
    axes[1, 0].plot(workers, speedups, marker="^", color="green", linewidth=2, label="Actual")
    axes[1, 0].plot(workers, workers, "--", color="gray", label="Ideal (Linear)")
    axes[1, 0].set_xlabel("Number of Workers")
    axes[1, 0].set_ylabel("Speedup")
    axes[1, 0].set_title("Speedup vs Workers")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_yscale("log", base=2)
    
    # Efficiency
    axes[1, 1].plot(workers, efficiencies, marker="d", color="purple", linewidth=2)
    axes[1, 1].axhline(y=1.0, color="gray", linestyle="--", label="Perfect Efficiency")
    axes[1, 1].set_xlabel("Number of Workers")
    axes[1, 1].set_ylabel("Efficiency")
    axes[1, 1].set_title("Parallel Efficiency")
    axes[1, 1].set_ylim(0, 1.2)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale("log", base=2)
    
    plt.tight_layout()
    plt.savefig("scaling_results.png", dpi=150)
    print("Saved scaling_results.png")
    plt.show()


def main():
    """Generate all plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument(
        "--type",
        choices=["scheduler", "scaling", "all"],
        default="all",
        help="Type of plot to generate",
    )
    
    args = parser.parse_args()
    
    if args.type in ["scheduler", "all"]:
        try:
            plot_scheduler_comparison()
        except FileNotFoundError:
            print("scheduler_benchmark_results.json not found. Run scheduler benchmark first.")
    
    if args.type in ["scaling", "all"]:
        try:
            plot_scaling_results()
        except FileNotFoundError:
            print("scaling_benchmark_results.json not found. Run scaling benchmark first.")


if __name__ == "__main__":
    main()
