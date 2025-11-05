"""
Benchmark Suite Example

Demonstrates how to compare multiple optimizers using the benchmarking framework.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
Repository: https://github.com/TIVerse/MorphML

Usage:
    python examples/benchmark_suite_example.py
"""

from morphml.benchmarks import (
    BenchmarkSuite,
    OptimizerComparator,
    SimpleProblem,
    ComplexProblem,
    get_all_problems,
)
from morphml.optimizers import (
    GeneticAlgorithm,
    RandomSearch,
    HillClimbing,
    GaussianProcessOptimizer,
    TPEOptimizer,
)


def example_1_simple_comparison():
    """Example 1: Compare optimizers on a simple problem."""
    print("\n" + "="*70)
    print("Example 1: Simple Optimizer Comparison")
    print("="*70)
    
    # Create benchmark suite
    suite = BenchmarkSuite()
    
    # Add optimizers to compare
    suite.add_optimizer("GA", GeneticAlgorithm, {
        'population_size': 10,
        'num_generations': 10
    })
    
    suite.add_optimizer("RS", RandomSearch, {
        'num_samples': 100
    })
    
    suite.add_optimizer("HC", HillClimbing, {
        'max_iterations': 100
    })
    
    # Create benchmark problem
    problem = SimpleProblem()
    
    print(f"\nRunning benchmark on: {problem.name}")
    print(f"  Optimizers: 3")
    print(f"  Runs per optimizer: 3")
    
    # Run benchmark
    results = suite.run([problem], num_runs=3)
    
    print(f"\n✓ Benchmark complete: {len(results)} total runs")
    
    # Display summary
    suite.print_summary()
    
    return results


def example_2_multiple_problems():
    """Example 2: Compare optimizers on multiple problems."""
    print("\n" + "="*70)
    print("Example 2: Multiple Benchmark Problems")
    print("="*70)
    
    suite = BenchmarkSuite()
    
    # Add optimizers
    suite.add_optimizer("GA", GeneticAlgorithm, {
        'population_size': 10,
        'num_generations': 8
    })
    
    suite.add_optimizer("TPE", TPEOptimizer, {
        'n_initial_points': 10,
        'max_iterations': 80
    })
    
    # Get multiple problems
    problems = [SimpleProblem(), ComplexProblem()]
    
    print(f"\nRunning benchmark:")
    print(f"  Optimizers: 2")
    print(f"  Problems: {len(problems)}")
    print(f"  Runs per combination: 2")
    
    # Run benchmark
    results = suite.run(problems, num_runs=2)
    
    print(f"\n✓ Complete: {len(results)} total runs")
    
    # Display summary
    suite.print_summary()
    
    return results


def example_3_optimizer_comparator():
    """Example 3: Using OptimizerComparator for statistical analysis."""
    print("\n" + "="*70)
    print("Example 3: Statistical Comparison with OptimizerComparator")
    print("="*70)
    
    comparator = OptimizerComparator()
    
    # Simulate results from multiple runs
    problem_name = "TestProblem"
    
    # GA results (5 runs)
    ga_fitnesses = [0.85, 0.88, 0.87, 0.86, 0.89]
    comparator.add_result("GA", problem_name, ga_fitnesses)
    
    # RS results (5 runs)
    rs_fitnesses = [0.75, 0.78, 0.76, 0.77, 0.79]
    comparator.add_result("RS", problem_name, rs_fitnesses)
    
    # TPE results (5 runs)
    tpe_fitnesses = [0.88, 0.91, 0.89, 0.90, 0.92]
    comparator.add_result("TPE", problem_name, tpe_fitnesses)
    
    print("\nAdded results from 3 optimizers (5 runs each)")
    
    # Get statistics
    print("\nStatistical Analysis:")
    for opt_name in ["GA", "RS", "TPE"]:
        stats = comparator.get_statistics(opt_name, problem_name)
        print(f"\n  {opt_name}:")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Median: {stats['median']:.4f}")
        print(f"    Std: {stats['std']:.4f}")
        print(f"    Min/Max: {stats['min']:.4f} / {stats['max']:.4f}")
    
    # Print full comparison
    print("\n" + "-"*70)
    comparator.print_comparison()
    
    # Get best optimizer
    best = comparator.get_best_optimizer()
    print(f"\n🏆 Best overall optimizer: {best}")
    
    return comparator


def example_4_export_results():
    """Example 4: Export benchmark results."""
    print("\n" + "="*70)
    print("Example 4: Exporting Benchmark Results")
    print("="*70)
    
    suite = BenchmarkSuite()
    
    # Add optimizers
    suite.add_optimizer("GA", GeneticAlgorithm, {
        'population_size': 8,
        'num_generations': 8
    })
    
    suite.add_optimizer("RS", RandomSearch, {
        'num_samples': 64
    })
    
    # Run on simple problem
    problem = SimpleProblem()
    results = suite.run([problem], num_runs=2)
    
    print(f"\n✓ Benchmark complete: {len(results)} runs")
    
    # Export to JSON
    output_file = "benchmark_results.json"
    suite.export_results(output_file)
    
    print(f"\n✓ Results exported to: {output_file}")
    print("\nExported data includes:")
    print("  • Individual run results")
    print("  • Summary statistics")
    print("  • Optimizer configurations")
    
    return results


def example_5_convergence_analysis():
    """Example 5: Analyze convergence behavior."""
    print("\n" + "="*70)
    print("Example 5: Convergence Analysis")
    print("="*70)
    
    from morphml.benchmarks import ConvergenceAnalyzer
    
    analyzer = ConvergenceAnalyzer()
    
    # Simulate convergence histories
    ga_history = [0.5 + 0.04*i for i in range(20)]
    rs_history = [0.4 + 0.03*i for i in range(20)]
    
    analyzer.add_history("GA", ga_history)
    analyzer.add_history("RS", rs_history)
    
    print("\nAdded convergence histories for 2 optimizers")
    
    # Calculate AUC
    ga_auc = analyzer.calculate_auc("GA")
    rs_auc = analyzer.calculate_auc("RS")
    
    print(f"\nArea Under Curve (AUC):")
    print(f"  GA: {ga_auc:.2f}")
    print(f"  RS: {rs_auc:.2f}")
    print(f"  → Higher AUC indicates faster convergence")
    
    # Get mean convergence curves
    ga_curve = analyzer.get_mean_convergence("GA")
    print(f"\nMean convergence curve lengths:")
    print(f"  GA: {len(ga_curve)} generations")
    
    return analyzer


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MorphML Benchmark Suite Examples")
    print("="*70)
    print("\nDemonstrating optimizer comparison and benchmarking tools")
    
    # Run examples
    example_1_simple_comparison()
    example_2_multiple_problems()
    example_3_optimizer_comparator()
    example_4_export_results()
    example_5_convergence_analysis()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • BenchmarkSuite automates optimizer comparison")
    print("  • OptimizerComparator provides statistical analysis")
    print("  • ConvergenceAnalyzer examines optimization dynamics")
    print("  • Results can be exported for further analysis")
    print("  • Use benchmarks to select the best optimizer for your problem")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
