"""
NSGA-II Multi-Objective Optimization Example

Demonstrates multi-objective architecture search optimizing for:
- Accuracy (maximize)
- Latency (minimize)
- Parameter count (minimize)

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
Repository: https://github.com/TIVerse/MorphML

Usage:
    python examples/nsga2_multiobjective_example.py
"""

import numpy as np

from morphml.core.dsl import Layer, SearchSpace
from morphml.optimizers.multi_objective import NSGA2Optimizer, optimize_with_nsga2
from morphml.visualization import plot_pareto_front_2d


def create_search_space():
    """Create a CNN search space."""
    space = SearchSpace("multiobjective_cnn")
    
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5]),
        Layer.relu(),
        Layer.batchnorm(),
        Layer.maxpool(pool_size=2),
        Layer.conv2d(filters=[64, 128, 256], kernel_size=3),
        Layer.relu(),
        Layer.maxpool(pool_size=2),
        Layer.flatten(),
        Layer.dense(units=[128, 256, 512]),
        Layer.dropout(rate=[0.3, 0.5]),
        Layer.output(units=10),
    )
    
    return space


def multi_objective_evaluator(graph):
    """
    Evaluate multiple objectives for an architecture.
    
    Returns:
        Dictionary with objective values
    """
    # Simulate accuracy (higher is better)
    num_nodes = len(graph.nodes)
    depth = graph.get_depth()
    accuracy = 0.6 + 0.04 * num_nodes - 0.001 * depth**2
    accuracy = min(1.0, max(0.0, accuracy))
    
    # Simulate latency in ms (lower is better)
    params = graph.estimate_parameters()
    latency = 10 + params / 100000  # Higher params = higher latency
    
    # Parameter count in millions (lower is better)
    param_millions = params / 1e6
    
    return {
        'accuracy': accuracy,
        'latency': latency,
        'params': param_millions
    }


def example_1_basic_nsga2():
    """Example 1: Basic NSGA-II optimization."""
    print("\n" + "="*70)
    print("Example 1: Basic NSGA-II Multi-Objective Optimization")
    print("="*70)
    
    space = create_search_space()
    
    # Define objectives
    objectives = [
        {'name': 'accuracy', 'maximize': True},
        {'name': 'latency', 'maximize': False},  # Minimize latency
        {'name': 'params', 'maximize': False},   # Minimize parameters
    ]
    
    # Create optimizer
    optimizer = NSGA2Optimizer(
        search_space=space,
        config={
            'population_size': 30,
            'num_generations': 20,
            'objectives': objectives
        }
    )
    
    print("\nRunning NSGA-II optimization...")
    print(f"  Population size: 30")
    print(f"  Generations: 20")
    print(f"  Objectives: {[obj['name'] for obj in objectives]}")
    
    # Run optimization
    pareto_front = optimizer.optimize(multi_objective_evaluator)
    
    print(f"\n✓ Optimization complete!")
    print(f"  Pareto front size: {len(pareto_front)} solutions")
    
    # Display some Pareto-optimal solutions
    print("\nTop 5 Pareto-optimal solutions:")
    for i, solution in enumerate(pareto_front[:5], 1):
        print(f"\n  Solution {i}:")
        print(f"    Accuracy: {solution.objectives['accuracy']:.4f}")
        print(f"    Latency: {solution.objectives['latency']:.2f} ms")
        print(f"    Params: {solution.objectives['params']:.2f}M")
        print(f"    Depth: {solution.genome.get_depth()}")
    
    return pareto_front


def example_2_convenience_function():
    """Example 2: Using convenience function."""
    print("\n" + "="*70)
    print("Example 2: NSGA-II with Convenience Function")
    print("="*70)
    
    space = create_search_space()
    
    print("\nUsing optimize_with_nsga2() convenience function...")
    
    # Quick optimization
    pareto_front = optimize_with_nsga2(
        search_space=space,
        evaluator=multi_objective_evaluator,
        objectives=[
            {'name': 'accuracy', 'maximize': True},
            {'name': 'latency', 'maximize': False},
        ],
        population_size=20,
        num_generations=15,
        verbose=True
    )
    
    print(f"\n✓ Found {len(pareto_front)} Pareto-optimal solutions")
    
    return pareto_front


def example_3_visualize_pareto():
    """Example 3: Visualize Pareto front."""
    print("\n" + "="*70)
    print("Example 3: Visualizing Pareto Front")
    print("="*70)
    
    space = create_search_space()
    
    # Run optimization
    pareto_front = optimize_with_nsga2(
        space,
        multi_objective_evaluator,
        objectives=[
            {'name': 'accuracy', 'maximize': True},
            {'name': 'latency', 'maximize': False},
        ],
        population_size=25,
        num_generations=15,
        verbose=False
    )
    
    print(f"\n✓ Optimization complete: {len(pareto_front)} solutions")
    print("\nGenerating 2D Pareto front visualization...")
    
    # Visualize accuracy vs latency trade-off
    try:
        plot_pareto_front_2d(
            pareto_front,
            objective_names=['accuracy', 'latency'],
            save_path='pareto_front_accuracy_latency.png',
            title='Accuracy vs Latency Trade-off'
        )
        print("  ✓ Saved to: pareto_front_accuracy_latency.png")
    except Exception as e:
        print(f"  ! Visualization skipped: {e}")


def example_4_analyze_tradeoffs():
    """Example 4: Analyze objective trade-offs."""
    print("\n" + "="*70)
    print("Example 4: Analyzing Objective Trade-offs")
    print("="*70)
    
    space = create_search_space()
    
    # Run optimization
    pareto_front = optimize_with_nsga2(
        space,
        multi_objective_evaluator,
        objectives=[
            {'name': 'accuracy', 'maximize': True},
            {'name': 'latency', 'maximize': False},
            {'name': 'params', 'maximize': False},
        ],
        population_size=30,
        num_generations=20,
        verbose=False
    )
    
    print(f"\n✓ Found {len(pareto_front)} Pareto-optimal solutions")
    
    # Analyze trade-offs
    accuracies = [s.objectives['accuracy'] for s in pareto_front]
    latencies = [s.objectives['latency'] for s in pareto_front]
    params = [s.objectives['params'] for s in pareto_front]
    
    print("\nObjective Statistics:")
    print(f"  Accuracy: min={min(accuracies):.4f}, max={max(accuracies):.4f}, mean={np.mean(accuracies):.4f}")
    print(f"  Latency: min={min(latencies):.2f}, max={max(latencies):.2f}, mean={np.mean(latencies):.2f} ms")
    print(f"  Params: min={min(params):.2f}, max={max(params):.2f}, mean={np.mean(params):.2f}M")
    
    # Find specific solutions
    print("\nInteresting Solutions:")
    
    # Best accuracy
    best_acc_idx = np.argmax(accuracies)
    print(f"\n  Best Accuracy Solution:")
    print(f"    Accuracy: {pareto_front[best_acc_idx].objectives['accuracy']:.4f}")
    print(f"    Latency: {pareto_front[best_acc_idx].objectives['latency']:.2f} ms")
    print(f"    Params: {pareto_front[best_acc_idx].objectives['params']:.2f}M")
    
    # Fastest (lowest latency)
    fastest_idx = np.argmin(latencies)
    print(f"\n  Fastest Solution:")
    print(f"    Accuracy: {pareto_front[fastest_idx].objectives['accuracy']:.4f}")
    print(f"    Latency: {pareto_front[fastest_idx].objectives['latency']:.2f} ms")
    print(f"    Params: {pareto_front[fastest_idx].objectives['params']:.2f}M")
    
    # Smallest model
    smallest_idx = np.argmin(params)
    print(f"\n  Smallest Model:")
    print(f"    Accuracy: {pareto_front[smallest_idx].objectives['accuracy']:.4f}")
    print(f"    Latency: {pareto_front[smallest_idx].objectives['latency']:.2f} ms")
    print(f"    Params: {pareto_front[smallest_idx].objectives['params']:.2f}M")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MorphML NSGA-II Multi-Objective Optimization Examples")
    print("="*70)
    print("\nDemonstrating multi-objective architecture search with NSGA-II")
    print("Optimizing for accuracy, latency, and parameter count")
    
    # Run examples
    example_1_basic_nsga2()
    example_2_convenience_function()
    example_3_visualize_pareto()
    example_4_analyze_tradeoffs()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • NSGA-II finds a set of Pareto-optimal solutions")
    print("  • Each solution represents a different trade-off")
    print("  • Users can select based on their preferences")
    print("  • Visualization helps understand trade-offs")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
