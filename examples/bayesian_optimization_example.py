"""
Bayesian Optimization Examples for MorphML

This example demonstrates all three Bayesian optimization algorithms:
1. Gaussian Process (GP) - Most sample-efficient, best for small-medium search
2. Tree-structured Parzen Estimator (TPE) - Scalable, good for large search
3. SMAC - Random Forest based, handles mixed spaces well

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
Repository: https://github.com/TIVerse/MorphML

Usage:
    python examples/bayesian_optimization_example.py
"""

from morphml.core.dsl import Layer, SearchSpace
from morphml.evaluation import HeuristicEvaluator
from morphml.optimizers.bayesian import (
    GaussianProcessOptimizer,
    SMACOptimizer,
    TPEOptimizer,
    optimize_with_gp,
    optimize_with_smac,
    optimize_with_tpe,
)


def create_search_space():
    """Create a CNN search space for demonstration."""
    space = SearchSpace("bayesian_demo")
    
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5]),
        Layer.relu(),
        Layer.batchnorm(),
        Layer.maxpool(pool_size=2),
        Layer.conv2d(filters=[64, 128, 256], kernel_size=3),
        Layer.relu(),
        Layer.maxpool(pool_size=2),
        Layer.dense(units=[128, 256, 512]),
        Layer.dropout(rate=[0.3, 0.5]),
        Layer.output(units=10),
    )
    
    return space


def example_1_gaussian_process():
    """Example 1: Gaussian Process optimization."""
    print("\n" + "="*70)
    print("Example 1: Gaussian Process Optimization")
    print("="*70)
    
    space = create_search_space()
    evaluator = HeuristicEvaluator()
    
    # Create GP optimizer
    gp = GaussianProcessOptimizer(
        search_space=space,
        config={
            'acquisition': 'ei',          # Expected Improvement
            'kernel': 'matern',           # Matern kernel
            'n_initial_points': 5,        # Quick demo
            'xi': 0.01,                   # Exploration parameter
        }
    )
    
    # Define callback for progress
    def progress_callback(iteration, best, history):
        if iteration % 5 == 0:
            print(f"  Iteration {iteration}: Best fitness = {best.fitness:.4f}")
    
    # Run optimization
    print("\nRunning GP optimization...")
    best = gp.optimize(
        evaluator=evaluator,
        max_evaluations=20,
        callback=progress_callback
    )
    
    print(f"\nFinal Results:")
    print(f"  Best fitness: {best.fitness:.4f}")
    print(f"  Architecture nodes: {len(best.graph.nodes)}")
    
    # Show GP statistics
    stats = gp.get_gp_statistics()
    print(f"\nGP Statistics:")
    print(f"  Observations: {stats['n_observations']}")
    print(f"  Log-likelihood: {stats['log_marginal_likelihood']:.2f}")
    print(f"  Mean fitness: {stats['mean_observed']:.4f}")
    
    return best


def example_2_tpe():
    """Example 2: Tree-structured Parzen Estimator."""
    print("\n" + "="*70)
    print("Example 2: TPE Optimization")
    print("="*70)
    
    space = create_search_space()
    evaluator = HeuristicEvaluator()
    
    # Quick TPE optimization using convenience function
    print("\nRunning TPE optimization...")
    best = optimize_with_tpe(
        search_space=space,
        evaluator=evaluator,
        n_iterations=20,
        n_initial=5,
        gamma=0.25,
        verbose=False  # We'll add custom progress
    )
    
    print(f"\nFinal Results:")
    print(f"  Best fitness: {best.fitness:.4f}")
    print(f"  Architecture nodes: {len(best.graph.nodes)}")
    
    return best


def example_3_smac():
    """Example 3: SMAC with Random Forest."""
    print("\n" + "="*70)
    print("Example 3: SMAC Optimization")
    print("="*70)
    
    space = create_search_space()
    evaluator = HeuristicEvaluator()
    
    # Create SMAC optimizer
    smac = SMACOptimizer(
        search_space=space,
        config={
            'n_estimators': 30,           # Number of trees
            'max_depth': 8,               # Tree depth
            'n_initial_points': 5,
        }
    )
    
    # Run optimization
    print("\nRunning SMAC optimization...")
    best = smac.optimize(evaluator, max_evaluations=20)
    
    print(f"\nFinal Results:")
    print(f"  Best fitness: {best.fitness:.4f}")
    print(f"  Architecture nodes: {len(best.graph.nodes)}")
    
    # Show Random Forest statistics
    rf_stats = smac.get_rf_statistics()
    print(f"\nRandom Forest Statistics:")
    print(f"  Trees: {rf_stats['n_estimators']}")
    print(f"  Avg tree depth: {rf_stats['avg_tree_depth']:.1f}")
    print(f"  Avg nodes per tree: {rf_stats['avg_tree_nodes']:.0f}")
    
    return best


def example_4_comparison():
    """Example 4: Compare all three optimizers."""
    print("\n" + "="*70)
    print("Example 4: Optimizer Comparison")
    print("="*70)
    
    space = create_search_space()
    evaluator = HeuristicEvaluator()
    
    # Define optimizers
    optimizers = {
        'GP': GaussianProcessOptimizer(
            space,
            {'acquisition': 'ei', 'n_initial_points': 5}
        ),
        'TPE': TPEOptimizer(
            space,
            {'gamma': 0.25, 'n_initial_points': 5}
        ),
        'SMAC': SMACOptimizer(
            space,
            {'n_estimators': 30, 'n_initial_points': 5}
        ),
    }
    
    results = {}
    
    # Run each optimizer
    for name, optimizer in optimizers.items():
        print(f"\nRunning {name}...")
        best = optimizer.optimize(evaluator, max_evaluations=20)
        results[name] = {
            'fitness': best.fitness,
            'nodes': len(best.graph.nodes),
            'history': optimizer.get_history()
        }
        print(f"  {name} best fitness: {best.fitness:.4f}")
    
    # Compare results
    print("\n" + "-"*70)
    print("Comparison Results:")
    print("-"*70)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Best fitness: {result['fitness']:.4f}")
        print(f"  Architecture nodes: {result['nodes']}")
        print(f"  Evaluations: {len(result['history'])}")
        
        # Compute sample efficiency
        target_fitness = 0.8
        evals_to_target = sum(
            1 for h in result['history'] if h['fitness'] < target_fitness
        )
        print(f"  Evaluations to reach {target_fitness:.2f}: {evals_to_target}")
    
    # Identify best optimizer
    best_optimizer = max(results.keys(), key=lambda k: results[k]['fitness'])
    print(f"\n🏆 Best Optimizer: {best_optimizer} "
          f"(fitness={results[best_optimizer]['fitness']:.4f})")


def example_5_advanced_gp():
    """Example 5: Advanced GP features."""
    print("\n" + "="*70)
    print("Example 5: Advanced GP Features")
    print("="*70)
    
    space = create_search_space()
    evaluator = HeuristicEvaluator()
    
    # Create GP with different acquisition functions
    acquisitions = ['ei', 'ucb', 'pi']
    
    print("\nTesting different acquisition functions:")
    
    for acq in acquisitions:
        print(f"\n  {acq.upper()}:")
        gp = GaussianProcessOptimizer(
            space,
            {'acquisition': acq, 'n_initial_points': 3}
        )
        
        best = gp.optimize(evaluator, max_evaluations=10)
        print(f"    Best fitness: {best.fitness:.4f}")
        
        # Get predictions on random samples
        test_graphs = [space.sample() for _ in range(5)]
        means, stds = gp.predict(test_graphs, return_std=True)
        
        print(f"    Prediction uncertainty (mean σ): {stds.mean():.4f}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MorphML Bayesian Optimization Examples")
    print("="*70)
    print("\nThis script demonstrates three Bayesian optimization algorithms:")
    print("1. Gaussian Process (GP) - Best for small-medium search")
    print("2. TPE - Scalable, good for large search spaces")
    print("3. SMAC - Random Forest based, handles mixed spaces")
    
    # Run examples
    example_1_gaussian_process()
    example_2_tpe()
    example_3_smac()
    example_4_comparison()
    example_5_advanced_gp()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • GP: Most sample-efficient, but slower for large n")
    print("  • TPE: Fast and scalable, good default choice")
    print("  • SMAC: Robust and handles mixed spaces well")
    print("\nRecommendation: Start with TPE for most problems")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
