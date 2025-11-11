"""Advanced features example showcasing P1, P2, and P3 enhancements.

This example demonstrates:
- Flatten layer usage (P1)
- Enhanced constraint violation messages (P2)
- Adaptive crossover rates (P3)
- Crossover visualization (P3)
- Custom layer handlers (P3)

Run:
    python examples/advanced_features_example.py
"""

from morphml.core.dsl import Layer, SearchSpace
from morphml.constraints import ConstraintHandler, MaxParametersConstraint, DepthConstraint
from morphml.evaluation import HeuristicEvaluator
from morphml.optimizers import GeneticAlgorithm
from morphml.optimizers.adaptive_operators import AdaptiveOperatorScheduler
from morphml.utils import ArchitectureExporter
from morphml.visualization.crossover_viz import quick_crossover_viz


def demo_enhanced_constraints():
    """Demonstrate enhanced constraint violation messages (P2)."""
    print("\n" + "="*70)
    print("DEMO 1: Enhanced Constraint Violation Messages")
    print("="*70)
    
    # Create a search space
    space = SearchSpace("constraint_demo")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=256, kernel_size=3),  # Large filters
        Layer.conv2d(filters=512, kernel_size=3),  # Very large
        Layer.flatten(),
        Layer.dense(units=1024),  # Large dense layer
        Layer.dense(units=10),
    )
    
    # Set up strict constraints
    handler = ConstraintHandler()
    handler.add_constraint(MaxParametersConstraint(max_params=100000))
    handler.add_constraint(DepthConstraint(min_depth=8, max_depth=15))
    
    # Sample and check
    graph = space.sample()
    
    print(f"\nArchitecture stats:")
    print(f"  - Nodes: {len(graph.nodes)}")
    print(f"  - Parameters: {graph.estimate_parameters():,}")
    print(f"  - Depth: {graph.depth()}")
    
    # Check constraints
    if not handler.check(graph):
        print("\n❌ Constraint violations detected!\n")
        print(handler.format_violations(graph))
    else:
        print("\n✓ All constraints satisfied!")
    
    # Show detailed violations
    violations = handler.get_detailed_violations(graph)
    if violations:
        print("\nDetailed violation data:")
        for v in violations:
            print(f"  - {v['name']}: penalty={v['penalty']:.4f}")


def demo_adaptive_crossover():
    """Demonstrate adaptive crossover rates (P3)."""
    print("\n" + "="*70)
    print("DEMO 2: Adaptive Crossover Rates")
    print("="*70)
    
    from morphml.core.search.population import Population
    from morphml.core.search.individual import Individual
    
    # Create search space
    space = SearchSpace("adaptive_demo")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=[32, 64], kernel_size=3),
        Layer.relu(),
        Layer.maxpool(pool_size=2),
        Layer.flatten(),
        Layer.dense(units=[128, 256]),
        Layer.dense(units=10),
    )
    
    # Create population
    population = Population(max_size=20)
    for i in range(20):
        graph = space.sample()
        ind = Individual(graph)
        ind.set_fitness(0.5 + i * 0.02)  # Varying fitness
        population.add(ind)
    
    # Create adaptive scheduler
    scheduler = AdaptiveOperatorScheduler(
        initial_crossover=0.8,
        initial_mutation=0.2,
    )
    
    print("\nSimulating 10 generations with adaptive rates:")
    print("-" * 70)
    print(f"{'Gen':<5} {'Crossover':<12} {'Mutation':<12} {'Diversity':<15}")
    print("-" * 70)
    
    for gen in range(10):
        best_fitness = max(ind.fitness for ind in population.individuals.values())
        crossover_rate, mutation_rate = scheduler.get_rates(
            population, best_fitness, gen
        )
        
        stats = scheduler.get_statistics()
        trend = stats['diversity_trend']
        
        print(f"{gen:<5} {crossover_rate:<12.3f} {mutation_rate:<12.3f} {trend:<15}")
        
        # Simulate fitness changes
        for ind in population.individuals.values():
            ind.set_fitness(ind.fitness + 0.01)
    
    print("-" * 70)
    print("\n✓ Rates adapted based on population diversity and progress!")


def demo_crossover_visualization():
    """Demonstrate crossover visualization (P3)."""
    print("\n" + "="*70)
    print("DEMO 3: Crossover Visualization")
    print("="*70)
    
    # Create two parent architectures
    space = SearchSpace("viz_demo")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=32, kernel_size=3),
        Layer.relu(),
        Layer.maxpool(pool_size=2),
        Layer.conv2d(filters=64, kernel_size=3),
        Layer.relu(),
        Layer.flatten(),
        Layer.dense(units=128),
        Layer.dense(units=10),
    )
    
    parent1 = space.sample()
    parent2 = space.sample()
    
    print(f"\nParent 1: {len(parent1.nodes)} nodes")
    print(f"Parent 2: {len(parent2.nodes)} nodes")
    
    # Visualize crossover
    try:
        output_file = "/tmp/morphml_crossover_demo.png"
        quick_crossover_viz(parent1, parent2, output_file)
        print(f"\n✓ Crossover visualization saved to: {output_file}")
        print("  Open the file to see parent and offspring architectures!")
    except Exception as e:
        print(f"\n⚠ Visualization requires matplotlib: {e}")
        print("  Install with: pip install matplotlib")


def demo_custom_layer_handler():
    """Demonstrate custom layer handlers (P3)."""
    print("\n" + "="*70)
    print("DEMO 4: Custom Layer Handlers")
    print("="*70)
    
    # Create exporter with custom handler
    exporter = ArchitectureExporter()
    
    # Define custom layer handler
    def attention_handler(node, shapes):
        """Handler for custom attention layer."""
        heads = node.params.get("heads", 8)
        dim = node.params.get("dim", 64)
        return f"nn.MultiheadAttention(embed_dim={dim}, num_heads={heads})"
    
    # Register handler
    exporter.add_custom_layer_handler(
        "attention",
        pytorch_handler=attention_handler
    )
    
    print("\n✓ Registered custom handler for 'attention' layer")
    print("  Handler generates: nn.MultiheadAttention(...)")
    
    # Note: To actually use this, you would need to:
    # 1. Add Layer.custom() method to DSL
    # 2. Create graph with custom layer
    # 3. Export with custom handler
    
    print("\nUsage pattern:")
    print("  1. Define custom layer in search space")
    print("  2. Register handler with exporter")
    print("  3. Export generates custom code")


def demo_shape_inference():
    """Demonstrate improved shape inference (P2)."""
    print("\n" + "="*70)
    print("DEMO 5: Shape Inference in Export")
    print("="*70)
    
    # Create architecture
    space = SearchSpace("shape_demo")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=64, kernel_size=3),
        Layer.batchnorm(),
        Layer.relu(),
        Layer.maxpool(pool_size=2),
        Layer.flatten(),
        Layer.dense(units=128),
        Layer.dense(units=10),
    )
    
    graph = space.sample()
    exporter = ArchitectureExporter()
    
    print("\nGenerated PyTorch code with shape inference:")
    print("-" * 70)
    
    code = exporter.to_pytorch(graph, "InferredModel")
    
    # Show relevant lines
    for line in code.split('\n'):
        if 'Conv2d' in line or 'BatchNorm' in line or 'Linear' in line:
            print(line)
    
    print("-" * 70)
    print("\n✓ Notice: Dimensions are inferred automatically!")
    print("  - Conv2d: in_channels inferred from input")
    print("  - BatchNorm: num_features inferred from conv")
    print("  - Linear: in_features calculated from flatten")


def demo_complete_workflow():
    """Demonstrate complete workflow with all features."""
    print("\n" + "="*70)
    print("DEMO 6: Complete Workflow")
    print("="*70)
    
    # 1. Define search space with flatten
    space = SearchSpace("complete_workflow")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=[32, 64], kernel_size=[3, 5]),
        Layer.relu(),
        Layer.maxpool(pool_size=2),
        Layer.conv2d(filters=[64, 128], kernel_size=3),
        Layer.relu(),
        Layer.maxpool(pool_size=2),
        Layer.flatten(),  # P1: Flatten layer
        Layer.dense(units=[128, 256]),
        Layer.relu(),
        Layer.dropout(rate=[0.3, 0.5]),
        Layer.dense(units=10),
    )
    
    # 2. Set up constraints with enhanced messages (P2)
    handler = ConstraintHandler()
    handler.add_constraint(MaxParametersConstraint(max_params=500000))
    handler.add_constraint(DepthConstraint(min_depth=5, max_depth=15))
    
    # 3. Create evaluator
    evaluator = HeuristicEvaluator()
    
    # 4. Run optimization with adaptive operators (P3)
    print("\nRunning NAS with adaptive operators...")
    
    optimizer = GeneticAlgorithm(
        search_space=space,
        config={
            "population_size": 10,
            "num_generations": 5,
            "mutation_rate": 0.2,
            "crossover_rate": 0.8,
        }
    )
    
    best = optimizer.optimize(evaluator)
    
    # 5. Check constraints
    print(f"\n✓ Best architecture found:")
    print(f"  - Fitness: {best.fitness:.4f}")
    print(f"  - Nodes: {len(best.graph.nodes)}")
    print(f"  - Parameters: {best.graph.estimate_parameters():,}")
    
    if not handler.check(best.graph):
        print("\n⚠ Constraint violations:")
        print(handler.format_violations(best.graph))
    else:
        print("\n✓ All constraints satisfied!")
    
    # 6. Export with shape inference (P2)
    exporter = ArchitectureExporter()
    code = exporter.to_pytorch(best.graph, "BestModel")
    
    print("\nExported code preview:")
    print("-" * 70)
    for i, line in enumerate(code.split('\n')[15:25]):
        print(line)
    print("-" * 70)
    
    print("\n✓ Complete workflow executed successfully!")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print(" "*10 + "MORPHML ADVANCED FEATURES DEMONSTRATION")
    print("="*70)
    print("\nShowcasing enhancements from P1, P2, and P3 phases:")
    print("  P1: Core fixes (flatten layer, crossover)")
    print("  P2: Medium priority (constraints, shape inference)")
    print("  P3: Enhancements (adaptive rates, visualization, custom layers)")
    
    try:
        demo_enhanced_constraints()
        demo_adaptive_crossover()
        demo_crossover_visualization()
        demo_custom_layer_handler()
        demo_shape_inference()
        demo_complete_workflow()
        
        print("\n" + "="*70)
        print("All demos completed successfully!")
        print("="*70)
        print("\nKey Features Demonstrated:")
        print("  ✓ Enhanced constraint violation messages")
        print("  ✓ Adaptive crossover and mutation rates")
        print("  ✓ Crossover visualization")
        print("  ✓ Custom layer handlers")
        print("  ✓ Automatic shape inference")
        print("  ✓ Complete NAS workflow")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
