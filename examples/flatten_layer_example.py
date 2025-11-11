"""Example demonstrating the flatten layer in MorphML.

The flatten layer is essential for transitioning from convolutional layers
to dense (fully connected) layers in neural networks. This example shows
various use cases and best practices.

Run:
    python examples/flatten_layer_example.py
"""

from morphml.core.dsl import Layer, SearchSpace
from morphml.core.graph import ModelGraph
from morphml.evaluation import HeuristicEvaluator
from morphml.optimizers import GeneticAlgorithm
from morphml.utils import ArchitectureExporter


def example_1_basic_cnn_with_flatten():
    """Basic CNN architecture using flatten layer."""
    print("\n" + "="*60)
    print("Example 1: Basic CNN with Flatten Layer")
    print("="*60)
    
    space = SearchSpace("basic_cnn")
    space.add_layers(
        # Input: 3-channel 32x32 image
        Layer.input(shape=(3, 32, 32)),
        
        # Convolutional feature extraction
        Layer.conv2d(filters=32, kernel_size=3, padding="same"),
        Layer.relu(),
        Layer.maxpool(pool_size=2),  # -> (32, 16, 16)
        
        Layer.conv2d(filters=64, kernel_size=3, padding="same"),
        Layer.relu(),
        Layer.maxpool(pool_size=2),  # -> (64, 8, 8)
        
        # Flatten before dense layers
        # Converts (64, 8, 8) -> (4096,)
        Layer.flatten(),
        
        # Dense classification head
        Layer.dense(units=128),
        Layer.relu(),
        Layer.dropout(rate=0.5),
        Layer.dense(units=10),
        Layer.softmax(),
    )
    
    # Sample an architecture
    graph = space.sample()
    
    print(f"✓ Created architecture with {len(graph.nodes)} layers")
    print(f"✓ Flatten layer converts spatial features to vector")
    
    # Export to see the flatten layer in action
    exporter = ArchitectureExporter()
    pytorch_code = exporter.to_pytorch(graph, "BasicCNN")
    
    print("\nGenerated PyTorch code snippet:")
    print("-" * 60)
    for line in pytorch_code.split('\n')[15:25]:  # Show relevant part
        print(line)
    print("-" * 60)
    
    return graph


def example_2_search_with_flatten():
    """Search for architectures with flatten layer."""
    print("\n" + "="*60)
    print("Example 2: NAS with Flatten Layer")
    print("="*60)
    
    # Define search space with variable architecture
    space = SearchSpace("nas_with_flatten")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        
        # Search over different conv configurations
        Layer.conv2d(filters=[32, 64], kernel_size=[3, 5]),
        Layer.relu(),
        Layer.maxpool(pool_size=2),
        
        Layer.conv2d(filters=[64, 128], kernel_size=[3, 5]),
        Layer.relu(),
        Layer.maxpool(pool_size=2),
        
        # Flatten is crucial here
        Layer.flatten(),
        
        # Search over dense layer sizes
        Layer.dense(units=[128, 256, 512]),
        Layer.relu(),
        Layer.dropout(rate=[0.3, 0.5, 0.7]),
        Layer.dense(units=10),
    )
    
    # Run quick search
    evaluator = HeuristicEvaluator()
    
    optimizer = GeneticAlgorithm(
        search_space=space,
        config={
            "population_size": 10,
            "num_generations": 5,
            "mutation_rate": 0.2,
            "crossover_rate": 0.8,
        }
    )
    
    print("Running NAS (5 generations, 10 population)...")
    best = optimizer.optimize(evaluator)
    
    print(f"\n✓ Best architecture found:")
    print(f"  - Fitness: {best.fitness:.4f}")
    print(f"  - Nodes: {len(best.graph.nodes)}")
    print(f"  - Parameters: ~{best.graph.estimate_parameters():,}")
    
    return best.graph


def example_3_without_flatten_error():
    """Demonstrate why flatten is needed."""
    print("\n" + "="*60)
    print("Example 3: Why Flatten is Necessary")
    print("="*60)
    
    # Architecture WITHOUT flatten (problematic)
    print("\n❌ Architecture WITHOUT flatten:")
    print("   Conv2D(64, 8, 8) -> Dense(128)")
    print("   Problem: Dense expects 1D input, but gets 3D!")
    print("   Shape mismatch: (64, 8, 8) vs (?,)")
    
    # Architecture WITH flatten (correct)
    print("\n✓ Architecture WITH flatten:")
    print("   Conv2D(64, 8, 8) -> Flatten() -> Dense(128)")
    print("   Flatten converts: (64, 8, 8) -> (4096,)")
    print("   Dense receives correct 1D input")
    
    # Create correct architecture
    space = SearchSpace("correct_architecture")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=64, kernel_size=3),
        Layer.maxpool(pool_size=4),  # -> (64, 8, 8)
        Layer.flatten(),              # -> (4096,)
        Layer.dense(units=128),       # Expects 1D input
    )
    
    graph = space.sample()
    
    # Show shape inference
    exporter = ArchitectureExporter()
    code = exporter.to_pytorch(graph)
    
    print("\nShape inference in generated code:")
    print("-" * 60)
    for line in code.split('\n'):
        if 'Flatten' in line or 'Linear' in line:
            print(line)
    print("-" * 60)


def example_4_multiple_flatten_patterns():
    """Different patterns using flatten."""
    print("\n" + "="*60)
    print("Example 4: Different Flatten Patterns")
    print("="*60)
    
    patterns = []
    
    # Pattern 1: Single flatten
    print("\n1. Single Flatten (Standard CNN):")
    space1 = SearchSpace("single_flatten")
    space1.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=32, kernel_size=3),
        Layer.maxpool(pool_size=2),
        Layer.conv2d(filters=64, kernel_size=3),
        Layer.maxpool(pool_size=2),
        Layer.flatten(),  # Single flatten before classification
        Layer.dense(units=128),
        Layer.dense(units=10),
    )
    patterns.append(("Single Flatten", space1.sample()))
    print("   Conv -> Pool -> Conv -> Pool -> Flatten -> Dense")
    
    # Pattern 2: Global average pooling (alternative to flatten)
    print("\n2. Global Average Pooling (Alternative):")
    space2 = SearchSpace("gap_pattern")
    space2.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=32, kernel_size=3),
        Layer.conv2d(filters=64, kernel_size=3),
        Layer.avgpool(pool_size=8),  # Reduce to (64, 1, 1)
        Layer.flatten(),              # Flatten to (64,)
        Layer.dense(units=10),
    )
    patterns.append(("Global Avg Pool + Flatten", space2.sample()))
    print("   Conv -> Conv -> GlobalAvgPool -> Flatten -> Dense")
    
    # Pattern 3: Early flatten (for 1D data)
    print("\n3. Early Flatten (1D Processing):")
    space3 = SearchSpace("early_flatten")
    space3.add_layers(
        Layer.input(shape=(100,)),  # Already 1D
        Layer.dense(units=128),
        Layer.relu(),
        Layer.dense(units=64),
        Layer.dense(units=10),
    )
    patterns.append(("No Flatten Needed (1D)", space3.sample()))
    print("   Dense -> Dense -> Dense (no flatten needed)")
    
    # Summary
    print("\n" + "-"*60)
    print("Summary:")
    for name, graph in patterns:
        has_flatten = any(n.operation == "flatten" for n in graph.nodes.values())
        print(f"  {name}: {'Has' if has_flatten else 'No'} flatten layer")


def example_5_export_comparison():
    """Compare exports with and without flatten."""
    print("\n" + "="*60)
    print("Example 5: Export Comparison")
    print("="*60)
    
    space = SearchSpace("export_example")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=64, kernel_size=3),
        Layer.maxpool(pool_size=2),
        Layer.flatten(),
        Layer.dense(units=128),
        Layer.dense(units=10),
    )
    
    graph = space.sample()
    exporter = ArchitectureExporter()
    
    # PyTorch export
    print("\nPyTorch Export:")
    print("-" * 60)
    pytorch = exporter.to_pytorch(graph, "MyModel")
    for line in pytorch.split('\n')[20:30]:
        if line.strip():
            print(line)
    
    # Keras export
    print("\nKeras Export:")
    print("-" * 60)
    keras = exporter.to_keras(graph, "my_model")
    for line in keras.split('\n')[10:20]:
        if line.strip():
            print(line)
    print("-" * 60)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" "*15 + "FLATTEN LAYER EXAMPLES")
    print("="*70)
    print("\nThe flatten layer is essential for connecting convolutional")
    print("and dense layers. These examples demonstrate its usage.")
    
    # Run examples
    example_1_basic_cnn_with_flatten()
    example_2_search_with_flatten()
    example_3_without_flatten_error()
    example_4_multiple_flatten_patterns()
    example_5_export_comparison()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. Use flatten when transitioning from Conv2D to Dense layers")
    print("  2. Flatten converts (C, H, W) -> (C*H*W,)")
    print("  3. Essential for CNN classification heads")
    print("  4. Shape inference helps generate correct code")
    print("  5. Alternative: Global Average Pooling + Flatten")
    print()


if __name__ == "__main__":
    main()
