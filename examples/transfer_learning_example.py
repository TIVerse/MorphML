"""
Transfer Learning Example for MorphML

Demonstrates how to transfer architectures across related tasks
and use fine-tuning strategies.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from morphml.core.dsl import SearchSpace, Layer
from morphml.core.graph import ModelGraph
from morphml.meta_learning.transfer import (
    ArchitectureTransfer,
    FineTuningStrategy,
    MultiTaskNAS,
)
from morphml.meta_learning.experiment_database import TaskMetadata
from morphml.optimizers.genetic_algorithm import GeneticAlgorithm
from morphml.evaluation.heuristic import HeuristicEvaluator
from morphml.execution.local_executor import LocalExecutor
from morphml.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def example_1_basic_transfer():
    """Example 1: Basic architecture transfer between CIFAR-10 and CIFAR-100."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 1: Transfer CIFAR-10 → CIFAR-100")
    logger.info("=" * 70)
    
    # Define source task (CIFAR-10)
    source_task = TaskMetadata(
        task_id="cifar10_exp",
        dataset_name="CIFAR-10",
        num_classes=10,
        input_size=(3, 32, 32),
        num_samples=50000,
        problem_type="image_classification",
    )
    
    # Define target task (CIFAR-100)
    target_task = TaskMetadata(
        task_id="cifar100_exp",
        dataset_name="CIFAR-100",
        num_classes=100,
        input_size=(3, 32, 32),
        num_samples=50000,
        problem_type="image_classification",
    )
    
    # Create a sample architecture (pretend this is the best from CIFAR-10)
    from morphml.core.graph import ModelGraph, GraphNode
    
    source_arch = ModelGraph()
    
    # Build simple CNN
    input_node = GraphNode("input", "input", {"input_shape": (3, 32, 32)})
    conv1 = GraphNode("conv1", "conv2d", {"filters": 64, "kernel_size": 3})
    pool1 = GraphNode("pool1", "maxpool", {"pool_size": 2})
    conv2 = GraphNode("conv2", "conv2d", {"filters": 128, "kernel_size": 3})
    pool2 = GraphNode("pool2", "maxpool", {"pool_size": 2})
    dense1 = GraphNode("dense1", "dense", {"units": 256})
    dropout = GraphNode("dropout", "dropout", {"rate": 0.5})
    output_node = GraphNode("output", "dense", {"units": 10})  # 10 classes
    
    source_arch.add_node(input_node)
    source_arch.add_node(conv1)
    source_arch.add_node(pool1)
    source_arch.add_node(conv2)
    source_arch.add_node(pool2)
    source_arch.add_node(dense1)
    source_arch.add_node(dropout)
    source_arch.add_node(output_node)
    
    source_arch.add_edge_by_id("input", "conv1")
    source_arch.add_edge_by_id("conv1", "pool1")
    source_arch.add_edge_by_id("pool1", "conv2")
    source_arch.add_edge_by_id("conv2", "pool2")
    source_arch.add_edge_by_id("pool2", "dense1")
    source_arch.add_edge_by_id("dense1", "dropout")
    source_arch.add_edge_by_id("dropout", "output")
    
    logger.info(f"Source architecture: {len(source_arch.nodes)} nodes")
    logger.info(f"Output layer classes: {source_arch.nodes['output'].params['units']}")
    
    # Evaluate transferability
    transferability = ArchitectureTransfer.evaluate_transferability(
        source_task, target_task
    )
    logger.info(f"\nTransferability score: {transferability:.3f}")
    
    # Get recommended strategy
    recommendation = ArchitectureTransfer.recommend_transfer_strategy(
        source_task, target_task
    )
    logger.info(f"Recommended strategy: {recommendation['strategy']}")
    logger.info(f"Capacity scale: {recommendation['capacity_scale']}")
    logger.info(f"Reasoning: {recommendation['reasoning']}")
    
    # Transfer architecture
    transferred = ArchitectureTransfer.transfer_architecture(
        source_arch=source_arch,
        source_task=source_task,
        target_task=target_task,
        adaptation_strategy=recommendation["strategy"],
        capacity_scale=recommendation["capacity_scale"],
    )
    
    logger.info(f"\nTransferred architecture: {len(transferred.nodes)} nodes")
    logger.info(f"Output layer classes: {transferred.nodes['output'].params['units']}")
    
    logger.info("\n✓ Transfer successful!")
    
    return transferred


def example_2_fine_tuning_strategy():
    """Example 2: Get fine-tuning recommendations."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 2: Fine-Tuning Strategy Selection")
    logger.info("=" * 70)
    
    scenarios = [
        ("same_domain", 20, 50000, "Similar datasets, large data"),
        ("similar_tasks", 15, 5000, "Related tasks, small data"),
        ("distant_tasks", 30, 100000, "Different domains, large data"),
    ]
    
    for transfer_type, depth, dataset_size, description in scenarios:
        logger.info(f"\nScenario: {description}")
        logger.info(f"  Transfer type: {transfer_type}")
        logger.info(f"  Model depth: {depth} layers")
        logger.info(f"  Dataset size: {dataset_size:,} samples")
        
        strategy = FineTuningStrategy.get_strategy(
            transfer_type=transfer_type,
            model_depth=depth,
            target_dataset_size=dataset_size,
        )
        
        logger.info(f"\nRecommended strategy:")
        logger.info(f"  Method: {strategy['method']}")
        if "freeze_ratio" in strategy:
            logger.info(f"  Freeze ratio: {strategy['freeze_ratio']:.1%}")
        logger.info(f"  Learning rate: {strategy['learning_rate']}")
        logger.info(f"  Epochs: {strategy['num_epochs']}")
        logger.info(f"  Description: {strategy['description']}")
        
        # Generate freeze mask
        if strategy["method"] == "freeze_early":
            freeze_mask = FineTuningStrategy.generate_freeze_mask(
                num_layers=depth,
                freeze_ratio=strategy["freeze_ratio"],
            )
            num_frozen = sum(freeze_mask)
            logger.info(f"  Frozen layers: {num_frozen}/{depth}")


def example_3_multi_task_nas():
    """Example 3: Multi-task NAS across related datasets."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 3: Multi-Task NAS")
    logger.info("=" * 70)
    
    # Define multiple related tasks
    tasks = [
        TaskMetadata(
            task_id="cifar10",
            dataset_name="CIFAR-10",
            num_classes=10,
            input_size=(3, 32, 32),
            num_samples=50000,
            problem_type="image_classification",
        ),
        TaskMetadata(
            task_id="cifar100",
            dataset_name="CIFAR-100",
            num_classes=100,
            input_size=(3, 32, 32),
            num_samples=50000,
            problem_type="image_classification",
        ),
        TaskMetadata(
            task_id="svhn",
            dataset_name="SVHN",
            num_classes=10,
            input_size=(3, 32, 32),
            num_samples=73257,
            problem_type="image_classification",
        ),
    ]
    
    # Create search space
    search_space = SearchSpace(
        name="multi_task_space",
        layers=[
            Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5]),
            Layer.maxpool(pool_size=[2]),
            Layer.dense(units=[128, 256]),
            Layer.dropout(rate=[0.3, 0.5]),
        ],
    )
    
    # Initialize multi-task NAS
    task_weights = [0.5, 0.3, 0.2]  # Prioritize CIFAR-10
    
    multi_nas = MultiTaskNAS(
        tasks=tasks,
        search_space=search_space,
        task_weights=task_weights,
    )
    
    logger.info(f"Configured for {len(tasks)} tasks:")
    for task, weight in zip(tasks, task_weights):
        logger.info(f"  - {task.dataset_name}: weight={weight:.2f}")
    
    # Sample an architecture
    sample_arch = search_space.sample()
    
    # Create a simple evaluator
    evaluator = HeuristicEvaluator()
    
    # Evaluate on all tasks
    logger.info("\nEvaluating sample architecture on all tasks...")
    
    results = multi_nas.evaluate_multi_task_fitness(
        architecture=sample_arch,
        evaluator_fn=lambda arch, task: evaluator(arch),
    )
    
    logger.info("\nPer-task fitness:")
    for task_id, fitness in results.items():
        if task_id != "weighted_average":
            logger.info(f"  {task_id}: {fitness:.4f}")
    
    logger.info(f"\nWeighted average fitness: {results['weighted_average']:.4f}")
    
    logger.info("\n✓ Multi-task evaluation complete!")


def example_4_progressive_transfer():
    """Example 4: Progressive transfer across task sequence."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 4: Progressive Transfer Learning")
    logger.info("=" * 70)
    
    # Task sequence: MNIST → Fashion-MNIST → CIFAR-10
    task_sequence = [
        TaskMetadata(
            task_id="mnist",
            dataset_name="MNIST",
            num_classes=10,
            input_size=(1, 28, 28),
            num_samples=60000,
            problem_type="image_classification",
        ),
        TaskMetadata(
            task_id="fashion_mnist",
            dataset_name="Fashion-MNIST",
            num_classes=10,
            input_size=(1, 28, 28),
            num_samples=60000,
            problem_type="image_classification",
        ),
        TaskMetadata(
            task_id="cifar10",
            dataset_name="CIFAR-10",
            num_classes=10,
            input_size=(3, 32, 32),
            num_samples=50000,
            problem_type="image_classification",
        ),
    ]
    
    logger.info("Task sequence:")
    for i, task in enumerate(task_sequence):
        logger.info(f"  {i+1}. {task.dataset_name}")
    
    # Simulate progressive transfer
    logger.info("\nProgressive transfer simulation:")
    
    for i in range(len(task_sequence) - 1):
        source = task_sequence[i]
        target = task_sequence[i + 1]
        
        logger.info(f"\nStep {i+1}: {source.dataset_name} → {target.dataset_name}")
        
        # Evaluate transferability
        score = ArchitectureTransfer.evaluate_transferability(source, target)
        logger.info(f"  Transferability: {score:.3f}")
        
        # Get strategy
        strategy = ArchitectureTransfer.recommend_transfer_strategy(source, target)
        logger.info(f"  Strategy: {strategy['strategy']}")
        logger.info(f"  Reasoning: {strategy['reasoning']}")


def main():
    """Run all examples."""
    setup_logging(verbose=True)
    
    logger.info("\n" + "=" * 70)
    logger.info("MorphML Transfer Learning Examples")
    logger.info("=" * 70)
    
    # Run examples
    example_1_basic_transfer()
    example_2_fine_tuning_strategy()
    example_3_multi_task_nas()
    example_4_progressive_transfer()
    
    logger.info("\n" + "=" * 70)
    logger.info("All examples completed! ✓")
    logger.info("=" * 70)
    
    logger.info("\nKey Takeaways:")
    logger.info("  1. Use ArchitectureTransfer for adapting models to new tasks")
    logger.info("  2. Evaluate transferability before transfer")
    logger.info("  3. Use recommended fine-tuning strategies")
    logger.info("  4. Multi-task NAS can find generalizable architectures")
    logger.info("  5. Progressive transfer through related tasks improves results")


if __name__ == "__main__":
    main()
