#!/usr/bin/env python3
"""Example: Warm-starting NAS from past experiments.

Demonstrates how to accelerate search using transfer learning.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from morphml.core.dsl import Layer, SearchSpace
from morphml.meta_learning import (
    ExperimentDatabase,
    TaskMetadata,
    WarmStarter,
)
from morphml.optimizers import GeneticAlgorithm
from morphml.logging_config import get_logger

logger = get_logger(__name__)


def create_sample_knowledge_base():
    """Create sample knowledge base with past experiments."""
    logger.info("Creating sample knowledge base...")
    
    db = ExperimentDatabase()
    
    # Past Task 1: CIFAR-10
    task1 = TaskMetadata(
        task_id="cifar10_exp",
        dataset_name="CIFAR-10",
        num_samples=50000,
        num_classes=10,
        input_size=(3, 32, 32),
        problem_type="classification",
        metadata={"best_accuracy": 0.92},
    )
    db.add_task(task1)
    
    # Add some good architectures from CIFAR-10
    space1 = SearchSpace("cifar10")
    space1.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=64, kernel_size=3),
        Layer.conv2d(filters=128, kernel_size=3),
        Layer.maxpool2d(pool_size=2),
        Layer.conv2d(filters=256, kernel_size=3),
        Layer.flatten(),
        Layer.dense(units=256),
        Layer.dropout(rate=0.5),
        Layer.output(units=10),
    )
    
    for i in range(10):
        graph = space1.sample()
        db.add_architecture("cifar10_exp", graph, fitness=0.85 + i * 0.01)
    
    # Past Task 2: SVHN
    task2 = TaskMetadata(
        task_id="svhn_exp",
        dataset_name="SVHN",
        num_samples=73257,
        num_classes=10,
        input_size=(3, 32, 32),
        problem_type="classification",
        metadata={"best_accuracy": 0.94},
    )
    db.add_task(task2)
    
    for i in range(10):
        graph = space1.sample()
        db.add_architecture("svhn_exp", graph, fitness=0.88 + i * 0.01)
    
    logger.info(f"Created knowledge base with {len(db.get_all_tasks())} past tasks")
    
    return db


def run_baseline_search(search_space, evaluator, population_size=30):
    """Run baseline search without warm-starting."""
    logger.info("\n" + "="*80)
    logger.info("BASELINE: Random Initialization")
    logger.info("="*80)
    
    optimizer = GeneticAlgorithm(
        search_space=search_space,
        evaluator=evaluator,
        population_size=population_size,
        num_generations=10,
        mutation_prob=0.2,
        crossover_prob=0.8,
    )
    
    best = optimizer.search()
    
    logger.info(f"\nBaseline Best Fitness: {best.fitness:.4f}")
    logger.info(f"Total Evaluations: {len(optimizer.history)}")
    
    return optimizer.history


def run_warm_started_search(
    search_space, evaluator, knowledge_base, current_task, population_size=30
):
    """Run search with warm-starting."""
    logger.info("\n" + "="*80)
    logger.info("WARM-STARTED: Transfer from Past Experiments")
    logger.info("="*80)
    
    # Create warm-starter
    warm_starter = WarmStarter(
        knowledge_base,
        config={
            "transfer_ratio": 0.5,  # 50% from past, 50% random
            "min_similarity": 0.6,
            "similarity_method": "meta_features",
        },
    )
    
    # Generate warm-started population
    initial_population = warm_starter.generate_initial_population(
        current_task=current_task,
        population_size=population_size,
        search_space=search_space,
    )
    
    logger.info(f"Generated initial population of {len(initial_population)} architectures")
    
    # Run optimizer with warm-started population
    optimizer = GeneticAlgorithm(
        search_space=search_space,
        evaluator=evaluator,
        population_size=population_size,
        num_generations=10,
        mutation_prob=0.2,
        crossover_prob=0.8,
    )
    
    # Set initial population
    optimizer.population = [
        type('Individual', (), {'architecture': g, 'fitness': None})()
        for g in initial_population
    ]
    
    best = optimizer.search()
    
    logger.info(f"\nWarm-Started Best Fitness: {best.fitness:.4f}")
    logger.info(f"Total Evaluations: {len(optimizer.history)}")
    
    return optimizer.history


def main():
    """Run warm-starting example."""
    print("\n" + "🔥"*40)
    print(" "*25 + "Warm-Starting Example")
    print("🔥"*40 + "\n")
    
    # Create knowledge base
    kb = create_sample_knowledge_base()
    
    # Define current task (similar to CIFAR-10)
    current_task = TaskMetadata(
        task_id="cifar100_exp",
        dataset_name="CIFAR-100",
        num_samples=50000,
        num_classes=100,
        input_size=(3, 32, 32),
        problem_type="classification",
    )
    
    logger.info(f"\nCurrent Task: {current_task.dataset_name}")
    logger.info(f"  Samples: {current_task.num_samples}")
    logger.info(f"  Classes: {current_task.num_classes}")
    logger.info(f"  Input Size: {current_task.input_size}")
    
    # Define search space
    search_space = SearchSpace("cifar100")
    search_space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=64, kernel_size=3),
        Layer.conv2d(filters=128, kernel_size=3),
        Layer.maxpool2d(pool_size=2),
        Layer.conv2d(filters=256, kernel_size=3),
        Layer.flatten(),
        Layer.dense(units=512),
        Layer.dropout(rate=0.5),
        Layer.output(units=100),
    )
    
    # Simple evaluator (layer count + diversity)
    def evaluator(graph):
        """Simple fitness function."""
        # Prefer deeper networks
        depth_score = len(graph.layers) / 20.0
        
        # Prefer diverse operations
        ops = [layer.layer_type for layer in graph.layers]
        diversity_score = len(set(ops)) / 10.0
        
        # Combined score
        fitness = 0.6 * depth_score + 0.4 * diversity_score
        
        return min(fitness, 1.0)
    
    # Run baseline
    baseline_history = run_baseline_search(
        search_space, evaluator, population_size=30
    )
    
    # Run warm-started
    warm_started_history = run_warm_started_search(
        search_space, evaluator, kb, current_task, population_size=30
    )
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    baseline_best = max(baseline_history, key=lambda x: x.fitness)
    warm_started_best = max(warm_started_history, key=lambda x: x.fitness)
    
    print(f"\nBaseline:")
    print(f"  Best Fitness: {baseline_best.fitness:.4f}")
    print(f"  Evaluations: {len(baseline_history)}")
    
    print(f"\nWarm-Started:")
    print(f"  Best Fitness: {warm_started_best.fitness:.4f}")
    print(f"  Evaluations: {len(warm_started_history)}")
    
    improvement = (
        (warm_started_best.fitness - baseline_best.fitness) / baseline_best.fitness * 100
    )
    print(f"\nImprovement: {improvement:+.1f}%")
    
    print("\n" + "="*80)
    print("✅ Warm-starting example complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
