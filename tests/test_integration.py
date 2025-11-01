"""Comprehensive integration tests for MorphML Phase 1."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from morphml.core.dsl import Layer, SearchSpace
from morphml.core.graph import GraphEdge, GraphNode, ModelGraph
from morphml.core.search import Individual, Population
from morphml.evaluation import HeuristicEvaluator
from morphml.optimizers import (
    DifferentialEvolution,
    GeneticAlgorithm,
    HillClimbing,
    RandomSearch,
    SimulatedAnnealing,
)
from morphml.utils import ArchitectureExporter, Checkpoint


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_full_experiment_workflow(self) -> None:
        """
        Integration test: complete workflow from search space to results.
        
        This test validates:
        1. Search space definition
        2. Optimizer initialization
        3. Population evolution
        4. Architecture evaluation
        5. Result export
        """
        # Define search space
        space = SearchSpace("integration_test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64], kernel_size=3),
            Layer.relu(),
            Layer.maxpool(pool_size=2),
            Layer.dense(units=[128, 256]),
            Layer.dropout(rate=0.5),
            Layer.output(units=10)
        )
        
        # Configure optimizer
        config = {
            'population_size': 10,
            'num_generations': 5,
            'elite_size': 2,
            'mutation_rate': 0.15,
            'crossover_rate': 0.7
        }
        
        # Create evaluator
        evaluator = HeuristicEvaluator()
        
        # Run optimization
        optimizer = GeneticAlgorithm(space, **config)
        best = optimizer.optimize(evaluator)
        
        # Validate results
        assert best is not None
        assert best.is_evaluated()
        assert best.fitness > 0
        assert best.graph.is_valid_dag()
        assert len(best.graph.nodes) > 0
        
        # Verify population evolved
        assert optimizer.population.generation == config['num_generations']
        assert optimizer.population.size() == config['population_size']
        
        # Check statistics
        stats = optimizer.population.get_statistics()
        assert 'best_fitness' in stats
        assert 'mean_fitness' in stats
        assert stats['best_fitness'] >= stats['mean_fitness']

    def test_multiple_optimizers_same_problem(self) -> None:
        """Test multiple optimizers on the same search space."""
        space = SearchSpace("multi_optimizer")
        space.add_layers(
            Layer.input(shape=(784,)),
            Layer.dense(units=[64, 128]),
            Layer.relu(),
            Layer.dropout(rate=[0.2, 0.5]),
            Layer.output(units=10)
        )
        
        evaluator = HeuristicEvaluator()
        results = {}
        
        # Test RandomSearch
        rs = RandomSearch(space, num_samples=20)
        rs_best = rs.optimize(evaluator)
        results['RandomSearch'] = rs_best.fitness
        
        # Test HillClimbing
        hc = HillClimbing(space, max_iterations=20)
        hc_best = hc.optimize(evaluator)
        results['HillClimbing'] = hc_best.fitness
        
        # Test GeneticAlgorithm
        ga = GeneticAlgorithm(space, population_size=10, num_generations=5)
        ga_best = ga.optimize(evaluator)
        results['GeneticAlgorithm'] = ga_best.fitness
        
        # Verify all produced valid results
        for optimizer_name, fitness in results.items():
            assert fitness > 0, f"{optimizer_name} produced invalid fitness"

    def test_checkpoint_save_and_resume(self) -> None:
        """Test checkpointing and resuming optimization."""
        space = SearchSpace("checkpoint_test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64),
            Layer.relu(),
            Layer.output(units=10)
        )
        
        evaluator = HeuristicEvaluator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.json")
            
            # Run for a few generations
            ga1 = GeneticAlgorithm(space, population_size=10, num_generations=3)
            ga1.optimize(evaluator)
            
            # Save checkpoint
            Checkpoint.save(ga1, checkpoint_path)
            assert os.path.exists(checkpoint_path)
            
            # Resume from checkpoint
            ga2 = Checkpoint.load(checkpoint_path, space, GeneticAlgorithm)
            assert ga2 is not None
            assert ga2.population.generation == ga1.population.generation
            
            # Continue optimization
            ga2.config['num_generations'] = 6
            final_best = ga2.optimize(evaluator)
            
            assert final_best is not None
            assert ga2.population.generation > ga1.population.generation

    def test_architecture_export_workflow(self) -> None:
        """Test complete architecture export workflow."""
        # Create a sample architecture
        space = SearchSpace("export_test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64, kernel_size=3),
            Layer.relu(),
            Layer.batchnorm(),
            Layer.maxpool(pool_size=2),
            Layer.conv2d(filters=128, kernel_size=3),
            Layer.relu(),
            Layer.flatten(),
            Layer.dense(units=256),
            Layer.dropout(rate=0.5),
            Layer.output(units=10)
        )
        
        # Sample architecture
        arch = space.sample()
        
        # Export to all formats
        exporter = ArchitectureExporter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # PyTorch export
            pytorch_code = exporter.to_pytorch(arch, "TestModel")
            pytorch_path = os.path.join(tmpdir, "model_pytorch.py")
            with open(pytorch_path, 'w') as f:
                f.write(pytorch_code)
            
            assert os.path.exists(pytorch_path)
            assert "import torch" in pytorch_code
            assert "class TestModel" in pytorch_code
            
            # Keras export
            keras_code = exporter.to_keras(arch, "test_model")
            keras_path = os.path.join(tmpdir, "model_keras.py")
            with open(keras_path, 'w') as f:
                f.write(keras_code)
            
            assert os.path.exists(keras_path)
            assert "keras" in keras_code
            assert "def test_model" in keras_code
            
            # JSON export
            json_str = exporter.to_json(arch)
            json_path = os.path.join(tmpdir, "model.json")
            with open(json_path, 'w') as f:
                f.write(json_str)
            
            assert os.path.exists(json_path)
            data = json.loads(json_str)
            assert "nodes" in data
            assert "edges" in data

    def test_constraint_satisfaction(self) -> None:
        """Test that constraints are properly enforced."""
        from morphml.constraints import MaxParametersConstraint, DepthConstraint
        
        space = SearchSpace("constraint_test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64, 128], kernel_size=3),
            Layer.relu(),
            Layer.maxpool(pool_size=2),
            Layer.dense(units=[128, 256, 512]),
            Layer.output(units=10)
        )
        
        # Add constraints
        space.add_constraint(MaxParametersConstraint(max_params=1000000))
        space.add_constraint(DepthConstraint(min_depth=4, max_depth=10))
        
        # Sample multiple architectures
        for _ in range(20):
            arch = space.sample()
            
            # Verify constraints
            assert arch.get_depth() >= 4
            assert arch.get_depth() <= 10
            assert arch.estimate_parameters() <= 1000000

    def test_diversity_maintenance(self) -> None:
        """Test that population maintains diversity."""
        space = SearchSpace("diversity_test")
        space.add_layers(
            Layer.input(shape=(784,)),
            Layer.dense(units=[32, 64, 128, 256]),
            Layer.relu(),
            Layer.dropout(rate=[0.1, 0.3, 0.5]),
            Layer.dense(units=[32, 64, 128]),
            Layer.output(units=10)
        )
        
        evaluator = HeuristicEvaluator()
        
        ga = GeneticAlgorithm(space, population_size=20, num_generations=10)
        ga.optimize(evaluator)
        
        # Check diversity
        diversity = ga.population.get_diversity()
        assert 0.0 <= diversity <= 1.0
        
        # Population should not converge to identical solutions
        node_counts = [len(ind.graph.nodes) for ind in ga.population.individuals]
        assert len(set(node_counts)) > 1, "Population converged to identical solutions"

    def test_evaluation_caching(self) -> None:
        """Test that evaluator caches results."""
        evaluator = HeuristicEvaluator()
        
        # Create identical graphs
        graph = ModelGraph()
        input_node = GraphNode.create("input", {"shape": (3, 32, 32)})
        dense_node = GraphNode.create("dense", {"units": 128})
        output_node = GraphNode.create("dense", {"units": 10})
        
        graph.add_node(input_node)
        graph.add_node(dense_node)
        graph.add_node(output_node)
        graph.add_edge(GraphEdge(input_node, dense_node))
        graph.add_edge(GraphEdge(dense_node, output_node))
        
        # First evaluation
        fitness1 = evaluator(graph)
        
        # Second evaluation (should be cached)
        fitness2 = evaluator(graph)
        
        # Should be identical
        assert fitness1 == fitness2

    def test_mutation_preserves_validity(self) -> None:
        """Test that all mutations preserve graph validity."""
        from morphml.core.mutation import GraphMutator
        
        space = SearchSpace("mutation_test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64),
            Layer.relu(),
            Layer.maxpool(pool_size=2),
            Layer.dense(units=128),
            Layer.output(units=10)
        )
        
        mutator = GraphMutator(space)
        
        # Test 100 mutations
        for _ in range(100):
            graph = space.sample()
            mutated = mutator.mutate(graph)
            
            # Must maintain DAG property
            assert mutated.is_valid_dag()
            # Must have nodes
            assert len(mutated.nodes) > 0
            # Must be connected
            assert mutated.get_depth() > 0

    def test_selection_strategies(self) -> None:
        """Test different selection strategies."""
        space = SearchSpace("selection_test")
        space.add_layers(
            Layer.input(shape=(784,)),
            Layer.dense(units=128),
            Layer.relu(),
            Layer.output(units=10)
        )
        
        evaluator = HeuristicEvaluator()
        
        strategies = ['tournament', 'roulette', 'rank']
        
        for strategy in strategies:
            ga = GeneticAlgorithm(
                space,
                population_size=15,
                num_generations=5,
                selection_strategy=strategy
            )
            
            best = ga.optimize(evaluator)
            assert best is not None
            assert best.fitness > 0

    def test_early_stopping(self) -> None:
        """Test early stopping when no improvement."""
        space = SearchSpace("early_stop_test")
        space.add_layers(
            Layer.input(shape=(784,)),
            Layer.dense(units=64),
            Layer.output(units=10)
        )
        
        evaluator = HeuristicEvaluator()
        
        hc = HillClimbing(space, max_iterations=100, patience=5)
        best = hc.optimize(evaluator)
        
        # Should stop early
        history = hc.get_history()
        assert len(history) < 100, "Early stopping did not work"


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_multi_stage_optimization(self) -> None:
        """Test multi-stage optimization workflow."""
        space = SearchSpace("multi_stage")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64], kernel_size=3),
            Layer.relu(),
            Layer.maxpool(pool_size=2),
            Layer.dense(units=[128, 256]),
            Layer.output(units=10)
        )
        
        evaluator = HeuristicEvaluator()
        
        # Stage 1: Random sampling
        rs = RandomSearch(space, num_samples=30)
        rs_best = rs.optimize(evaluator)
        
        # Stage 2: Hill climbing from best random
        hc = HillClimbing(space, max_iterations=20)
        hc.current = rs_best.graph
        hc_best = hc.optimize(evaluator)
        
        # Stage 3: Genetic algorithm
        ga = GeneticAlgorithm(space, population_size=10, num_generations=10)
        ga_best = ga.optimize(evaluator)
        
        # All should produce valid results
        assert rs_best.fitness > 0
        assert hc_best.fitness > 0
        assert ga_best.fitness > 0

    def test_parallel_search_spaces(self) -> None:
        """Test searching multiple spaces in parallel."""
        evaluator = HeuristicEvaluator()
        results = {}
        
        # CNN space
        cnn_space = SearchSpace("cnn")
        cnn_space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64, kernel_size=3),
            Layer.relu(),
            Layer.maxpool(pool_size=2),
            Layer.output(units=10)
        )
        
        # MLP space
        mlp_space = SearchSpace("mlp")
        mlp_space.add_layers(
            Layer.input(shape=(784,)),
            Layer.dense(units=[128, 256]),
            Layer.relu(),
            Layer.dense(units=[64, 128]),
            Layer.output(units=10)
        )
        
        # Search both
        cnn_ga = GeneticAlgorithm(cnn_space, population_size=10, num_generations=5)
        cnn_best = cnn_ga.optimize(evaluator)
        results['CNN'] = cnn_best
        
        mlp_ga = GeneticAlgorithm(mlp_space, population_size=10, num_generations=5)
        mlp_best = mlp_ga.optimize(evaluator)
        results['MLP'] = mlp_best
        
        # Both should succeed
        assert results['CNN'].fitness > 0
        assert results['MLP'].fitness > 0


def test_complete_production_workflow() -> None:
    """
    Test a complete production-like workflow.
    
    This simulates a real user workflow:
    1. Define search space
    2. Run optimization
    3. Save checkpoints
    4. Export best model
    5. Analyze results
    """
    # Setup
    space = SearchSpace("production_workflow")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=[32, 64], kernel_size=[3, 5]),
        Layer.relu(),
        Layer.batchnorm(),
        Layer.maxpool(pool_size=2),
        Layer.conv2d(filters=[64, 128], kernel_size=3),
        Layer.relu(),
        Layer.batchnorm(),
        Layer.maxpool(pool_size=2),
        Layer.flatten(),
        Layer.dense(units=[256, 512]),
        Layer.dropout(rate=[0.3, 0.5]),
        Layer.dense(units=[128, 256]),
        Layer.output(units=10)
    )
    
    evaluator = HeuristicEvaluator()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run optimization with checkpointing
        ga = GeneticAlgorithm(
            space,
            population_size=20,
            num_generations=10,
            elite_size=2,
            mutation_rate=0.15,
            crossover_rate=0.7
        )
        
        checkpoint_path = os.path.join(tmpdir, "checkpoint.json")
        
        # Optimize with callback for checkpointing
        def checkpoint_callback(generation, population):
            if generation % 3 == 0:
                Checkpoint.save(ga, checkpoint_path)
        
        best = ga.optimize(evaluator, callback=checkpoint_callback)
        
        # Export best model
        exporter = ArchitectureExporter()
        
        # PyTorch
        pytorch_code = exporter.to_pytorch(best.graph, "BestModel")
        pytorch_path = os.path.join(tmpdir, "best_model.py")
        with open(pytorch_path, 'w') as f:
            f.write(pytorch_code)
        
        # Keras
        keras_code = exporter.to_keras(best.graph, "best_model")
        keras_path = os.path.join(tmpdir, "best_model_keras.py")
        with open(keras_path, 'w') as f:
            f.write(keras_code)
        
        # JSON architecture
        json_str = exporter.to_json(best.graph)
        json_path = os.path.join(tmpdir, "best_architecture.json")
        with open(json_path, 'w') as f:
            f.write(json_str)
        
        # Save results summary
        summary = {
            'best_fitness': best.fitness,
            'final_generation': ga.population.generation,
            'population_size': ga.population.size(),
            'best_model_nodes': len(best.graph.nodes),
            'best_model_depth': best.graph.get_depth(),
            'estimated_parameters': best.graph.estimate_parameters(),
            'statistics': ga.population.get_statistics()
        }
        
        summary_path = os.path.join(tmpdir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Verify all outputs
        assert os.path.exists(checkpoint_path)
        assert os.path.exists(pytorch_path)
        assert os.path.exists(keras_path)
        assert os.path.exists(json_path)
        assert os.path.exists(summary_path)
        
        # Verify content
        assert best.fitness > 0
        assert best.graph.is_valid_dag()
        assert len(best.graph.nodes) > 0
        
        print(f"\n✓ Production workflow test passed!")
        print(f"  Best fitness: {best.fitness:.4f}")
        print(f"  Generations: {ga.population.generation}")
        print(f"  Model nodes: {len(best.graph.nodes)}")
        print(f"  Model depth: {best.graph.get_depth()}")
        print(f"  Parameters: {best.graph.estimate_parameters():,}")
