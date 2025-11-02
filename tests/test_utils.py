"""Tests for utility modules."""

import json
import os
import tempfile

from morphml.core.dsl import Layer, SearchSpace
from morphml.core.graph import GraphEdge, GraphNode, ModelGraph
from morphml.optimizers import GeneticAlgorithm, RandomSearch
from morphml.utils import ArchitectureExporter, Checkpoint


class TestCheckpoint:
    """Test checkpoint functionality."""

    def create_test_optimizer(self, space: SearchSpace) -> GeneticAlgorithm:
        """Create test optimizer."""
        return GeneticAlgorithm(
            search_space=space,
            population_size=10,
            num_generations=5
        )

    def create_space(self) -> SearchSpace:
        """Create test space."""
        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64),
            Layer.relu(),
            Layer.output(units=10)
        )
        return space

    def simple_evaluator(self, graph: ModelGraph) -> float:
        """Simple evaluator."""
        return 0.75

    def test_checkpoint_save(self) -> None:
        """Test saving checkpoint."""
        space = self.create_space()
        ga = self.create_test_optimizer(space)

        # Run some generations
        ga.initialize_population()
        ga.evaluate_population(self.simple_evaluator)

        # Save checkpoint
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            checkpoint_path = f.name

        try:
            Checkpoint.save(ga, checkpoint_path)

            # Check file exists
            assert os.path.exists(checkpoint_path)

            # Check file is valid JSON
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)

            assert "optimizer_type" in data
            assert data["optimizer_type"] == "GeneticAlgorithm"

        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

    def test_checkpoint_load(self) -> None:
        """Test loading checkpoint."""
        space = self.create_space()
        ga = self.create_test_optimizer(space)

        # Run and save
        ga.initialize_population()
        ga.evaluate_population(self.simple_evaluator)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            checkpoint_path = f.name

        try:
            Checkpoint.save(ga, checkpoint_path)

            # Load checkpoint
            loaded_ga = Checkpoint.load(checkpoint_path, space)

            assert loaded_ga is not None
            assert loaded_ga.population.size() == ga.population.size()
            assert loaded_ga.population.generation == ga.population.generation

        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

    def test_checkpoint_save_load_cycle(self) -> None:
        """Test full save-load cycle."""
        space = self.create_space()
        ga = self.create_test_optimizer(space)

        # Run optimization
        ga.initialize_population()
        ga.evaluate_population(self.simple_evaluator)
        ga.evolve_generation()
        ga.evaluate_population(self.simple_evaluator)

        original_gen = ga.population.generation
        original_size = ga.population.size()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            checkpoint_path = f.name

        try:
            # Save
            Checkpoint.save(ga, checkpoint_path)

            # Load
            loaded_ga = Checkpoint.load(checkpoint_path, space, GeneticAlgorithm)

            # Verify
            assert loaded_ga.population.generation == original_gen
            assert loaded_ga.population.size() == original_size

            # Continue optimization
            loaded_ga.evolve_generation()
            assert loaded_ga.population.generation == original_gen + 1

        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

    def test_checkpoint_random_search(self) -> None:
        """Test checkpoint with RandomSearch."""
        space = self.create_space()
        rs = RandomSearch(space, num_samples=10)

        # Run partially
        for _ in range(5):
            graph = space.sample()
            from morphml.core.search import Individual
            ind = Individual(graph)
            ind.set_fitness(0.8)
            rs.evaluated.append(ind)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            checkpoint_path = f.name

        try:
            Checkpoint.save(rs, checkpoint_path)

            # Load
            loaded_rs = Checkpoint.load(checkpoint_path, space, RandomSearch)

            assert len(loaded_rs.evaluated) == 5

        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)


class TestArchitectureExporter:
    """Test architecture export functionality."""

    def create_sample_graph(self) -> ModelGraph:
        """Create sample graph for testing."""
        graph = ModelGraph()

        input_node = GraphNode.create("input", {"shape": (3, 32, 32)})
        conv1 = GraphNode.create("conv2d", {"filters": 64, "kernel_size": 3})
        relu = GraphNode.create("relu")
        pool = GraphNode.create("maxpool", {"pool_size": 2})
        dense = GraphNode.create("dense", {"units": 128})
        output = GraphNode.create("dense", {"units": 10})

        graph.add_node(input_node)
        graph.add_node(conv1)
        graph.add_node(relu)
        graph.add_node(pool)
        graph.add_node(dense)
        graph.add_node(output)

        graph.add_edge(GraphEdge(input_node, conv1))
        graph.add_edge(GraphEdge(conv1, relu))
        graph.add_edge(GraphEdge(relu, pool))
        graph.add_edge(GraphEdge(pool, dense))
        graph.add_edge(GraphEdge(dense, output))

        return graph

    def test_exporter_creation(self) -> None:
        """Test creating exporter."""
        exporter = ArchitectureExporter()

        assert exporter is not None

    def test_pytorch_export(self) -> None:
        """Test PyTorch code generation."""
        exporter = ArchitectureExporter()
        graph = self.create_sample_graph()

        code = exporter.to_pytorch(graph, "TestModel")

        assert "import torch" in code
        assert "import torch.nn as nn" in code
        assert "class TestModel" in code
        assert "def __init__" in code
        assert "def forward" in code

    def test_pytorch_custom_class_name(self) -> None:
        """Test PyTorch with custom class name."""
        exporter = ArchitectureExporter()
        graph = self.create_sample_graph()

        code = exporter.to_pytorch(graph, "MyCustomModel")

        assert "class MyCustomModel" in code

    def test_keras_export(self) -> None:
        """Test Keras code generation."""
        exporter = ArchitectureExporter()
        graph = self.create_sample_graph()

        code = exporter.to_keras(graph, "test_model")

        assert "import tensorflow" in code
        assert "from tensorflow import keras" in code
        assert "def test_model" in code
        assert "keras.Input" in code
        assert "keras.Model" in code

    def test_keras_custom_name(self) -> None:
        """Test Keras with custom function name."""
        exporter = ArchitectureExporter()
        graph = self.create_sample_graph()

        code = exporter.to_keras(graph, "my_custom_model")

        assert "def my_custom_model" in code

    def test_json_export(self) -> None:
        """Test JSON export."""
        exporter = ArchitectureExporter()
        graph = self.create_sample_graph()

        json_str = exporter.to_json(graph)

        # Should be valid JSON
        data = json.loads(json_str)

        assert "nodes" in data
        assert "edges" in data

    def test_export_simple_architecture(self) -> None:
        """Test exporting simple architecture."""
        exporter = ArchitectureExporter()

        graph = ModelGraph()
        input_node = GraphNode.create("input", {"shape": (784,)})
        dense = GraphNode.create("dense", {"units": 128})
        output = GraphNode.create("dense", {"units": 10})

        graph.add_node(input_node)
        graph.add_node(dense)
        graph.add_node(output)
        graph.add_edge(GraphEdge(input_node, dense))
        graph.add_edge(GraphEdge(dense, output))

        # PyTorch
        pytorch_code = exporter.to_pytorch(graph)
        assert "torch.nn" in pytorch_code

        # Keras
        keras_code = exporter.to_keras(graph)
        assert "keras" in keras_code

    def test_export_with_all_layer_types(self) -> None:
        """Test exporting with various layer types."""
        exporter = ArchitectureExporter()

        graph = ModelGraph()

        input_node = GraphNode.create("input", {"shape": (3, 32, 32)})
        conv = GraphNode.create("conv2d", {"filters": 32, "kernel_size": 3})
        relu = GraphNode.create("relu")
        maxpool = GraphNode.create("maxpool", {"pool_size": 2})
        avgpool = GraphNode.create("avgpool", {"pool_size": 2})
        dropout = GraphNode.create("dropout", {"rate": 0.5})
        batchnorm = GraphNode.create("batchnorm")
        dense = GraphNode.create("dense", {"units": 128})
        sigmoid = GraphNode.create("sigmoid")
        output = GraphNode.create("dense", {"units": 10})

        nodes = [input_node, conv, relu, maxpool, avgpool, dropout, batchnorm, dense, sigmoid, output]

        prev = None
        for node in nodes:
            graph.add_node(node)
            if prev is not None:
                graph.add_edge(GraphEdge(prev, node))
            prev = node

        # Should not crash
        pytorch_code = exporter.to_pytorch(graph)
        keras_code = exporter.to_keras(graph)

        assert len(pytorch_code) > 100
        assert len(keras_code) > 100

    def test_export_file_write(self) -> None:
        """Test writing exported code to file."""
        exporter = ArchitectureExporter()
        graph = self.create_sample_graph()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            pytorch_path = f.name

        try:
            # Generate and write PyTorch code
            code = exporter.to_pytorch(graph)
            with open(pytorch_path, 'w') as f:
                f.write(code)

            # Verify file
            assert os.path.exists(pytorch_path)

            with open(pytorch_path, 'r') as f:
                content = f.read()

            assert "import torch" in content

        finally:
            if os.path.exists(pytorch_path):
                os.remove(pytorch_path)


class TestUtilsIntegration:
    """Integration tests for utils."""

    def test_checkpoint_export_workflow(self) -> None:
        """Test complete workflow with checkpoint and export."""
        from morphml.evaluation import HeuristicEvaluator

        # Create and run optimization
        space = SearchSpace("workflow")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64], kernel_size=3),
            Layer.relu(),
            Layer.maxpool(pool_size=2),
            Layer.dense(units=[128, 256]),
            Layer.output(units=10)
        )

        ga = GeneticAlgorithm(space, population_size=10, num_generations=3)
        evaluator = HeuristicEvaluator()

        best = ga.optimize(evaluator)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint
            checkpoint_path = os.path.join(tmpdir, "checkpoint.json")
            Checkpoint.save(ga, checkpoint_path)

            # Export best architecture
            exporter = ArchitectureExporter()

            pytorch_path = os.path.join(tmpdir, "model.py")
            pytorch_code = exporter.to_pytorch(best.graph, "BestModel")
            with open(pytorch_path, 'w') as f:
                f.write(pytorch_code)

            keras_path = os.path.join(tmpdir, "model_keras.py")
            keras_code = exporter.to_keras(best.graph, "best_model")
            with open(keras_path, 'w') as f:
                f.write(keras_code)

            json_path = os.path.join(tmpdir, "architecture.json")
            json_str = exporter.to_json(best.graph)
            with open(json_path, 'w') as f:
                f.write(json_str)

            # Verify all files exist
            assert os.path.exists(checkpoint_path)
            assert os.path.exists(pytorch_path)
            assert os.path.exists(keras_path)
            assert os.path.exists(json_path)

            # Verify content
            with open(pytorch_path, 'r') as f:
                assert "BestModel" in f.read()

            with open(json_path, 'r') as f:
                json.load(f)  # Should be valid JSON

    def test_resume_and_export(self) -> None:
        """Test resuming from checkpoint and exporting."""
        from morphml.evaluation import HeuristicEvaluator

        space = SearchSpace("resume_test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64),
            Layer.relu(),
            Layer.output(units=10)
        )

        evaluator = HeuristicEvaluator()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.json")

            # First run
            ga1 = GeneticAlgorithm(space, population_size=5, num_generations=2)
            ga1.optimize(evaluator)

            Checkpoint.save(ga1, checkpoint_path)

            # Resume
            ga2 = Checkpoint.load(checkpoint_path, space, GeneticAlgorithm)

            # Continue for more generations
            ga2.config['num_generations'] = 4
            best = ga2.optimize(evaluator)

            # Export final result
            exporter = ArchitectureExporter()
            code = exporter.to_pytorch(best.graph)

            assert "import torch" in code


def test_utils_complete_workflow() -> None:
    """Complete workflow test."""
    from morphml.evaluation import HeuristicEvaluator

    # Setup
    space = SearchSpace("complete_workflow")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=[32, 64], kernel_size=[3, 5]),
        Layer.relu(),
        Layer.maxpool(pool_size=2),
        Layer.conv2d(filters=[64, 128], kernel_size=3),
        Layer.relu(),
        Layer.maxpool(pool_size=2),
        Layer.dense(units=[256, 512]),
        Layer.dropout(rate=[0.3, 0.5]),
        Layer.output(units=10)
    )

    evaluator = HeuristicEvaluator()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Phase 1: Initial search
        print("Phase 1: Random sampling...")
        rs = RandomSearch(space, num_samples=20)
        rs_best = rs.optimize(evaluator)

        # Phase 2: Evolutionary search
        print("Phase 2: Genetic algorithm...")
        ga = GeneticAlgorithm(space, population_size=15, num_generations=5)

        # Save checkpoint every 2 generations
        def callback(gen, pop):
            if gen % 2 == 0:
                cp_path = os.path.join(tmpdir, f"checkpoint_gen_{gen}.json")
                Checkpoint.save(ga, cp_path)

        ga_best = ga.optimize(evaluator, callback=callback)

        # Phase 3: Export results
        print("Phase 3: Exporting...")
        exporter = ArchitectureExporter()

        # Export random search best
        rs_pytorch = exporter.to_pytorch(rs_best.graph, "RandomSearchBest")
        with open(os.path.join(tmpdir, "rs_model.py"), 'w') as f:
            f.write(rs_pytorch)

        # Export GA best
        ga_pytorch = exporter.to_pytorch(ga_best.graph, "GABest")
        with open(os.path.join(tmpdir, "ga_model.py"), 'w') as f:
            f.write(ga_pytorch)

        ga_keras = exporter.to_keras(ga_best.graph, "ga_best_model")
        with open(os.path.join(tmpdir, "ga_model_keras.py"), 'w') as f:
            f.write(ga_keras)

        # Verify all outputs
        assert os.path.exists(os.path.join(tmpdir, "rs_model.py"))
        assert os.path.exists(os.path.join(tmpdir, "ga_model.py"))
        assert os.path.exists(os.path.join(tmpdir, "ga_model_keras.py"))

        # Verify checkpoints
        checkpoints = [f for f in os.listdir(tmpdir) if f.startswith("checkpoint_")]
        assert len(checkpoints) > 0

        print(f"Workflow complete! Generated {len(checkpoints)} checkpoints and 3 model files.")
