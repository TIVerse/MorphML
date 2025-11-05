"""Tests for Worker node."""

import time

import pytest

from morphml.core.dsl import Layer, SearchSpace
from morphml.core.graph import ModelGraph

# Skip if gRPC not available
pytest.importorskip("grpc")

from morphml.distributed import WorkerNode


class TestWorkerNode:
    """Test WorkerNode."""

    def test_worker_initialization(self) -> None:
        """Test worker initialization."""
        config = {
            "master_host": "localhost",
            "master_port": 50051,
            "port": 50052,
            "num_gpus": 2,
            "gpu_ids": [0, 1],
        }

        worker = WorkerNode(config)

        assert worker.master_host == "localhost"
        assert worker.master_port == 50051
        assert worker.port == 50052
        assert worker.num_gpus == 2
        assert worker.gpu_ids == [0, 1]
        assert not worker.running

    def test_worker_with_custom_evaluator(self) -> None:
        """Test worker with custom evaluator."""

        def custom_eval(graph: ModelGraph) -> dict:
            return {"fitness": 0.95, "custom_metric": 42.0}

        config = {
            "master_host": "localhost",
            "master_port": 50051,
            "evaluator": custom_eval,
        }

        worker = WorkerNode(config)

        assert worker.evaluator is custom_eval

    def test_default_evaluation(self) -> None:
        """Test default heuristic evaluation."""
        config = {"master_host": "localhost", "master_port": 50051}

        worker = WorkerNode(config)

        # Create test architecture
        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64, kernel_size=3),
            Layer.relu(),
            Layer.output(units=10),
        )
        graph = space.sample()

        # Evaluate
        result = worker.evaluate_architecture(graph)

        assert "fitness" in result
        assert "params" in result
        assert "depth" in result
        assert "worker_id" in result
        assert result["fitness"] >= 0.0

    def test_custom_evaluation(self) -> None:
        """Test custom evaluator."""

        def custom_eval(graph: ModelGraph) -> float:
            # Return constant fitness
            return 0.85

        config = {
            "master_host": "localhost",
            "master_port": 50051,
            "evaluator": custom_eval,
        }

        worker = WorkerNode(config)

        space = SearchSpace("test")
        space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))
        graph = space.sample()

        result = worker.evaluate_architecture(graph)

        assert result["fitness"] == 0.85

    def test_custom_evaluation_dict(self) -> None:
        """Test custom evaluator returning dictionary."""

        def custom_eval(graph: ModelGraph) -> dict:
            return {
                "fitness": 0.92,
                "accuracy": 0.92,
                "latency": 15.2,
                "params": 1_000_000,
            }

        config = {
            "master_host": "localhost",
            "master_port": 50051,
            "evaluator": custom_eval,
        }

        worker = WorkerNode(config)

        space = SearchSpace("test")
        space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))
        graph = space.sample()

        result = worker.evaluate_architecture(graph)

        assert result["fitness"] == 0.92
        assert result["accuracy"] == 0.92
        assert result["latency"] == 15.2
        assert result["params"] == 1_000_000

    def test_get_status(self) -> None:
        """Test getting worker status."""
        config = {"master_host": "localhost", "master_port": 50051}

        worker = WorkerNode(config)
        worker.start_time = time.time()

        status = worker.get_status()

        assert "worker_id" in status
        assert "status" in status
        assert "tasks_completed" in status
        assert "tasks_failed" in status
        assert "uptime_seconds" in status
        assert status["status"] == "idle"
        assert status["tasks_completed"] == 0


def test_worker_evaluation_with_error() -> None:
    """Test worker handling evaluation errors."""

    def failing_eval(graph: ModelGraph) -> float:
        raise ValueError("Intentional evaluation failure")

    config = {
        "master_host": "localhost",
        "master_port": 50051,
        "evaluator": failing_eval,
    }

    worker = WorkerNode(config)

    space = SearchSpace("test")
    space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))
    graph = space.sample()

    # Should raise the exception
    with pytest.raises(ValueError, match="Intentional evaluation failure"):
        worker.evaluate_architecture(graph)


def test_worker_multiple_evaluations() -> None:
    """Test worker performing multiple evaluations."""
    evaluation_count = 0

    def counting_eval(graph: ModelGraph) -> dict:
        nonlocal evaluation_count
        evaluation_count += 1
        return {"fitness": 0.5 + evaluation_count * 0.1}

    config = {
        "master_host": "localhost",
        "master_port": 50051,
        "evaluator": counting_eval,
    }

    worker = WorkerNode(config)

    space = SearchSpace("test")
    space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))

    # Perform multiple evaluations
    results = []
    for _i in range(3):
        graph = space.sample()
        result = worker.evaluate_architecture(graph)
        results.append(result["fitness"])

    assert evaluation_count == 3
    assert results[0] == 0.6
    assert results[1] == 0.7
    assert results[2] == 0.8
