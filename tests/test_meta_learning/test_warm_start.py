"""Tests for warm-starting functionality."""

import pytest

from morphml.core.dsl import Layer, SearchSpace
from morphml.meta_learning import (
    ArchitectureSimilarity,
    ExperimentDatabase,
    TaskMetadata,
    WarmStarter,
)


class TestTaskMetadata:
    """Test TaskMetadata dataclass."""

    def test_creation(self):
        """Test creating task metadata."""
        task = TaskMetadata(
            task_id="test_task",
            dataset_name="CIFAR-10",
            num_samples=50000,
            num_classes=10,
            input_size=(3, 32, 32),
            problem_type="classification",
        )

        assert task.task_id == "test_task"
        assert task.num_classes == 10

    def test_to_dict(self):
        """Test serialization."""
        task = TaskMetadata(
            task_id="test",
            dataset_name="CIFAR-10",
            num_samples=50000,
            num_classes=10,
            input_size=(3, 32, 32),
        )

        data = task.to_dict()

        assert data["task_id"] == "test"
        assert data["num_classes"] == 10

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "task_id": "test",
            "dataset_name": "CIFAR-10",
            "num_samples": 50000,
            "num_classes": 10,
            "input_size": [3, 32, 32],
        }

        task = TaskMetadata.from_dict(data)

        assert task.task_id == "test"
        assert task.input_size == (3, 32, 32)


class TestExperimentDatabase:
    """Test experiment database."""

    def test_initialization(self):
        """Test database initialization."""
        db = ExperimentDatabase()

        assert db._task_cache is not None

    def test_add_and_get_task(self):
        """Test adding and retrieving tasks."""
        db = ExperimentDatabase()

        task = TaskMetadata(
            task_id="task1",
            dataset_name="CIFAR-10",
            num_samples=50000,
            num_classes=10,
            input_size=(3, 32, 32),
        )

        db.add_task(task)

        retrieved = db.get_task("task1")
        assert retrieved is not None
        assert retrieved.task_id == "task1"

    def test_get_all_tasks(self):
        """Test getting all tasks."""
        db = ExperimentDatabase()

        task1 = TaskMetadata("t1", "D1", 1000, 10, (3, 32, 32))
        task2 = TaskMetadata("t2", "D2", 2000, 100, (3, 64, 64))

        db.add_task(task1)
        db.add_task(task2)

        all_tasks = db.get_all_tasks()

        assert len(all_tasks) == 2


class TestArchitectureSimilarity:
    """Test architecture similarity metrics."""

    @pytest.fixture
    def sample_graphs(self):
        """Create sample graphs."""
        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64),
            Layer.conv2d(filters=128),
            Layer.output(units=10),
        )

        graph1 = space.sample()
        graph2 = space.sample()

        return graph1, graph2

    def test_operation_distribution_similarity(self, sample_graphs):
        """Test operation distribution similarity."""
        graph1, graph2 = sample_graphs

        similarity = ArchitectureSimilarity.operation_distribution_similarity(graph1, graph2)

        assert 0.0 <= similarity <= 1.0

    def test_graph_structure_similarity(self, sample_graphs):
        """Test graph structure similarity."""
        graph1, graph2 = sample_graphs

        similarity = ArchitectureSimilarity.graph_structure_similarity(graph1, graph2)

        assert 0.0 <= similarity <= 1.0

    def test_compute_method(self, sample_graphs):
        """Test compute method."""
        graph1, graph2 = sample_graphs

        # Test different methods
        sim1 = ArchitectureSimilarity.compute(graph1, graph2, method="operation_distribution")
        sim2 = ArchitectureSimilarity.compute(graph1, graph2, method="graph_structure")
        sim3 = ArchitectureSimilarity.compute(graph1, graph2, method="combined")

        assert all(0.0 <= s <= 1.0 for s in [sim1, sim2, sim3])


class TestWarmStarter:
    """Test warm-starting."""

    @pytest.fixture
    def knowledge_base(self):
        """Create knowledge base with sample data."""
        db = ExperimentDatabase()

        # Add past tasks
        task1 = TaskMetadata(
            task_id="past1",
            dataset_name="CIFAR-10",
            num_samples=50000,
            num_classes=10,
            input_size=(3, 32, 32),
            problem_type="classification",
        )

        task2 = TaskMetadata(
            task_id="past2",
            dataset_name="CIFAR-100",
            num_samples=50000,
            num_classes=100,
            input_size=(3, 32, 32),
            problem_type="classification",
        )

        db.add_task(task1)
        db.add_task(task2)

        # Add sample architectures
        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64),
            Layer.output(units=10),
        )

        for _ in range(5):
            graph = space.sample()
            db.add_architecture("past1", graph, fitness=0.9)
            db.add_architecture("past2", graph, fitness=0.85)

        return db

    def test_initialization(self, knowledge_base):
        """Test warm-starter initialization."""
        warm_starter = WarmStarter(knowledge_base)

        assert warm_starter.transfer_ratio == 0.5
        assert warm_starter.min_similarity == 0.6

    def test_generate_initial_population(self, knowledge_base):
        """Test population generation."""
        warm_starter = WarmStarter(knowledge_base, {"transfer_ratio": 0.5})

        # Current task similar to past tasks
        current_task = TaskMetadata(
            task_id="current",
            dataset_name="CIFAR-100",
            num_samples=50000,
            num_classes=100,
            input_size=(3, 32, 32),
            problem_type="classification",
        )

        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64),
            Layer.output(units=100),
        )

        population = warm_starter.generate_initial_population(
            current_task, population_size=20, search_space=space
        )

        assert len(population) == 20
        assert all(g is not None for g in population)

    def test_no_similar_tasks(self):
        """Test with no similar tasks."""
        db = ExperimentDatabase()
        warm_starter = WarmStarter(db)

        current_task = TaskMetadata(
            task_id="unique",
            dataset_name="Unique",
            num_samples=1000,
            num_classes=5,
            input_size=(1, 28, 28),
        )

        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(1, 28, 28)),
            Layer.dense(units=128),
            Layer.output(units=5),
        )

        population = warm_starter.generate_initial_population(
            current_task, population_size=10, search_space=space
        )

        # Should fall back to random
        assert len(population) == 10


def test_meta_learning_imports():
    """Test that meta-learning classes can be imported."""
    from morphml.meta_learning import (
        ArchitectureSimilarity,
        ExperimentDatabase,
        TaskMetadata,
        WarmStarter,
    )

    assert WarmStarter is not None
    assert TaskMetadata is not None
    assert ExperimentDatabase is not None
    assert ArchitectureSimilarity is not None
