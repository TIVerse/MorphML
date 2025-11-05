"""Tests for Master node."""

import time
from unittest.mock import Mock, patch

import pytest

from morphml.core.dsl import Layer, SearchSpace
from morphml.core.graph import ModelGraph
from morphml.optimizers import GeneticAlgorithm

# Skip if gRPC not available
pytest.importorskip("grpc")

from morphml.distributed import MasterNode, WorkerInfo, Task


class TestWorkerInfo:
    """Test WorkerInfo dataclass."""
    
    def test_worker_info_creation(self) -> None:
        """Test creating WorkerInfo."""
        worker = WorkerInfo(
            worker_id="worker-1",
            host="localhost",
            port=50052,
            num_gpus=2,
            gpu_ids=[0, 1],
        )
        
        assert worker.worker_id == "worker-1"
        assert worker.host == "localhost"
        assert worker.port == 50052
        assert worker.num_gpus == 2
        assert worker.status == "idle"
    
    def test_worker_is_alive(self) -> None:
        """Test worker alive check."""
        worker = WorkerInfo(
            worker_id="worker-1",
            host="localhost",
            port=50052,
            num_gpus=1,
        )
        
        assert worker.is_alive(timeout=30.0)
        
        # Simulate old heartbeat
        worker.last_heartbeat = time.time() - 60
        assert not worker.is_alive(timeout=30.0)
    
    def test_worker_is_available(self) -> None:
        """Test worker availability check."""
        worker = WorkerInfo(
            worker_id="worker-1",
            host="localhost",
            port=50052,
            num_gpus=1,
        )
        
        assert worker.is_available()
        
        worker.status = "busy"
        assert not worker.is_available()
        
        worker.status = "idle"
        worker.last_heartbeat = time.time() - 60
        assert not worker.is_available()


class TestTask:
    """Test Task dataclass."""
    
    def test_task_creation(self) -> None:
        """Test creating Task."""
        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64),
            Layer.output(units=10),
        )
        graph = space.sample()
        
        task = Task(task_id="task-1", architecture=graph)
        
        assert task.task_id == "task-1"
        assert task.status == "pending"
        assert task.worker_id is None
        assert task.result is None
    
    def test_task_duration(self) -> None:
        """Test task duration calculation."""
        space = SearchSpace("test")
        space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))
        graph = space.sample()
        
        task = Task(task_id="task-1", architecture=graph)
        
        assert task.duration() is None
        
        task.started_at = time.time()
        time.sleep(0.1)
        task.completed_at = time.time()
        
        duration = task.duration()
        assert duration is not None
        assert duration > 0.09
    
    def test_task_can_retry(self) -> None:
        """Test task retry logic."""
        space = SearchSpace("test")
        space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))
        graph = space.sample()
        
        task = Task(task_id="task-1", architecture=graph, max_retries=3)
        
        assert task.can_retry()
        
        task.num_retries = 2
        assert task.can_retry()
        
        task.num_retries = 3
        assert not task.can_retry()


class TestMasterNode:
    """Test MasterNode."""
    
    def create_test_space(self) -> SearchSpace:
        """Create test search space."""
        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64], kernel_size=3),
            Layer.relu(),
            Layer.output(units=10),
        )
        return space
    
    def test_master_initialization(self) -> None:
        """Test master node initialization."""
        space = self.create_test_space()
        optimizer = GeneticAlgorithm(space, population_size=10, num_generations=5)
        
        config = {"port": 50051, "num_workers": 4}
        master = MasterNode(optimizer, config)
        
        assert master.port == 50051
        assert master.num_workers == 4
        assert len(master.workers) == 0
        assert not master.running
    
    def test_master_start_stop(self) -> None:
        """Test master start and stop."""
        space = self.create_test_space()
        optimizer = GeneticAlgorithm(space, population_size=10, num_generations=5)
        
        config = {"port": 50061, "num_workers": 2}  # Use different port
        master = MasterNode(optimizer, config)
        
        master.start()
        assert master.running
        assert master.server is not None
        
        time.sleep(0.5)  # Let threads start
        
        master.stop()
        assert not master.running
    
    def test_worker_registration(self) -> None:
        """Test worker registration."""
        space = self.create_test_space()
        optimizer = GeneticAlgorithm(space, population_size=10, num_generations=5)
        
        master = MasterNode(optimizer, {"port": 50071})
        
        worker_info = {
            "host": "localhost",
            "port": 50052,
            "num_gpus": 2,
            "gpu_ids": [0, 1],
        }
        
        success = master.register_worker("worker-1", worker_info)
        
        assert success
        assert "worker-1" in master.workers
        assert master.workers["worker-1"].host == "localhost"
        assert master.workers["worker-1"].num_gpus == 2
    
    def test_heartbeat_update(self) -> None:
        """Test heartbeat update."""
        space = self.create_test_space()
        optimizer = GeneticAlgorithm(space, population_size=10, num_generations=5)
        
        master = MasterNode(optimizer, {"port": 50081})
        
        worker_info = {"host": "localhost", "port": 50052, "num_gpus": 1}
        master.register_worker("worker-1", worker_info)
        
        time.sleep(0.1)
        
        success = master.update_heartbeat("worker-1", "busy")
        
        assert success
        assert master.workers["worker-1"].status == "busy"
        assert master.workers["worker-1"].is_alive()
    
    def test_task_submission(self) -> None:
        """Test task submission."""
        space = self.create_test_space()
        optimizer = GeneticAlgorithm(space, population_size=10, num_generations=5)
        
        master = MasterNode(optimizer, {"port": 50091})
        graph = space.sample()
        
        task_id = master.submit_task(graph)
        
        assert task_id is not None
        assert len(task_id) > 0
    
    def test_find_available_worker(self) -> None:
        """Test finding available worker."""
        space = self.create_test_space()
        optimizer = GeneticAlgorithm(space, population_size=10, num_generations=5)
        
        master = MasterNode(optimizer, {"port": 50101})
        
        # No workers
        worker = master._find_available_worker()
        assert worker is None
        
        # Add idle worker
        master.register_worker("worker-1", {"host": "localhost", "port": 50052, "num_gpus": 1})
        worker = master._find_available_worker()
        assert worker is not None
        assert worker.worker_id == "worker-1"
        
        # Make worker busy
        master.workers["worker-1"].status = "busy"
        worker = master._find_available_worker()
        assert worker is None
    
    def test_statistics(self) -> None:
        """Test getting statistics."""
        space = self.create_test_space()
        optimizer = GeneticAlgorithm(space, population_size=10, num_generations=5)
        
        master = MasterNode(optimizer, {"port": 50111})
        
        stats = master.get_statistics()
        
        assert "workers_total" in stats
        assert "tasks_pending" in stats
        assert "tasks_completed" in stats
        assert stats["workers_total"] == 0


def test_master_integration_basic() -> None:
    """Basic integration test for master node."""
    # Create search space
    space = SearchSpace("test")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)), Layer.conv2d(filters=32), Layer.output(units=10)
    )
    
    # Create optimizer
    optimizer = GeneticAlgorithm(space, population_size=5, num_generations=2)
    
    # Create master
    config = {"port": 50121, "num_workers": 1, "heartbeat_interval": 5}
    master = MasterNode(optimizer, config)
    
    # Start master
    master.start()
    assert master.running
    
    # Register a worker
    worker_info = {"host": "localhost", "port": 50052, "num_gpus": 1}
    master.register_worker("test-worker", worker_info)
    
    # Submit a task
    graph = space.sample()
    task_id = master.submit_task(graph)
    assert task_id is not None
    
    # Check statistics
    stats = master.get_statistics()
    assert stats["workers_total"] == 1
    
    # Stop master
    master.stop()
    assert not master.running
