"""Tests for task schedulers."""

import pytest

from morphml.core.dsl import Layer, SearchSpace
from morphml.distributed import (
    AdaptiveScheduler,
    FIFOScheduler,
    LoadBalancingScheduler,
    PriorityScheduler,
    RoundRobinScheduler,
    Task,
    WorkerInfo,
    WorkStealingScheduler,
    create_scheduler,
)


class TestFIFOScheduler:
    """Test FIFO scheduler."""
    
    def test_initialization(self) -> None:
        """Test scheduler initialization."""
        scheduler = FIFOScheduler()
        assert scheduler.assignments == 0
    
    def test_assign_to_first_available(self) -> None:
        """Test assignment to first available worker."""
        scheduler = FIFOScheduler()
        
        workers = [
            WorkerInfo("w1", "host1", 50052, 2, status="busy"),
            WorkerInfo("w2", "host2", 50052, 2, status="idle"),
            WorkerInfo("w3", "host3", 50052, 2, status="idle"),
        ]
        
        space = SearchSpace("test")
        space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))
        task = Task("task-1", space.sample())
        
        worker = scheduler.assign_task(task, workers)
        
        assert worker is not None
        assert worker.worker_id == "w2"
        assert scheduler.assignments == 1
    
    def test_no_available_workers(self) -> None:
        """Test when no workers available."""
        scheduler = FIFOScheduler()
        
        workers = [
            WorkerInfo("w1", "host1", 50052, 2, status="busy"),
            WorkerInfo("w2", "host2", 50052, 2, status="busy"),
        ]
        
        space = SearchSpace("test")
        space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))
        task = Task("task-1", space.sample())
        
        worker = scheduler.assign_task(task, workers)
        
        assert worker is None


class TestPriorityScheduler:
    """Test priority scheduler."""
    
    def test_priority_queue(self) -> None:
        """Test priority queue ordering."""
        scheduler = PriorityScheduler()
        
        space = SearchSpace("test")
        space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))
        
        task1 = Task("task-1", space.sample())
        task2 = Task("task-2", space.sample())
        task3 = Task("task-3", space.sample())
        
        scheduler.enqueue(task1, priority=0.5)
        scheduler.enqueue(task2, priority=0.9)
        scheduler.enqueue(task3, priority=0.7)
        
        # Highest priority first
        first = scheduler.dequeue()
        assert first is not None
        assert first.task_id == "task-2"
        
        second = scheduler.dequeue()
        assert second.task_id == "task-3"
    
    def test_assign_to_least_loaded(self) -> None:
        """Test assignment to least loaded worker."""
        scheduler = PriorityScheduler()
        
        workers = [
            WorkerInfo("w1", "host1", 50052, 2, status="idle", tasks_completed=10),
            WorkerInfo("w2", "host2", 50052, 2, status="idle", tasks_completed=5),
        ]
        
        space = SearchSpace("test")
        space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))
        task = Task("task-1", space.sample())
        
        worker = scheduler.assign_task(task, workers)
        
        assert worker.worker_id == "w2"  # Fewer completed tasks


class TestLoadBalancingScheduler:
    """Test load balancing scheduler."""
    
    def test_load_calculation(self) -> None:
        """Test load calculation."""
        scheduler = LoadBalancingScheduler()
        
        worker_idle = WorkerInfo("w1", "host1", 50052, 4, status="idle")
        worker_busy = WorkerInfo("w2", "host2", 50052, 2, status="busy")
        
        load_idle = scheduler._calculate_load(worker_idle)
        load_busy = scheduler._calculate_load(worker_busy)
        
        assert load_busy > load_idle
    
    def test_assign_to_least_loaded(self) -> None:
        """Test assignment to least loaded worker."""
        scheduler = LoadBalancingScheduler()
        
        workers = [
            WorkerInfo("w1", "host1", 50052, 4, status="busy", tasks_failed=2),
            WorkerInfo("w2", "host2", 50052, 4, status="idle", tasks_failed=0),
            WorkerInfo("w3", "host3", 50052, 2, status="busy", tasks_failed=0),
        ]
        
        space = SearchSpace("test")
        space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))
        task = Task("task-1", space.sample())
        
        worker = scheduler.assign_task(task, workers)
        
        assert worker.worker_id == "w2"  # Idle with no failures


class TestWorkStealingScheduler:
    """Test work stealing scheduler."""
    
    def test_no_stealing_below_threshold(self) -> None:
        """Test no stealing when below threshold."""
        scheduler = WorkStealingScheduler(steal_threshold=2)
        
        idle_worker = WorkerInfo("w1", "host1", 50052, 2, status="idle")
        
        workers = [
            idle_worker,
            WorkerInfo("w2", "host2", 50052, 2, status="busy"),
        ]
        
        # w2 has only 1 task (below threshold)
        scheduler.worker_queues["w2"] = []
        
        stolen = scheduler.steal_task(idle_worker, workers)
        
        assert stolen is None
    
    def test_stealing_from_busiest(self) -> None:
        """Test stealing from busiest worker."""
        scheduler = WorkStealingScheduler(steal_threshold=2)
        
        space = SearchSpace("test")
        space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))
        
        tasks = [Task(f"task-{i}", space.sample()) for i in range(5)]
        
        idle_worker = WorkerInfo("w1", "host1", 50052, 2, status="idle")
        busy_worker = WorkerInfo("w2", "host2", 50052, 2, status="busy")
        
        # w2 has many tasks
        from collections import deque
        scheduler.worker_queues["w2"] = deque(tasks)
        
        stolen = scheduler.steal_task(idle_worker, [idle_worker, busy_worker])
        
        assert stolen is not None
        assert stolen in tasks
        assert len(scheduler.worker_queues["w2"]) == 4
        assert scheduler.steals == 1


class TestAdaptiveScheduler:
    """Test adaptive scheduler."""
    
    def test_initial_assignment(self) -> None:
        """Test assignment for new workers."""
        scheduler = AdaptiveScheduler()
        
        workers = [
            WorkerInfo("w1", "host1", 50052, 2, status="idle"),
            WorkerInfo("w2", "host2", 50052, 4, status="idle"),
        ]
        
        space = SearchSpace("test")
        space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))
        task = Task("task-1", space.sample())
        
        worker = scheduler.assign_task(task, workers)
        
        # Should prefer worker with more GPUs initially
        assert worker.worker_id == "w2"
    
    def test_learning_from_history(self) -> None:
        """Test learning from completion history."""
        scheduler = AdaptiveScheduler(learning_rate=0.5)
        
        space = SearchSpace("test")
        space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))
        
        # Record fast completions for w1
        task = Task("task-1", space.sample())
        scheduler.record_completion("w1", task, duration=5.0, success=True)
        scheduler.record_completion("w1", task, duration=4.0, success=True)
        
        # Record slow completions for w2
        scheduler.record_completion("w2", task, duration=20.0, success=True)
        scheduler.record_completion("w2", task, duration=25.0, success=True)
        
        # Should prefer w1 (faster)
        workers = [
            WorkerInfo("w1", "host1", 50052, 2, status="idle"),
            WorkerInfo("w2", "host2", 50052, 2, status="idle"),
        ]
        
        worker = scheduler.assign_task(task, workers)
        assert worker.worker_id == "w1"
    
    def test_success_rate_impact(self) -> None:
        """Test impact of success rate on assignment."""
        scheduler = AdaptiveScheduler(learning_rate=0.5)
        
        space = SearchSpace("test")
        space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))
        task = Task("task-1", space.sample())
        
        # w1: Fast but unreliable
        scheduler.record_completion("w1", task, duration=5.0, success=False)
        scheduler.record_completion("w1", task, duration=5.0, success=False)
        
        # w2: Slower but reliable
        scheduler.record_completion("w2", task, duration=10.0, success=True)
        scheduler.record_completion("w2", task, duration=10.0, success=True)
        
        workers = [
            WorkerInfo("w1", "host1", 50052, 2, status="idle"),
            WorkerInfo("w2", "host2", 50052, 2, status="idle"),
        ]
        
        worker = scheduler.assign_task(task, workers)
        # Should prefer reliable w2
        assert worker.worker_id == "w2"


class TestRoundRobinScheduler:
    """Test round-robin scheduler."""
    
    def test_round_robin_assignment(self) -> None:
        """Test round-robin assignment."""
        scheduler = RoundRobinScheduler()
        
        workers = [
            WorkerInfo("w1", "host1", 50052, 2, status="idle"),
            WorkerInfo("w2", "host2", 50052, 2, status="idle"),
            WorkerInfo("w3", "host3", 50052, 2, status="idle"),
        ]
        
        space = SearchSpace("test")
        space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))
        
        # Assign 3 tasks
        tasks = [Task(f"task-{i}", space.sample()) for i in range(3)]
        
        assignments = []
        for task in tasks:
            worker = scheduler.assign_task(task, workers)
            assignments.append(worker.worker_id)
        
        # Should cycle through workers
        assert assignments == ["w1", "w2", "w3"]


def test_create_scheduler() -> None:
    """Test scheduler factory."""
    fifo = create_scheduler("fifo")
    assert isinstance(fifo, FIFOScheduler)
    
    priority = create_scheduler("priority")
    assert isinstance(priority, PriorityScheduler)
    
    lb = create_scheduler("load_balancing")
    assert isinstance(lb, LoadBalancingScheduler)
    
    ws = create_scheduler("work_stealing", steal_threshold=3)
    assert isinstance(ws, WorkStealingScheduler)
    assert ws.steal_threshold == 3
    
    adaptive = create_scheduler("adaptive", learning_rate=0.2)
    assert isinstance(adaptive, AdaptiveScheduler)
    assert adaptive.learning_rate == 0.2
    
    rr = create_scheduler("round_robin")
    assert isinstance(rr, RoundRobinScheduler)


def test_invalid_scheduler() -> None:
    """Test invalid scheduler name."""
    with pytest.raises(ValueError, match="Unknown scheduler strategy"):
        create_scheduler("invalid_strategy")


def test_scheduler_statistics() -> None:
    """Test scheduler statistics."""
    scheduler = FIFOScheduler()
    
    stats = scheduler.get_statistics()
    assert "assignments" in stats
    assert "strategy" in stats
    assert stats["strategy"] == "FIFO"
