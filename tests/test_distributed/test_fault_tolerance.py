"""Tests for fault tolerance mechanisms."""

import time

import pytest

from morphml.core.dsl import Layer, SearchSpace
from morphml.distributed import (
    CircuitBreaker,
    FailureType,
    FaultToleranceManager,
    Task,
    WorkerInfo,
)


class TestCircuitBreaker:
    """Test circuit breaker pattern."""
    
    def test_initial_state(self):
        """Test circuit breaker starts closed."""
        breaker = CircuitBreaker(failure_threshold=3, timeout=300)
        
        assert breaker.get_state() == "CLOSED"
        assert not breaker.is_open()
    
    def test_open_on_failures(self):
        """Test circuit opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3, timeout=300)
        
        # Record failures
        breaker.record_failure()
        assert breaker.get_state() == "CLOSED"
        
        breaker.record_failure()
        assert breaker.get_state() == "CLOSED"
        
        breaker.record_failure()
        assert breaker.get_state() == "OPEN"
        assert breaker.is_open()
    
    def test_success_resets_failure_count(self):
        """Test successes reduce failure count."""
        breaker = CircuitBreaker(failure_threshold=3, timeout=300)
        
        breaker.record_failure()
        breaker.record_failure()
        
        # Success should reset
        breaker.record_success()
        
        assert breaker.failure_count == 1
        assert not breaker.is_open()
    
    def test_half_open_to_closed(self):
        """Test transition from HALF_OPEN to CLOSED."""
        breaker = CircuitBreaker(
            failure_threshold=2, timeout=0.1, success_threshold=2
        )
        
        # Open circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open()
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Check - should be HALF_OPEN
        is_open = breaker.is_open()
        assert not is_open
        assert breaker.get_state() == "HALF_OPEN"
        
        # Record successes
        breaker.record_success()
        breaker.record_success()
        
        # Should close
        assert breaker.get_state() == "CLOSED"
    
    def test_half_open_reopens_on_failure(self):
        """Test HALF_OPEN reopens on failure."""
        breaker = CircuitBreaker(failure_threshold=2, timeout=0.1)
        
        # Open circuit
        breaker.record_failure()
        breaker.record_failure()
        
        # Wait for timeout
        time.sleep(0.2)
        breaker.is_open()  # Trigger transition to HALF_OPEN
        
        # Fail during testing
        breaker.record_failure()
        
        assert breaker.get_state() == "OPEN"


class TestFaultToleranceManager:
    """Test fault tolerance manager."""
    
    @pytest.fixture
    def manager(self):
        """Create fault tolerance manager."""
        return FaultToleranceManager(
            {"max_retries": 3, "retry_delay": 0.1, "circuit_breaker_threshold": 3}
        )
    
    @pytest.fixture
    def task(self):
        """Create test task."""
        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64),
            Layer.output(units=10),
        )
        return Task("task-1", space.sample())
    
    def test_task_retry(self, manager, task):
        """Test task retry logic."""
        # First 2 failures should retry
        assert manager.handle_task_failure(
            task, FailureType.NETWORK_ERROR, "Connection timeout"
        )
        assert manager.task_retries["task-1"] == 1
        
        assert manager.handle_task_failure(
            task, FailureType.NETWORK_ERROR, "Connection timeout"
        )
        assert manager.task_retries["task-1"] == 2
        
        # Third failure should still retry (max_retries=3)
        assert manager.handle_task_failure(
            task, FailureType.NETWORK_ERROR, "Connection timeout"
        )
        assert manager.task_retries["task-1"] == 3
        
        # Fourth failure should not retry
        assert not manager.handle_task_failure(
            task, FailureType.NETWORK_ERROR, "Connection timeout"
        )
        assert manager.task_retries["task-1"] == 4
        assert len(manager.failed_tasks) == 1
    
    def test_worker_failure_handling(self, manager):
        """Test worker failure handling."""
        worker_id = "worker-1"
        
        # Handle failure
        manager.handle_worker_failure(worker_id, FailureType.WORKER_CRASH)
        
        assert worker_id in manager.failed_workers
        assert worker_id in manager.worker_failures
        assert len(manager.worker_failures[worker_id]) == 1
    
    def test_worker_recovery(self, manager):
        """Test worker recovery."""
        worker_id = "worker-1"
        
        # Fail then recover
        manager.handle_worker_failure(worker_id)
        assert worker_id in manager.failed_workers
        
        manager.handle_worker_recovery(worker_id)
        assert worker_id not in manager.failed_workers
    
    def test_task_reassignment(self, manager):
        """Test task reassignment."""
        failed_worker = "worker-1"
        task_ids = ["task-1", "task-2", "task-3"]
        
        workers = [
            WorkerInfo("worker-2", "host2", 50052, 2, status="idle"),
            WorkerInfo("worker-3", "host3", 50052, 2, status="idle"),
        ]
        
        reassignment = manager.reassign_tasks(failed_worker, task_ids, workers)
        
        # All tasks should be reassigned
        assert len(reassignment) == 3
        
        # Should use different workers (round-robin)
        assert reassignment["task-1"] == "worker-2"
        assert reassignment["task-2"] == "worker-3"
        assert reassignment["task-3"] == "worker-2"
    
    def test_unhealthy_worker_detection(self, manager):
        """Test unhealthy worker detection."""
        worker_id = "worker-1"
        
        # Initially healthy
        assert not manager.is_worker_unhealthy(worker_id)
        
        # Mark as failed
        manager.handle_worker_failure(worker_id)
        assert manager.is_worker_unhealthy(worker_id)
    
    def test_circuit_breaker_integration(self, manager):
        """Test circuit breaker integration."""
        worker_id = "worker-1"
        
        # Trigger multiple failures
        for _ in range(4):
            manager.handle_worker_failure(worker_id)
        
        # Circuit should be open
        breaker = manager.circuit_breakers[worker_id]
        assert breaker.is_open()
        assert manager.is_worker_unhealthy(worker_id)
    
    def test_statistics(self, manager, task):
        """Test statistics collection."""
        # Generate some failures
        manager.handle_task_failure(task, FailureType.NETWORK_ERROR, "error")
        manager.handle_worker_failure("worker-1")
        
        stats = manager.get_statistics()
        
        assert stats["total_task_failures"] == 1
        assert stats["total_worker_failures"] == 1
        assert stats["tasks_with_retries"] == 1
        assert stats["failed_workers"] == 1
    
    def test_reset(self, manager, task):
        """Test state reset."""
        # Create some state
        manager.handle_task_failure(task, FailureType.NETWORK_ERROR, "error")
        manager.handle_worker_failure("worker-1")
        
        # Reset
        manager.reset()
        
        assert len(manager.task_retries) == 0
        assert len(manager.failed_tasks) == 0
        assert len(manager.failed_workers) == 0


def test_fault_tolerance_imports():
    """Test that fault tolerance classes can be imported."""
    from morphml.distributed import (
        CircuitBreaker,
        FailureEvent,
        FailureType,
        FaultToleranceManager,
    )
    
    # Verify classes exist
    assert FaultToleranceManager is not None
    assert CircuitBreaker is not None
    assert FailureType is not None
    assert FailureEvent is not None
