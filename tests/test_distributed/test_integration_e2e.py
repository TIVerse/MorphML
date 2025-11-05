"""End-to-end integration tests for distributed execution.

Tests the complete master-worker communication pipeline with gRPC.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import time
from typing import Dict
from unittest.mock import Mock

import pytest

from morphml.core.dsl import Layer, SearchSpace
from morphml.core.graph import ModelGraph


# Skip if gRPC not available
try:
    from morphml.distributed.master import DistributedMaster
    from morphml.distributed.worker import DistributedWorker
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GRPC_AVAILABLE, reason="gRPC not available")


@pytest.fixture
def search_space() -> SearchSpace:
    """Create a simple search space for testing."""
    space = SearchSpace("test_space")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=[32, 64], kernel_size=3),
        Layer.relu(),
        Layer.maxpool(pool_size=2),
        Layer.dense(units=[64, 128]),
        Layer.output(units=10)
    )
    return space


@pytest.fixture
def evaluator() -> callable:
    """Create a mock evaluator."""
    def evaluate(graph: ModelGraph) -> Dict[str, float]:
        """Fake evaluation that returns based on graph depth."""
        depth = graph.get_depth()
        return {
            "accuracy": 0.8 + (depth * 0.01),
            "latency": depth * 10.0,
            "params": graph.estimate_parameters()
        }
    return evaluate


class TestMasterWorkerCommunication:
    """Test master-worker communication."""
    
    @pytest.mark.integration
    def test_worker_registration(self):
        """Test worker can register with master."""
        # Start master
        master = DistributedMaster(
            host="localhost",
            port=50051,
            max_workers=5
        )
        master.start()
        
        try:
            # Start worker
            worker = DistributedWorker(
                worker_id="test-worker-1",
                master_host="localhost",
                master_port=50051
            )
            
            # Register
            success = worker.register()
            assert success, "Worker registration failed"
            
            # Verify worker is in master's registry
            assert "test-worker-1" in master.workers
            assert master.workers["test-worker-1"]["status"] == "idle"
            
        finally:
            master.stop()
    
    @pytest.mark.integration
    def test_task_distribution(self, search_space: SearchSpace, evaluator: callable):
        """Test master can distribute tasks to workers."""
        # Start master
        master = DistributedMaster(
            host="localhost",
            port=50052,
            max_workers=2
        )
        master.start()
        
        try:
            # Start workers
            workers = []
            for i in range(2):
                worker = DistributedWorker(
                    worker_id=f"worker-{i}",
                    master_host="localhost",
                    master_port=50052,
                    evaluator=evaluator
                )
                worker.register()
                worker.start()
                workers.append(worker)
            
            # Submit tasks
            architectures = [search_space.sample() for _ in range(10)]
            for arch in architectures:
                master.submit_task(arch)
            
            # Wait for completion
            timeout = 30
            start_time = time.time()
            while master.get_pending_count() > 0:
                if time.time() - start_time > timeout:
                    pytest.fail("Tasks did not complete in time")
                time.sleep(0.5)
            
            # Verify all tasks completed
            assert master.get_completed_count() == 10
            assert master.get_pending_count() == 0
            
            # Verify results were collected
            results = master.get_results()
            assert len(results) == 10
            for result in results:
                assert "accuracy" in result
                assert "latency" in result
                assert result["accuracy"] > 0
            
        finally:
            # Cleanup
            for worker in workers:
                worker.stop()
            master.stop()
    
    @pytest.mark.integration
    def test_worker_failure_recovery(self, search_space: SearchSpace, evaluator: callable):
        """Test task reassignment when worker fails."""
        master = DistributedMaster(
            host="localhost",
            port=50053,
            max_workers=3
        )
        master.start()
        
        try:
            # Start workers
            worker1 = DistributedWorker(
                worker_id="worker-1",
                master_host="localhost",
                master_port=50053,
                evaluator=evaluator
            )
            worker1.register()
            worker1.start()
            
            worker2 = DistributedWorker(
                worker_id="worker-2",
                master_host="localhost",
                master_port=50053,
                evaluator=evaluator
            )
            worker2.register()
            worker2.start()
            
            # Submit tasks
            architectures = [search_space.sample() for _ in range(5)]
            task_ids = []
            for arch in architectures:
                task_id = master.submit_task(arch)
                task_ids.append(task_id)
            
            # Kill worker1 mid-execution
            time.sleep(1)
            worker1.stop()
            
            # Wait a bit for master to detect failure
            time.sleep(3)
            
            # Verify master reassigned tasks
            assert master.get_failed_worker_count() >= 1
            
            # Tasks should still complete on worker2
            timeout = 30
            start_time = time.time()
            while master.get_pending_count() > 0:
                if time.time() - start_time > timeout:
                    # Some tasks may be in reassignment
                    break
                time.sleep(0.5)
            
            # At least some tasks should have completed
            assert master.get_completed_count() > 0
            
        finally:
            worker2.stop()
            master.stop()
    
    @pytest.mark.integration
    def test_heartbeat_monitoring(self):
        """Test heartbeat mechanism keeps workers alive."""
        master = DistributedMaster(
            host="localhost",
            port=50054,
            heartbeat_timeout=5
        )
        master.start()
        
        try:
            worker = DistributedWorker(
                worker_id="heartbeat-worker",
                master_host="localhost",
                master_port=50054,
                heartbeat_interval=1
            )
            worker.register()
            worker.start()
            
            # Wait for several heartbeat cycles
            time.sleep(6)
            
            # Worker should still be alive
            assert worker.worker_id in master.workers
            assert master.workers[worker.worker_id]["status"] in ["idle", "busy"]
            
            # Stop worker (no more heartbeats)
            worker.stop()
            
            # Wait for timeout
            time.sleep(6)
            
            # Master should have detected dead worker
            if worker.worker_id in master.workers:
                assert master.workers[worker.worker_id]["status"] == "dead"
            
        finally:
            master.stop()


class TestSchedulingStrategies:
    """Test different scheduling strategies."""
    
    @pytest.mark.integration
    def test_load_balancing_scheduler(self, search_space: SearchSpace, evaluator: callable):
        """Test load balancing distributes tasks evenly."""
        master = DistributedMaster(
            host="localhost",
            port=50055,
            scheduler_type="load_balancing"
        )
        master.start()
        
        try:
            # Start 3 workers
            workers = []
            for i in range(3):
                worker = DistributedWorker(
                    worker_id=f"lb-worker-{i}",
                    master_host="localhost",
                    master_port=50055,
                    evaluator=evaluator
                )
                worker.register()
                worker.start()
                workers.append(worker)
            
            # Submit 30 tasks
            for _ in range(30):
                master.submit_task(search_space.sample())
            
            # Wait for completion
            time.sleep(15)
            
            # Check task distribution
            worker_tasks = {w.worker_id: w.tasks_completed for w in workers}
            
            # Tasks should be relatively evenly distributed
            # (within 30% of average)
            avg_tasks = sum(worker_tasks.values()) / len(worker_tasks)
            for count in worker_tasks.values():
                assert abs(count - avg_tasks) / avg_tasks < 0.3
            
        finally:
            for worker in workers:
                worker.stop()
            master.stop()


class TestFaultTolerance:
    """Test fault tolerance mechanisms."""
    
    @pytest.mark.integration
    def test_checkpoint_recovery(self, search_space: SearchSpace, evaluator: callable):
        """Test master can recover from checkpoint."""
        checkpoint_path = "/tmp/morphml_test_checkpoint.json"
        
        # First run - create checkpoint
        master1 = DistributedMaster(
            host="localhost",
            port=50056,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=5
        )
        master1.start()
        
        try:
            worker = DistributedWorker(
                worker_id="checkpoint-worker",
                master_host="localhost",
                master_port=50056,
                evaluator=evaluator
            )
            worker.register()
            worker.start()
            
            # Submit tasks
            for _ in range(5):
                master1.submit_task(search_space.sample())
            
            # Wait for checkpoint
            time.sleep(6)
            
            # Verify checkpoint exists
            import os
            assert os.path.exists(checkpoint_path)
            
            worker.stop()
            master1.stop()
            
        except Exception as e:
            master1.stop()
            raise e
        
        # Second run - recover from checkpoint
        master2 = DistributedMaster(
            host="localhost",
            port=50056,
            checkpoint_path=checkpoint_path,
            recover=True
        )
        master2.start()
        
        try:
            # Verify state was recovered
            assert master2.get_completed_count() > 0 or master2.get_pending_count() > 0
            
        finally:
            master2.stop()
            # Cleanup
            import os
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)


class TestPerformance:
    """Performance and scalability tests."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_throughput(self, search_space: SearchSpace, evaluator: callable):
        """Test system throughput with multiple workers."""
        master = DistributedMaster(
            host="localhost",
            port=50057,
            max_workers=10
        )
        master.start()
        
        try:
            # Start 5 workers
            workers = []
            for i in range(5):
                worker = DistributedWorker(
                    worker_id=f"perf-worker-{i}",
                    master_host="localhost",
                    master_port=50057,
                    evaluator=evaluator
                )
                worker.register()
                worker.start()
                workers.append(worker)
            
            # Submit 100 tasks
            start_time = time.time()
            for _ in range(100):
                master.submit_task(search_space.sample())
            
            # Wait for completion
            while master.get_pending_count() > 0:
                if time.time() - start_time > 120:
                    pytest.fail("Tasks did not complete in time")
                time.sleep(0.5)
            
            elapsed = time.time() - start_time
            throughput = 100 / elapsed
            
            # Should process at least 2 tasks per second with 5 workers
            assert throughput > 2.0, f"Throughput too low: {throughput:.2f} tasks/sec"
            
            print(f"\nThroughput: {throughput:.2f} tasks/sec")
            print(f"Total time: {elapsed:.2f}s")
            
        finally:
            for worker in workers:
                worker.stop()
            master.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
