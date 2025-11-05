"""Tests for resource manager."""

import pytest

from morphml.distributed import (
    GPUAffinityManager,
    ResourceManager,
    TaskRequirements,
    WorkerResources,
)


class TestWorkerResources:
    """Test WorkerResources dataclass."""

    def test_creation(self) -> None:
        """Test creating WorkerResources."""
        resources = WorkerResources(
            worker_id="w1",
            total_gpus=4,
            available_gpus=3,
            gpu_memory_total=64.0,
            gpu_memory_available=48.0,
        )

        assert resources.worker_id == "w1"
        assert resources.total_gpus == 4
        assert resources.available_gpus == 3

    def test_gpu_utilization(self) -> None:
        """Test GPU utilization calculation."""
        resources = WorkerResources(worker_id="w1", total_gpus=4, available_gpus=3)

        assert resources.gpu_utilization == 25.0  # 1/4 = 25%

    def test_gpu_memory_utilization(self) -> None:
        """Test GPU memory utilization."""
        resources = WorkerResources(
            worker_id="w1",
            gpu_memory_total=16.0,
            gpu_memory_available=12.0,
        )

        assert resources.gpu_memory_utilization == 25.0  # 4/16 = 25%

    def test_can_run_task(self) -> None:
        """Test task feasibility check."""
        resources = WorkerResources(
            worker_id="w1",
            total_gpus=4,
            available_gpus=2,
            gpu_memory_total=16.0,
            gpu_memory_available=10.0,
            memory_percent=50.0,
        )

        # Feasible task
        req1 = TaskRequirements(min_gpus=1, min_gpu_memory=5.0)
        assert resources.can_run_task(req1)

        # Too many GPUs
        req2 = TaskRequirements(min_gpus=3, min_gpu_memory=5.0)
        assert not resources.can_run_task(req2)

        # Too much memory
        req3 = TaskRequirements(min_gpus=1, min_gpu_memory=12.0)
        assert not resources.can_run_task(req3)

        # Memory overload
        resources.memory_percent = 95.0
        assert not resources.can_run_task(req1)

    def test_allocate_and_release(self) -> None:
        """Test resource allocation and release."""
        resources = WorkerResources(
            worker_id="w1",
            total_gpus=4,
            available_gpus=4,
            gpu_memory_total=16.0,
            gpu_memory_available=16.0,
        )

        req = TaskRequirements(min_gpus=2, min_gpu_memory=8.0)

        # Allocate
        success = resources.allocate(req)
        assert success
        assert resources.available_gpus == 2
        assert resources.gpu_memory_available == 8.0

        # Release
        resources.release(req)
        assert resources.available_gpus == 4
        assert resources.gpu_memory_available == 16.0


class TestTaskRequirements:
    """Test TaskRequirements dataclass."""

    def test_creation(self) -> None:
        """Test creating TaskRequirements."""
        req = TaskRequirements(
            min_gpus=2,
            min_gpu_memory=8.0,
            estimated_time=300.0,
            priority=0.9,
        )

        assert req.min_gpus == 2
        assert req.min_gpu_memory == 8.0
        assert req.estimated_time == 300.0
        assert req.priority == 0.9

    def test_validation(self) -> None:
        """Test requirement validation."""
        with pytest.raises(ValueError):
            TaskRequirements(min_gpus=-1)

        with pytest.raises(ValueError):
            TaskRequirements(min_gpu_memory=-5.0)

        with pytest.raises(ValueError):
            TaskRequirements(estimated_time=-10.0)


class TestResourceManager:
    """Test ResourceManager."""

    def test_register_worker(self) -> None:
        """Test worker registration."""
        manager = ResourceManager()

        manager.register_worker(
            "w1",
            {
                "total_gpus": 4,
                "available_gpus": 4,
                "gpu_memory_total": 16.0,
                "gpu_memory_available": 16.0,
            },
        )

        assert "w1" in manager.resources
        assert manager.resources["w1"].total_gpus == 4

    def test_update_resources(self) -> None:
        """Test resource updates."""
        manager = ResourceManager()

        manager.register_worker(
            "w1",
            {"total_gpus": 4, "available_gpus": 4},
        )

        # Update
        manager.update_resources("w1", {"available_gpus": 2, "cpu_percent": 75.0})

        assert manager.resources["w1"].available_gpus == 2
        assert manager.resources["w1"].cpu_percent == 75.0

    def test_find_suitable_worker_first_fit(self) -> None:
        """Test first-fit worker selection."""
        manager = ResourceManager()

        manager.register_worker(
            "w1", {"total_gpus": 2, "available_gpus": 1, "gpu_memory_available": 8.0}
        )
        manager.register_worker(
            "w2", {"total_gpus": 4, "available_gpus": 4, "gpu_memory_available": 16.0}
        )

        req = TaskRequirements(min_gpus=2, min_gpu_memory=10.0)

        worker_id = manager.find_suitable_worker(req, strategy="first_fit")

        # w1 can't handle it (only 1 GPU), should get w2
        assert worker_id == "w2"

    def test_find_suitable_worker_best_fit(self) -> None:
        """Test best-fit worker selection."""
        manager = ResourceManager()

        manager.register_worker(
            "w1", {"total_gpus": 2, "available_gpus": 2, "gpu_memory_available": 8.0}
        )
        manager.register_worker(
            "w2", {"total_gpus": 8, "available_gpus": 8, "gpu_memory_available": 32.0}
        )

        req = TaskRequirements(min_gpus=2, min_gpu_memory=6.0)

        worker_id = manager.find_suitable_worker(req, strategy="best_fit")

        # Should pick w1 (exact fit, no waste)
        assert worker_id == "w1"

    def test_find_suitable_worker_worst_fit(self) -> None:
        """Test worst-fit worker selection."""
        manager = ResourceManager()

        manager.register_worker(
            "w1", {"total_gpus": 2, "available_gpus": 2, "gpu_memory_available": 8.0}
        )
        manager.register_worker(
            "w2", {"total_gpus": 8, "available_gpus": 8, "gpu_memory_available": 32.0}
        )

        req = TaskRequirements(min_gpus=2, min_gpu_memory=6.0)

        worker_id = manager.find_suitable_worker(req, strategy="worst_fit")

        # Should pick w2 (most available resources)
        assert worker_id == "w2"

    def test_no_suitable_worker(self) -> None:
        """Test when no worker meets requirements."""
        manager = ResourceManager()

        manager.register_worker(
            "w1", {"total_gpus": 1, "available_gpus": 1, "gpu_memory_available": 4.0}
        )

        req = TaskRequirements(min_gpus=4, min_gpu_memory=16.0)

        worker_id = manager.find_suitable_worker(req)

        assert worker_id is None

    def test_find_all_suitable_workers(self) -> None:
        """Test finding all suitable workers."""
        manager = ResourceManager()

        manager.register_worker(
            "w1", {"total_gpus": 2, "available_gpus": 2, "gpu_memory_available": 8.0}
        )
        manager.register_worker(
            "w2", {"total_gpus": 4, "available_gpus": 4, "gpu_memory_available": 16.0}
        )
        manager.register_worker(
            "w3", {"total_gpus": 1, "available_gpus": 1, "gpu_memory_available": 4.0}
        )

        req = TaskRequirements(min_gpus=2, min_gpu_memory=6.0)

        suitable = manager.find_all_suitable_workers(req)

        assert set(suitable) == {"w1", "w2"}

    def test_allocate_and_release(self) -> None:
        """Test resource allocation and release."""
        manager = ResourceManager()

        manager.register_worker(
            "w1",
            {
                "total_gpus": 4,
                "available_gpus": 4,
                "gpu_memory_total": 16.0,
                "gpu_memory_available": 16.0,
            },
        )

        req = TaskRequirements(min_gpus=2, min_gpu_memory=8.0)

        # Allocate
        success = manager.allocate_resources("w1", req)
        assert success
        assert manager.resources["w1"].available_gpus == 2

        # Release
        manager.release_resources("w1", req)
        assert manager.resources["w1"].available_gpus == 4

    def test_get_total_resources(self) -> None:
        """Test getting total resources."""
        manager = ResourceManager()

        manager.register_worker(
            "w1",
            {
                "total_gpus": 2,
                "available_gpus": 1,
                "gpu_memory_total": 8.0,
                "gpu_memory_available": 4.0,
            },
        )
        manager.register_worker(
            "w2",
            {
                "total_gpus": 4,
                "available_gpus": 3,
                "gpu_memory_total": 16.0,
                "gpu_memory_available": 12.0,
            },
        )

        total = manager.get_total_resources()

        assert total["total_workers"] == 2
        assert total["total_gpus"] == 6
        assert total["available_gpus"] == 4
        assert total["gpu_utilization"] == pytest.approx(33.33, rel=0.1)

    def test_get_statistics(self) -> None:
        """Test getting statistics."""
        manager = ResourceManager()

        manager.register_worker("w1", {"total_gpus": 4, "available_gpus": 4})

        stats = manager.get_statistics()

        assert "total_workers" in stats
        assert "total_gpus" in stats
        assert "worker_details" in stats
        assert "w1" in stats["worker_details"]


class TestGPUAffinityManager:
    """Test GPU affinity manager."""

    def test_assign_and_get_gpus(self) -> None:
        """Test GPU assignment."""
        manager = GPUAffinityManager()

        manager.assign_gpus("w1", "task-1", [0, 1])

        gpus = manager.get_assigned_gpus("w1", "task-1")

        assert gpus == [0, 1]

    def test_release_gpus(self) -> None:
        """Test GPU release."""
        manager = GPUAffinityManager()

        manager.assign_gpus("w1", "task-1", [0, 1])
        manager.release_gpus("w1", "task-1")

        gpus = manager.get_assigned_gpus("w1", "task-1")

        assert gpus is None

    def test_multiple_tasks(self) -> None:
        """Test multiple task assignments."""
        manager = GPUAffinityManager()

        manager.assign_gpus("w1", "task-1", [0, 1])
        manager.assign_gpus("w1", "task-2", [2, 3])
        manager.assign_gpus("w2", "task-3", [0])

        assert manager.get_assigned_gpus("w1", "task-1") == [0, 1]
        assert manager.get_assigned_gpus("w1", "task-2") == [2, 3]
        assert manager.get_assigned_gpus("w2", "task-3") == [0]
