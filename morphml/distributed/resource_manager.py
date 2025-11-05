"""Resource management for distributed workers.

Tracks and manages computational resources across worker nodes:
- GPU availability and memory
- CPU utilization
- Memory usage
- Task placement based on requirements

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from morphml.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class WorkerResources:
    """
    Worker computational resources.
    
    Tracks available and total resources for a worker node.
    
    Attributes:
        worker_id: Unique worker identifier
        total_gpus: Total number of GPUs
        available_gpus: Number of available GPUs
        gpu_memory_total: Total GPU memory per GPU (GB)
        gpu_memory_available: Available GPU memory (GB)
        cpu_percent: CPU utilization percentage
        memory_percent: RAM utilization percentage
        network_bandwidth: Network bandwidth (Mbps)
    """
    
    worker_id: str
    total_gpus: int = 0
    available_gpus: int = 0
    gpu_memory_total: float = 0.0  # GB
    gpu_memory_available: float = 0.0  # GB
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    network_bandwidth: float = 1000.0  # Mbps
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def gpu_utilization(self) -> float:
        """Calculate GPU utilization percentage."""
        if self.total_gpus == 0:
            return 0.0
        return (1.0 - self.available_gpus / self.total_gpus) * 100
    
    @property
    def gpu_memory_utilization(self) -> float:
        """Calculate GPU memory utilization percentage."""
        if self.gpu_memory_total == 0:
            return 0.0
        return (
            1.0 - self.gpu_memory_available / self.gpu_memory_total
        ) * 100
    
    def can_run_task(self, requirements: "TaskRequirements") -> bool:
        """
        Check if worker can run task with given requirements.
        
        Args:
            requirements: Task resource requirements
        
        Returns:
            True if worker has sufficient resources
        """
        return (
            self.available_gpus >= requirements.min_gpus
            and self.gpu_memory_available >= requirements.min_gpu_memory
            and self.memory_percent < 90.0  # Don't overload memory
        )
    
    def allocate(self, requirements: "TaskRequirements") -> bool:
        """
        Allocate resources for task.
        
        Args:
            requirements: Task requirements
        
        Returns:
            True if allocation successful
        """
        if not self.can_run_task(requirements):
            return False
        
        self.available_gpus -= requirements.min_gpus
        self.gpu_memory_available -= requirements.min_gpu_memory
        
        return True
    
    def release(self, requirements: "TaskRequirements") -> None:
        """
        Release allocated resources.
        
        Args:
            requirements: Task requirements to release
        """
        self.available_gpus = min(
            self.total_gpus, self.available_gpus + requirements.min_gpus
        )
        self.gpu_memory_available = min(
            self.gpu_memory_total,
            self.gpu_memory_available + requirements.min_gpu_memory,
        )


@dataclass
class TaskRequirements:
    """
    Task resource requirements.
    
    Specifies minimum resources needed to execute a task.
    
    Attributes:
        min_gpus: Minimum number of GPUs
        min_gpu_memory: Minimum GPU memory per GPU (GB)
        min_cpu_cores: Minimum CPU cores
        min_memory: Minimum RAM (GB)
        estimated_time: Estimated execution time (seconds)
        priority: Task priority (higher = more important)
    """
    
    min_gpus: int = 1
    min_gpu_memory: float = 2.0  # GB
    min_cpu_cores: int = 1
    min_memory: float = 4.0  # GB
    estimated_time: float = 300.0  # seconds
    priority: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate requirements."""
        if self.min_gpus < 0:
            raise ValueError("min_gpus must be >= 0")
        if self.min_gpu_memory < 0:
            raise ValueError("min_gpu_memory must be >= 0")
        if self.estimated_time < 0:
            raise ValueError("estimated_time must be >= 0")


class ResourceManager:
    """
    Manage computational resources across workers.
    
    Tracks resource availability and helps with intelligent task placement.
    
    Example:
        >>> manager = ResourceManager()
        >>> manager.update_resources('worker-1', {
        ...     'total_gpus': 4,
        ...     'available_gpus': 3,
        ...     'gpu_memory_total': 16.0,
        ...     'gpu_memory_available': 12.0,
        ... })
        >>> requirements = TaskRequirements(min_gpus=1, min_gpu_memory=4.0)
        >>> worker_id = manager.find_suitable_worker(requirements)
    """
    
    def __init__(self) -> None:
        """Initialize resource manager."""
        self.resources: Dict[str, WorkerResources] = {}
        self.allocation_history: List[Dict[str, Any]] = []
    
    def register_worker(
        self, worker_id: str, resources: Dict[str, Any]
    ) -> None:
        """
        Register worker with initial resources.
        
        Args:
            worker_id: Unique worker identifier
            resources: Initial resource state
        """
        self.resources[worker_id] = WorkerResources(
            worker_id=worker_id, **resources
        )
        
        logger.info(
            f"Registered worker {worker_id}: "
            f"{resources.get('total_gpus', 0)} GPUs, "
            f"{resources.get('gpu_memory_total', 0):.1f}GB GPU memory"
        )
    
    def update_resources(
        self, worker_id: str, resources: Dict[str, Any]
    ) -> None:
        """
        Update worker resource information.
        
        Args:
            worker_id: Worker identifier
            resources: Updated resource state
        """
        if worker_id not in self.resources:
            self.register_worker(worker_id, resources)
        else:
            # Update existing
            for key, value in resources.items():
                if hasattr(self.resources[worker_id], key):
                    setattr(self.resources[worker_id], key, value)
    
    def find_suitable_worker(
        self, requirements: TaskRequirements, strategy: str = "best_fit"
    ) -> Optional[str]:
        """
        Find worker that meets task requirements.
        
        Args:
            requirements: Task resource requirements
            strategy: Placement strategy
                - 'first_fit': First worker that fits
                - 'best_fit': Worker with least excess capacity
                - 'worst_fit': Worker with most excess capacity
        
        Returns:
            worker_id or None if no suitable worker
        """
        suitable_workers = [
            (wid, res)
            for wid, res in self.resources.items()
            if res.can_run_task(requirements)
        ]
        
        if not suitable_workers:
            logger.debug("No suitable worker found for task requirements")
            return None
        
        # Apply placement strategy
        if strategy == "first_fit":
            return suitable_workers[0][0]
        
        elif strategy == "best_fit":
            # Worker with least excess GPUs
            best = min(
                suitable_workers,
                key=lambda x: x[1].available_gpus - requirements.min_gpus,
            )
            return best[0]
        
        elif strategy == "worst_fit":
            # Worker with most excess GPUs (best for load balancing)
            worst = max(
                suitable_workers,
                key=lambda x: x[1].available_gpus,
            )
            return worst[0]
        
        else:
            raise ValueError(f"Unknown placement strategy: {strategy}")
    
    def find_all_suitable_workers(
        self, requirements: TaskRequirements
    ) -> List[str]:
        """
        Find all workers that meet requirements.
        
        Args:
            requirements: Task requirements
        
        Returns:
            List of suitable worker IDs
        """
        return [
            wid
            for wid, res in self.resources.items()
            if res.can_run_task(requirements)
        ]
    
    def allocate_resources(
        self, worker_id: str, requirements: TaskRequirements
    ) -> bool:
        """
        Allocate resources for task on worker.
        
        Args:
            worker_id: Worker to allocate on
            requirements: Resource requirements
        
        Returns:
            True if allocation successful
        """
        if worker_id not in self.resources:
            logger.error(f"Unknown worker: {worker_id}")
            return False
        
        success = self.resources[worker_id].allocate(requirements)
        
        if success:
            self.allocation_history.append(
                {
                    "worker_id": worker_id,
                    "gpus": requirements.min_gpus,
                    "memory": requirements.min_gpu_memory,
                }
            )
            
            logger.debug(
                f"Allocated resources on {worker_id}: "
                f"{requirements.min_gpus} GPUs, "
                f"{requirements.min_gpu_memory:.1f}GB memory"
            )
        
        return success
    
    def release_resources(
        self, worker_id: str, requirements: TaskRequirements
    ) -> None:
        """
        Release allocated resources.
        
        Args:
            worker_id: Worker to release resources from
            requirements: Resource requirements to release
        """
        if worker_id in self.resources:
            self.resources[worker_id].release(requirements)
            
            logger.debug(
                f"Released resources on {worker_id}: "
                f"{requirements.min_gpus} GPUs, "
                f"{requirements.min_gpu_memory:.1f}GB memory"
            )
    
    def get_total_resources(self) -> Dict[str, Any]:
        """
        Get total resources across all workers.
        
        Returns:
            Dictionary with aggregate resource statistics
        """
        total_gpus = sum(r.total_gpus for r in self.resources.values())
        available_gpus = sum(r.available_gpus for r in self.resources.values())
        total_gpu_memory = sum(
            r.gpu_memory_total for r in self.resources.values()
        )
        available_gpu_memory = sum(
            r.gpu_memory_available for r in self.resources.values()
        )
        
        return {
            "total_workers": len(self.resources),
            "total_gpus": total_gpus,
            "available_gpus": available_gpus,
            "gpu_utilization": (
                (1.0 - available_gpus / total_gpus) * 100
                if total_gpus > 0
                else 0.0
            ),
            "total_gpu_memory": total_gpu_memory,
            "available_gpu_memory": available_gpu_memory,
            "memory_utilization": (
                (1.0 - available_gpu_memory / total_gpu_memory) * 100
                if total_gpu_memory > 0
                else 0.0
            ),
        }
    
    def get_worker_resources(self, worker_id: str) -> Optional[WorkerResources]:
        """
        Get resources for specific worker.
        
        Args:
            worker_id: Worker identifier
        
        Returns:
            WorkerResources or None if worker not found
        """
        return self.resources.get(worker_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get resource manager statistics."""
        stats = self.get_total_resources()
        stats["allocations"] = len(self.allocation_history)
        stats["worker_details"] = {
            wid: {
                "total_gpus": res.total_gpus,
                "available_gpus": res.available_gpus,
                "gpu_utilization": res.gpu_utilization,
                "cpu_percent": res.cpu_percent,
                "memory_percent": res.memory_percent,
            }
            for wid, res in self.resources.items()
        }
        return stats


class GPUAffinityManager:
    """
    Manage GPU affinity for tasks.
    
    Ensures tasks are pinned to specific GPUs for better performance.
    """
    
    def __init__(self) -> None:
        """Initialize GPU affinity manager."""
        self.gpu_assignments: Dict[str, List[int]] = {}
    
    def assign_gpus(
        self, worker_id: str, task_id: str, gpu_ids: List[int]
    ) -> None:
        """
        Assign specific GPUs to task.
        
        Args:
            worker_id: Worker identifier
            task_id: Task identifier
            gpu_ids: List of GPU IDs to assign
        """
        key = f"{worker_id}:{task_id}"
        self.gpu_assignments[key] = gpu_ids
        
        logger.debug(f"Assigned GPUs {gpu_ids} to task {task_id} on {worker_id}")
    
    def get_assigned_gpus(
        self, worker_id: str, task_id: str
    ) -> Optional[List[int]]:
        """
        Get assigned GPUs for task.
        
        Args:
            worker_id: Worker identifier
            task_id: Task identifier
        
        Returns:
            List of GPU IDs or None
        """
        key = f"{worker_id}:{task_id}"
        return self.gpu_assignments.get(key)
    
    def release_gpus(self, worker_id: str, task_id: str) -> None:
        """
        Release GPU assignment.
        
        Args:
            worker_id: Worker identifier
            task_id: Task identifier
        """
        key = f"{worker_id}:{task_id}"
        if key in self.gpu_assignments:
            del self.gpu_assignments[key]
            logger.debug(f"Released GPU assignment for task {task_id}")
