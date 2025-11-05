"""Task scheduling strategies for distributed execution.

Implements various scheduling algorithms:
- FIFO (First-In-First-Out)
- Priority-based scheduling
- Load balancing
- Work stealing
- Adaptive learning scheduler

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, Dict, List, Optional, Tuple

from morphml.distributed.master import Task, WorkerInfo
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class TaskScheduler(ABC):
    """
    Base class for task schedulers.
    
    A scheduler decides which worker should execute which task,
    optimizing for different objectives (throughput, fairness, etc.).
    """
    
    @abstractmethod
    def assign_task(
        self, task: Task, workers: List[WorkerInfo]
    ) -> Optional[WorkerInfo]:
        """
        Assign task to a worker.
        
        Args:
            task: Task to assign
            workers: Available workers
        
        Returns:
            Worker to assign task to, or None if no suitable worker
        """
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {}


class FIFOScheduler(TaskScheduler):
    """
    First-In-First-Out scheduler.
    
    Assigns tasks to the first available idle worker.
    Simple but effective for homogeneous workloads.
    
    Example:
        >>> scheduler = FIFOScheduler()
        >>> worker = scheduler.assign_task(task, workers)
    """
    
    def __init__(self) -> None:
        """Initialize FIFO scheduler."""
        self.assignments = 0
    
    def assign_task(
        self, task: Task, workers: List[WorkerInfo]
    ) -> Optional[WorkerInfo]:
        """Assign to first idle worker."""
        for worker in workers:
            if worker.is_available():
                self.assignments += 1
                return worker
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {"assignments": self.assignments, "strategy": "FIFO"}


class PriorityScheduler(TaskScheduler):
    """
    Priority-based scheduler.
    
    Tasks with higher priority are scheduled first.
    Useful for multi-fidelity optimization where promising
    architectures receive more computational resources.
    
    Args:
        max_queue_size: Maximum priority queue size
    
    Example:
        >>> scheduler = PriorityScheduler()
        >>> scheduler.enqueue(task, priority=0.95)
        >>> worker = scheduler.assign_task(task, workers)
    """
    
    def __init__(self, max_queue_size: int = 10000) -> None:
        """Initialize priority scheduler."""
        self.task_queue: PriorityQueue = PriorityQueue(maxsize=max_queue_size)
        self.assignments = 0
        self.task_priorities: Dict[str, float] = {}
    
    def enqueue(self, task: Task, priority: float) -> None:
        """
        Add task with priority.
        
        Args:
            task: Task to enqueue
            priority: Task priority (higher = more important)
        """
        # Negative priority for max-heap behavior
        self.task_queue.put((-priority, time.time(), task))
        self.task_priorities[task.task_id] = priority
        
        logger.debug(f"Enqueued task {task.task_id} with priority {priority:.4f}")
    
    def dequeue(self) -> Optional[Task]:
        """Get highest priority task."""
        if not self.task_queue.empty():
            _, _, task = self.task_queue.get()
            return task
        return None
    
    def assign_task(
        self, task: Task, workers: List[WorkerInfo]
    ) -> Optional[WorkerInfo]:
        """Assign to worker with least load."""
        idle_workers = [w for w in workers if w.is_available()]
        
        if idle_workers:
            # Assign to worker with least completed tasks
            worker = min(idle_workers, key=lambda w: w.tasks_completed)
            self.assignments += 1
            return worker
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "assignments": self.assignments,
            "queue_size": self.task_queue.qsize(),
            "strategy": "Priority",
        }


class LoadBalancingScheduler(TaskScheduler):
    """
    Load balancing scheduler.
    
    Distributes tasks evenly based on worker capacity and current load.
    Considers GPU count and task queue length.
    
    Example:
        >>> scheduler = LoadBalancingScheduler()
        >>> worker = scheduler.assign_task(task, workers)
    """
    
    def __init__(self) -> None:
        """Initialize load balancing scheduler."""
        self.assignments = 0
        self.worker_loads: Dict[str, float] = {}
    
    def assign_task(
        self, task: Task, workers: List[WorkerInfo]
    ) -> Optional[WorkerInfo]:
        """Assign to least loaded worker."""
        available_workers = [w for w in workers if w.status != "dead"]
        
        if not available_workers:
            return None
        
        # Calculate load for each worker
        worker_loads = {
            w.worker_id: self._calculate_load(w) for w in available_workers
        }
        
        # Assign to least loaded
        min_worker_id = min(worker_loads, key=worker_loads.get)  # type: ignore
        worker = next(w for w in available_workers if w.worker_id == min_worker_id)
        
        self.assignments += 1
        self.worker_loads = worker_loads
        
        logger.debug(
            f"Assigned task {task.task_id} to {worker.worker_id} "
            f"(load: {worker_loads[min_worker_id]:.2f})"
        )
        
        return worker
    
    def _calculate_load(self, worker: WorkerInfo) -> float:
        """
        Calculate worker load score.
        
        Load = (running_tasks / num_gpus) + idle_penalty
        
        Args:
            worker: Worker to calculate load for
        
        Returns:
            Load score (lower = less loaded)
        """
        if worker.num_gpus == 0:
            return float("inf")
        
        # Current task load
        running_load = (
            1.0 if worker.status == "busy" else 0.0
        ) / worker.num_gpus
        
        # Penalize failed tasks
        failure_penalty = worker.tasks_failed * 0.1
        
        # Total load
        load = running_load + failure_penalty
        
        return load
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "assignments": self.assignments,
            "worker_loads": dict(self.worker_loads),
            "strategy": "LoadBalancing",
        }


class WorkStealingScheduler(TaskScheduler):
    """
    Work stealing scheduler.
    
    Idle workers can steal tasks from busy workers' queues.
    Improves load balancing for heterogeneous workloads.
    
    Args:
        steal_threshold: Minimum queue length to allow stealing
        max_steal_attempts: Maximum steal attempts per cycle
    
    Example:
        >>> scheduler = WorkStealingScheduler(steal_threshold=2)
        >>> task = scheduler.steal_task(idle_worker, all_workers)
    """
    
    def __init__(
        self, steal_threshold: int = 2, max_steal_attempts: int = 3
    ) -> None:
        """Initialize work stealing scheduler."""
        self.steal_threshold = steal_threshold
        self.max_steal_attempts = max_steal_attempts
        self.assignments = 0
        self.steals = 0
        self.worker_queues: Dict[str, deque] = {}
    
    def assign_task(
        self, task: Task, workers: List[WorkerInfo]
    ) -> Optional[WorkerInfo]:
        """Assign task to least loaded worker."""
        idle_workers = [w for w in workers if w.is_available()]
        
        if idle_workers:
            worker = idle_workers[0]
            
            # Add to worker's queue
            if worker.worker_id not in self.worker_queues:
                self.worker_queues[worker.worker_id] = deque()
            
            self.worker_queues[worker.worker_id].append(task)
            self.assignments += 1
            
            return worker
        
        return None
    
    def steal_task(
        self, idle_worker: WorkerInfo, workers: List[WorkerInfo]
    ) -> Optional[Task]:
        """
        Idle worker steals task from busiest worker.
        
        Args:
            idle_worker: Worker looking for work
            workers: All workers
        
        Returns:
            Stolen task or None
        """
        # Find workers with tasks above threshold
        busy_workers = [
            w
            for w in workers
            if w.worker_id in self.worker_queues
            and len(self.worker_queues[w.worker_id]) > self.steal_threshold
        ]
        
        if not busy_workers:
            return None
        
        # Steal from busiest
        busiest = max(
            busy_workers, key=lambda w: len(self.worker_queues[w.worker_id])
        )
        
        queue = self.worker_queues[busiest.worker_id]
        
        if queue:
            # Steal from end (LIFO for better locality)
            stolen_task = queue.pop()
            self.steals += 1
            
            logger.info(
                f"Worker {idle_worker.worker_id} stole task {stolen_task.task_id} "
                f"from {busiest.worker_id}"
            )
            
            return stolen_task
        
        return None
    
    def remove_task(self, worker_id: str, task: Task) -> None:
        """Remove completed task from queue."""
        if worker_id in self.worker_queues:
            try:
                self.worker_queues[worker_id].remove(task)
            except ValueError:
                pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        queue_lengths = {
            wid: len(queue) for wid, queue in self.worker_queues.items()
        }
        
        return {
            "assignments": self.assignments,
            "steals": self.steals,
            "queue_lengths": queue_lengths,
            "strategy": "WorkStealing",
        }


class AdaptiveScheduler(TaskScheduler):
    """
    Adaptive scheduler using performance history.
    
    Learns optimal assignment policy based on:
    - Worker performance history
    - Task characteristics
    - System state
    
    Uses exponential moving average to track worker performance.
    
    Args:
        learning_rate: Learning rate for EMA (0-1)
    
    Example:
        >>> scheduler = AdaptiveScheduler(learning_rate=0.1)
        >>> worker = scheduler.assign_task(task, workers)
        >>> scheduler.record_completion(worker.worker_id, task, duration=15.2, success=True)
    """
    
    def __init__(self, learning_rate: float = 0.1) -> None:
        """Initialize adaptive scheduler."""
        self.learning_rate = learning_rate
        self.worker_performance: Dict[str, PerformanceStats] = {}
        self.assignments = 0
        self.history: List[Tuple[str, Task, float, bool]] = []
    
    def assign_task(
        self, task: Task, workers: List[WorkerInfo]
    ) -> Optional[WorkerInfo]:
        """
        Assign task using learned policy.
        
        Computes assignment score based on:
        - Worker speed (inverse of avg completion time)
        - Worker success rate
        - Worker availability
        """
        available_workers = [w for w in workers if w.is_available()]
        
        if not available_workers:
            return None
        
        # Compute scores for each worker
        scores = {}
        for worker in available_workers:
            score = self._compute_assignment_score(task, worker)
            scores[worker.worker_id] = score
        
        # Assign to best worker
        best_worker_id = max(scores, key=scores.get)  # type: ignore
        worker = next(w for w in available_workers if w.worker_id == best_worker_id)
        
        self.assignments += 1
        
        logger.debug(
            f"Assigned task {task.task_id} to {worker.worker_id} "
            f"(score: {scores[best_worker_id]:.4f})"
        )
        
        return worker
    
    def _compute_assignment_score(
        self, task: Task, worker: WorkerInfo
    ) -> float:
        """
        Compute assignment score for worker.
        
        Higher score = better assignment.
        """
        perf = self.worker_performance.get(worker.worker_id)
        
        if perf is None:
            # New worker: default score based on capacity
            return float(worker.num_gpus)
        
        # Speed score (inverse of completion time)
        speed_score = 1.0 / (perf.avg_completion_time + 1e-6)
        
        # Success score
        success_score = perf.success_rate
        
        # GPU capacity bonus
        capacity_bonus = worker.num_gpus / 4.0  # Normalize to typical 4 GPUs
        
        # Combined score
        score = (speed_score * 0.5 + success_score * 0.3 + capacity_bonus * 0.2)
        
        return score
    
    def record_completion(
        self, worker_id: str, task: Task, duration: float, success: bool
    ) -> None:
        """
        Record task completion for learning.
        
        Args:
            worker_id: Worker that completed task
            task: Completed task
            duration: Execution duration (seconds)
            success: Whether task succeeded
        """
        # Initialize stats if needed
        if worker_id not in self.worker_performance:
            self.worker_performance[worker_id] = PerformanceStats()
        
        # Update statistics
        stats = self.worker_performance[worker_id]
        stats.update(duration, success, self.learning_rate)
        
        # Add to history
        self.history.append((worker_id, task, duration, success))
        
        # Keep history bounded
        if len(self.history) > 10000:
            self.history = self.history[-5000:]
        
        logger.debug(
            f"Updated stats for {worker_id}: "
            f"avg_time={stats.avg_completion_time:.2f}s, "
            f"success_rate={stats.success_rate:.2%}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "assignments": self.assignments,
            "workers_tracked": len(self.worker_performance),
            "history_size": len(self.history),
            "worker_performance": {
                wid: {
                    "avg_time": stats.avg_completion_time,
                    "success_rate": stats.success_rate,
                    "total_tasks": stats.total_tasks,
                }
                for wid, stats in self.worker_performance.items()
            },
            "strategy": "Adaptive",
        }


@dataclass
class PerformanceStats:
    """Worker performance statistics."""
    
    avg_completion_time: float = 10.0  # Default 10s
    success_rate: float = 1.0
    total_tasks: int = 0
    
    def update(self, duration: float, success: bool, alpha: float = 0.1) -> None:
        """
        Update statistics with new measurement.
        
        Uses exponential moving average for smooth adaptation.
        
        Args:
            duration: Task duration
            success: Whether task succeeded
            alpha: Learning rate (0-1)
        """
        self.total_tasks += 1
        
        # Exponential moving average for completion time
        self.avg_completion_time = (
            alpha * duration + (1 - alpha) * self.avg_completion_time
        )
        
        # Exponential moving average for success rate
        success_value = 1.0 if success else 0.0
        self.success_rate = alpha * success_value + (1 - alpha) * self.success_rate


class RoundRobinScheduler(TaskScheduler):
    """
    Round-robin scheduler.
    
    Distributes tasks in circular order across workers.
    Simple and fair for homogeneous workers.
    
    Example:
        >>> scheduler = RoundRobinScheduler()
        >>> worker = scheduler.assign_task(task, workers)
    """
    
    def __init__(self) -> None:
        """Initialize round-robin scheduler."""
        self.current_index = 0
        self.assignments = 0
    
    def assign_task(
        self, task: Task, workers: List[WorkerInfo]
    ) -> Optional[WorkerInfo]:
        """Assign task in round-robin fashion."""
        available_workers = [w for w in workers if w.is_available()]
        
        if not available_workers:
            return None
        
        # Select worker in round-robin
        worker = available_workers[self.current_index % len(available_workers)]
        
        self.current_index += 1
        self.assignments += 1
        
        return worker
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "assignments": self.assignments,
            "current_index": self.current_index,
            "strategy": "RoundRobin",
        }


def create_scheduler(strategy: str, **kwargs: Any) -> TaskScheduler:
    """
    Factory function to create scheduler.
    
    Args:
        strategy: Scheduler strategy name
            - 'fifo': First-In-First-Out
            - 'priority': Priority-based
            - 'load_balancing': Load balancing
            - 'work_stealing': Work stealing
            - 'adaptive': Adaptive learning
            - 'round_robin': Round-robin
        **kwargs: Additional scheduler-specific arguments
    
    Returns:
        TaskScheduler instance
    
    Example:
        >>> scheduler = create_scheduler('adaptive', learning_rate=0.15)
    """
    schedulers = {
        "fifo": FIFOScheduler,
        "priority": PriorityScheduler,
        "load_balancing": LoadBalancingScheduler,
        "work_stealing": WorkStealingScheduler,
        "adaptive": AdaptiveScheduler,
        "round_robin": RoundRobinScheduler,
    }
    
    if strategy not in schedulers:
        raise ValueError(
            f"Unknown scheduler strategy: {strategy}. "
            f"Available: {list(schedulers.keys())}"
        )
    
    return schedulers[strategy](**kwargs)
