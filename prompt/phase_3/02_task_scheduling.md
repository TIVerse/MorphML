# Component 2: Task Scheduling & Load Balancing

**Duration:** Week 3-4  
**LOC Target:** ~4,000  
**Dependencies:** Component 1 (Master-Worker)

---

## 🎯 Objective

Implement intelligent task scheduling strategies:
1. **Priority Queue** - Prioritize important evaluations
2. **Load Balancing** - Distribute work evenly across workers
3. **Work Stealing** - Idle workers steal from busy ones
4. **Adaptive Scheduling** - Learn optimal task assignment

---

## 📋 Files to Create

### 1. `distributed/scheduler.py` (~2,000 LOC)

**Scheduling strategies:**

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from queue import PriorityQueue

class TaskScheduler(ABC):
    """Base class for task schedulers."""
    
    @abstractmethod
    def assign_task(
        self,
        task: Task,
        workers: List[WorkerInfo]
    ) -> Optional[WorkerInfo]:
        """Assign task to a worker."""
        pass


class FIFOScheduler(TaskScheduler):
    """First-In-First-Out scheduling."""
    
    def assign_task(self, task: Task, workers: List[WorkerInfo]) -> Optional[WorkerInfo]:
        # Find first idle worker
        for worker in workers:
            if worker.status == 'idle':
                return worker
        return None


class PriorityScheduler(TaskScheduler):
    """
    Priority-based scheduling.
    
    Tasks with higher priority are scheduled first.
    Useful for multi-fidelity optimization where promising
    architectures get more resources.
    """
    
    def __init__(self):
        self.task_queue = PriorityQueue()
    
    def enqueue(self, task: Task, priority: float):
        """Add task with priority (higher = more important)."""
        self.task_queue.put((-priority, task))  # Negative for max-heap
    
    def assign_task(self, task: Task, workers: List[WorkerInfo]) -> Optional[WorkerInfo]:
        # Select worker with least load
        idle_workers = [w for w in workers if w.status == 'idle']
        
        if idle_workers:
            return min(idle_workers, key=lambda w: w.task_count)
        
        return None


class LoadBalancingScheduler(TaskScheduler):
    """
    Load balancing scheduler.
    
    Distributes tasks evenly based on worker capacity.
    """
    
    def assign_task(self, task: Task, workers: List[WorkerInfo]) -> Optional[WorkerInfo]:
        # Calculate load for each worker
        worker_loads = {
            w.worker_id: self._calculate_load(w)
            for w in workers
            if w.status != 'dead'
        }
        
        # Assign to least loaded worker
        if worker_loads:
            min_worker_id = min(worker_loads, key=worker_loads.get)
            return next(w for w in workers if w.worker_id == min_worker_id)
        
        return None
    
    def _calculate_load(self, worker: WorkerInfo) -> float:
        """
        Calculate worker load.
        
        Load = (running_tasks / num_gpus) + queue_length
        """
        running = worker.task_count
        capacity = worker.num_gpus
        queue_len = len(worker.task_queue)
        
        return (running / capacity) + queue_len


class WorkStealingScheduler(TaskScheduler):
    """
    Work stealing scheduler.
    
    Idle workers steal tasks from busy workers.
    """
    
    def __init__(self, steal_threshold: int = 2):
        self.steal_threshold = steal_threshold
    
    def steal_task(
        self,
        idle_worker: WorkerInfo,
        workers: List[WorkerInfo]
    ) -> Optional[Task]:
        """
        Idle worker steals task from busiest worker.
        
        Args:
            idle_worker: Worker looking for work
            workers: All workers
        
        Returns:
            Stolen task or None
        """
        # Find workers with > threshold tasks
        busy_workers = [
            w for w in workers
            if len(w.task_queue) > self.steal_threshold
        ]
        
        if not busy_workers:
            return None
        
        # Steal from busiest
        busiest = max(busy_workers, key=lambda w: len(w.task_queue))
        
        if busiest.task_queue:
            stolen_task = busiest.task_queue.pop()
            logger.info(
                f"Worker {idle_worker.worker_id} stole task "
                f"from {busiest.worker_id}"
            )
            return stolen_task
        
        return None


class AdaptiveScheduler(TaskScheduler):
    """
    Adaptive scheduler using reinforcement learning.
    
    Learns optimal assignment policy based on:
    - Worker performance history
    - Task characteristics
    - System state
    """
    
    def __init__(self):
        self.history: List[Assignment] = []
        self.worker_performance: Dict[str, PerformanceStats] = {}
    
    def assign_task(self, task: Task, workers: List[WorkerInfo]) -> Optional[WorkerInfo]:
        """
        Assign task using learned policy.
        
        Features:
        - Worker GPU count
        - Worker average completion time
        - Worker success rate
        - Task estimated complexity
        """
        # Compute scores for each worker
        scores = {}
        for worker in workers:
            if worker.status == 'idle':
                score = self._compute_assignment_score(task, worker)
                scores[worker.worker_id] = score
        
        if scores:
            best_worker_id = max(scores, key=scores.get)
            return next(w for w in workers if w.worker_id == best_worker_id)
        
        return None
    
    def _compute_assignment_score(self, task: Task, worker: WorkerInfo) -> float:
        """Compute assignment score."""
        perf = self.worker_performance.get(worker.worker_id)
        
        if perf is None:
            return 1.0  # Default for new worker
        
        # Score based on performance
        speed_score = 1.0 / (perf.avg_completion_time + 1e-6)
        success_score = perf.success_rate
        
        return speed_score * success_score
    
    def record_completion(self, worker_id: str, task: Task, duration: float, success: bool):
        """Record task completion for learning."""
        if worker_id not in self.worker_performance:
            self.worker_performance[worker_id] = PerformanceStats()
        
        stats = self.worker_performance[worker_id]
        stats.update(duration, success)


@dataclass
class PerformanceStats:
    """Worker performance statistics."""
    avg_completion_time: float = 0.0
    success_rate: float = 1.0
    total_tasks: int = 0
    
    def update(self, duration: float, success: bool):
        """Update statistics."""
        self.total_tasks += 1
        
        # Exponential moving average
        alpha = 0.1
        self.avg_completion_time = (
            alpha * duration + (1 - alpha) * self.avg_completion_time
        )
        
        self.success_rate = (
            alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
        )
```

---

### 2. `distributed/resource_manager.py` (~1,500 LOC)

**Resource management:**

```python
class ResourceManager:
    """
    Manage computational resources across workers.
    
    Tracks:
    - GPU availability
    - Memory usage
    - CPU utilization
    - Network bandwidth
    """
    
    def __init__(self):
        self.resources: Dict[str, WorkerResources] = {}
    
    def update_resources(self, worker_id: str, resources: Dict[str, Any]):
        """Update worker resource information."""
        self.resources[worker_id] = WorkerResources(**resources)
    
    def find_suitable_worker(
        self,
        requirements: TaskRequirements
    ) -> Optional[str]:
        """
        Find worker that meets task requirements.
        
        Args:
            requirements: Task resource requirements
                - min_gpus: Minimum GPUs
                - min_memory: Minimum GPU memory (GB)
                - estimated_time: Estimated execution time
        
        Returns:
            worker_id or None
        """
        for worker_id, res in self.resources.items():
            if (res.available_gpus >= requirements.min_gpus and
                res.gpu_memory >= requirements.min_memory):
                return worker_id
        
        return None


@dataclass
class WorkerResources:
    """Worker resource state."""
    total_gpus: int
    available_gpus: int
    gpu_memory: float  # GB
    cpu_percent: float
    memory_percent: float
```

---

### 3. `distributed/batch_scheduler.py` (~500 LOC)

**Batch task scheduling:**

```python
class BatchScheduler:
    """
    Schedule batches of tasks efficiently.
    
    Groups tasks for better GPU utilization.
    """
    
    def schedule_batch(
        self,
        tasks: List[Task],
        workers: List[WorkerInfo],
        batch_size: int = 4
    ) -> Dict[str, List[Task]]:
        """
        Assign tasks to workers in batches.
        
        Returns:
            Mapping of worker_id -> tasks
        """
        assignment = {w.worker_id: [] for w in workers}
        
        # Sort workers by capacity
        workers_sorted = sorted(workers, key=lambda w: w.num_gpus, reverse=True)
        
        # Distribute tasks
        worker_idx = 0
        for task in tasks:
            worker = workers_sorted[worker_idx % len(workers_sorted)]
            assignment[worker.worker_id].append(task)
            
            # Move to next worker if batch full
            if len(assignment[worker.worker_id]) >= batch_size:
                worker_idx += 1
        
        return assignment
```

---

## 🧪 Tests

```python
def test_load_balancing():
    """Test load balancing scheduler."""
    scheduler = LoadBalancingScheduler()
    
    workers = [
        WorkerInfo(worker_id='w1', num_gpus=2, task_count=0),
        WorkerInfo(worker_id='w2', num_gpus=2, task_count=3)
    ]
    
    task = Task(...)
    assigned = scheduler.assign_task(task, workers)
    
    assert assigned.worker_id == 'w1'  # Less loaded
```

---

## ✅ Deliverables

- [ ] Priority queue scheduler
- [ ] Load balancing scheduler
- [ ] Work stealing implementation
- [ ] Adaptive learning scheduler
- [ ] Resource manager
- [ ] Batch scheduling

---

**Next:** `03_storage.md`
