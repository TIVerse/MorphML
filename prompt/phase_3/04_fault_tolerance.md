# Component 4: Fault Tolerance & Recovery

**Duration:** Week 6  
**LOC Target:** ~3,000  
**Dependencies:** Components 1-3

---

## 🎯 Objective

Build robust fault tolerance:
1. **Worker Failure Detection** - Detect and handle worker crashes
2. **Task Reassignment** - Automatically retry failed tasks
3. **Checkpoint Recovery** - Resume from failures
4. **Graceful Degradation** - Continue with reduced capacity

---

## 📋 Files to Create

### 1. `distributed/fault_tolerance.py` (~2,000 LOC)

```python
from enum import Enum
from typing import Dict, List, Optional
import time

class FailureType(Enum):
    """Types of failures."""
    WORKER_CRASH = "worker_crash"
    TASK_TIMEOUT = "task_timeout"
    NETWORK_ERROR = "network_error"
    OUT_OF_MEMORY = "out_of_memory"


class FaultToleranceManager:
    """
    Manage fault tolerance and recovery.
    
    Features:
    - Automatic task retry
    - Worker health monitoring
    - Checkpoint-based recovery
    - Circuit breaker pattern
    
    Config:
        max_retries: Maximum task retries (default: 3)
        retry_delay: Delay between retries (seconds, default: 5)
        health_check_interval: Worker health check (seconds, default: 30)
        checkpoint_interval: Checkpoint frequency (generations, default: 10)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 5)
        self.health_check_interval = config.get('health_check_interval', 30)
        self.checkpoint_interval = config.get('checkpoint_interval', 10)
        
        # Task retry tracking
        self.task_retries: Dict[str, int] = {}
        self.failed_tasks: List[Task] = []
        
        # Worker failure tracking
        self.worker_failures: Dict[str, List[FailureEvent]] = {}
        
        # Circuit breaker states
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def handle_task_failure(
        self,
        task: Task,
        failure_type: FailureType,
        error: str
    ) -> bool:
        """
        Handle task failure.
        
        Returns:
            True if task should be retried, False otherwise
        """
        task_id = task.task_id
        
        # Increment retry count
        self.task_retries[task_id] = self.task_retries.get(task_id, 0) + 1
        
        logger.warning(
            f"Task {task_id} failed: {failure_type.value} - {error} "
            f"(retry {self.task_retries[task_id]}/{self.max_retries})"
        )
        
        # Check if should retry
        if self.task_retries[task_id] < self.max_retries:
            # Delay before retry
            time.sleep(self.retry_delay)
            return True
        else:
            logger.error(f"Task {task_id} exceeded max retries")
            self.failed_tasks.append(task)
            return False
    
    def handle_worker_failure(self, worker_id: str):
        """
        Handle worker failure.
        
        1. Mark worker as dead
        2. Reassign all tasks from failed worker
        3. Update circuit breaker
        """
        logger.error(f"Worker {worker_id} failed")
        
        # Record failure event
        if worker_id not in self.worker_failures:
            self.worker_failures[worker_id] = []
        
        self.worker_failures[worker_id].append(
            FailureEvent(
                timestamp=time.time(),
                failure_type=FailureType.WORKER_CRASH
            )
        )
        
        # Check if worker fails too often
        recent_failures = self._count_recent_failures(worker_id, window=3600)
        
        if recent_failures > 3:
            logger.warning(f"Worker {worker_id} has {recent_failures} recent failures")
            self._enable_circuit_breaker(worker_id)
    
    def reassign_tasks(
        self,
        failed_worker_id: str,
        task_ids: List[str],
        available_workers: List[WorkerInfo]
    ) -> Dict[str, str]:
        """
        Reassign tasks from failed worker.
        
        Returns:
            Mapping of task_id -> new_worker_id
        """
        reassignment = {}
        
        for task_id in task_ids:
            # Find healthy worker
            for worker in available_workers:
                if (worker.worker_id != failed_worker_id and
                    worker.status != 'dead' and
                    not self._is_circuit_open(worker.worker_id)):
                    
                    reassignment[task_id] = worker.worker_id
                    logger.info(f"Reassigned task {task_id} to worker {worker.worker_id}")
                    break
        
        return reassignment
    
    def recover_from_checkpoint(
        self,
        checkpoint: Dict,
        optimizer: BaseOptimizer
    ) -> int:
        """
        Recover experiment from checkpoint.
        
        Returns:
            Generation to resume from
        """
        logger.info(f"Recovering from checkpoint (gen {checkpoint['generation']})")
        
        # Restore optimizer state
        optimizer.load_state(checkpoint['optimizer_state'])
        
        # Restore population
        population = [
            Individual.from_dict(ind_dict)
            for ind_dict in checkpoint['population']
        ]
        optimizer.population = population
        
        return checkpoint['generation']
    
    def _count_recent_failures(self, worker_id: str, window: float) -> int:
        """Count failures within time window."""
        if worker_id not in self.worker_failures:
            return 0
        
        current_time = time.time()
        recent = [
            f for f in self.worker_failures[worker_id]
            if (current_time - f.timestamp) < window
        ]
        
        return len(recent)
    
    def _enable_circuit_breaker(self, worker_id: str):
        """Enable circuit breaker for worker."""
        if worker_id not in self.circuit_breakers:
            self.circuit_breakers[worker_id] = CircuitBreaker()
        
        self.circuit_breakers[worker_id].open()
        logger.warning(f"Circuit breaker opened for worker {worker_id}")
    
    def _is_circuit_open(self, worker_id: str) -> bool:
        """Check if circuit breaker is open."""
        if worker_id not in self.circuit_breakers:
            return False
        
        return self.circuit_breakers[worker_id].is_open()


@dataclass
class FailureEvent:
    """Failure event record."""
    timestamp: float
    failure_type: FailureType
    details: Optional[str] = None


class CircuitBreaker:
    """
    Circuit breaker pattern for workers.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Worker disabled due to failures
    - HALF_OPEN: Testing if worker recovered
    """
    
    def __init__(self, timeout: float = 300):
        self.state = 'CLOSED'
        self.opened_at: Optional[float] = None
        self.timeout = timeout
    
    def open(self):
        """Open circuit (disable worker)."""
        self.state = 'OPEN'
        self.opened_at = time.time()
    
    def is_open(self) -> bool:
        """Check if circuit is open."""
        if self.state == 'CLOSED':
            return False
        
        # Check if timeout elapsed (transition to HALF_OPEN)
        if (time.time() - self.opened_at) > self.timeout:
            self.state = 'HALF_OPEN'
            return False
        
        return True
    
    def close(self):
        """Close circuit (re-enable worker)."""
        self.state = 'CLOSED'
        self.opened_at = None
```

---

### 2. `distributed/health_monitor.py` (~1,000 LOC)

```python
import psutil
import GPUtil
from typing import Dict

class HealthMonitor:
    """
    Monitor worker health metrics.
    
    Tracks:
    - CPU usage
    - Memory usage
    - GPU utilization
    - Disk space
    - Network latency
    """
    
    @staticmethod
    def get_system_health() -> Dict[str, Any]:
        """Get current system health metrics."""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory
        mem = psutil.virtual_memory()
        
        # GPU
        gpus = GPUtil.getGPUs()
        gpu_stats = [{
            'id': gpu.id,
            'load': gpu.load * 100,
            'memory_used': gpu.memoryUsed,
            'memory_total': gpu.memoryTotal,
            'temperature': gpu.temperature
        } for gpu in gpus]
        
        # Disk
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': mem.percent,
            'memory_available_gb': mem.available / (1024**3),
            'gpus': gpu_stats,
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3)
        }
    
    @staticmethod
    def is_healthy(health: Dict[str, Any]) -> bool:
        """Check if system is healthy."""
        # CPU overload
        if health['cpu_percent'] > 95:
            return False
        
        # Memory critical
        if health['memory_percent'] > 95:
            return False
        
        # Disk full
        if health['disk_percent'] > 95:
            return False
        
        # GPU issues
        for gpu in health['gpus']:
            if gpu['temperature'] > 85:  # Too hot
                return False
            if gpu['memory_used'] / gpu['memory_total'] > 0.95:
                return False
        
        return True
```

---

## 🧪 Tests

```python
def test_task_retry():
    """Test automatic task retry."""
    ft_manager = FaultToleranceManager({'max_retries': 3})
    
    task = Task(task_id='test', ...)
    
    # First 2 failures should retry
    assert ft_manager.handle_task_failure(task, FailureType.NETWORK_ERROR, "timeout")
    assert ft_manager.handle_task_failure(task, FailureType.NETWORK_ERROR, "timeout")
    
    # Third failure should not retry
    assert not ft_manager.handle_task_failure(task, FailureType.NETWORK_ERROR, "timeout")
```

---

## ✅ Deliverables

- [ ] Fault tolerance manager
- [ ] Automatic task retry logic
- [ ] Worker failure detection and handling
- [ ] Circuit breaker pattern
- [ ] Health monitoring
- [ ] Checkpoint recovery

---

**Next:** `05_kubernetes.md`
