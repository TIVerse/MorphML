"""Fault tolerance and recovery mechanisms.

Handles worker failures, task retries, and checkpoint-based recovery.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from morphml.core.search import Individual
from morphml.distributed.master import Task, WorkerInfo
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class FailureType(Enum):
    """Types of failures that can occur."""

    WORKER_CRASH = "worker_crash"
    TASK_TIMEOUT = "task_timeout"
    NETWORK_ERROR = "network_error"
    OUT_OF_MEMORY = "out_of_memory"
    GPU_ERROR = "gpu_error"
    EVALUATION_ERROR = "evaluation_error"
    UNKNOWN = "unknown"


@dataclass
class FailureEvent:
    """Record of a failure event."""

    timestamp: float
    failure_type: FailureType
    worker_id: Optional[str] = None
    task_id: Optional[str] = None
    details: Optional[str] = None
    recovered: bool = False


class CircuitBreaker:
    """
    Circuit breaker pattern for worker management.

    Prevents repeated use of failing workers by temporarily disabling them.

    States:
    - CLOSED: Normal operation
    - OPEN: Worker disabled due to failures
    - HALF_OPEN: Testing if worker has recovered

    Args:
        failure_threshold: Number of failures to trigger open state
        timeout: Seconds before transitioning to HALF_OPEN
        success_threshold: Successes needed in HALF_OPEN to close

    Example:
        >>> breaker = CircuitBreaker(failure_threshold=3, timeout=300)
        >>> breaker.record_failure()
        >>> if breaker.is_open():
        ...     print("Circuit open, worker disabled")
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        timeout: float = 300.0,
        success_threshold: int = 2,
    ):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold

        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        self.opened_at: Optional[float] = None

    def record_failure(self) -> None:
        """Record a failure."""
        if self.state == "CLOSED":
            self.failure_count += 1

            if self.failure_count >= self.failure_threshold:
                self._open()

        elif self.state == "HALF_OPEN":
            # Failed during testing, re-open
            self._open()
            self.success_count = 0

    def record_success(self) -> None:
        """Record a success."""
        if self.state == "HALF_OPEN":
            self.success_count += 1

            if self.success_count >= self.success_threshold:
                self._close()

        elif self.state == "CLOSED":
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)

    def is_open(self) -> bool:
        """
        Check if circuit is open.

        Returns:
            True if circuit is open (worker disabled)
        """
        if self.state == "CLOSED":
            return False

        # Check if timeout elapsed (transition to HALF_OPEN)
        if self.state == "OPEN" and self.opened_at:
            if (time.time() - self.opened_at) > self.timeout:
                self._half_open()
                return False

        return self.state == "OPEN"

    def _open(self) -> None:
        """Open the circuit."""
        self.state = "OPEN"
        self.opened_at = time.time()
        logger.warning("Circuit breaker opened")

    def _half_open(self) -> None:
        """Transition to half-open state."""
        self.state = "HALF_OPEN"
        self.success_count = 0
        logger.info("Circuit breaker half-open (testing recovery)")

    def _close(self) -> None:
        """Close the circuit."""
        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        self.opened_at = None
        logger.info("Circuit breaker closed (worker recovered)")

    def get_state(self) -> str:
        """Get current state."""
        return self.state


class FaultToleranceManager:
    """
    Manage fault tolerance and recovery.

    Provides automatic recovery from failures:
    - Task retry with exponential backoff
    - Worker failure detection and handling
    - Task reassignment from failed workers
    - Circuit breaker pattern for unreliable workers
    - Checkpoint-based experiment recovery

    Args:
        config: Configuration dictionary
            - max_retries: Maximum task retries (default: 3)
            - retry_delay: Base delay between retries in seconds (default: 5)
            - health_check_interval: Health check frequency (default: 30)
            - checkpoint_interval: Checkpoint every N generations (default: 10)
            - circuit_breaker_threshold: Failures to open circuit (default: 3)
            - circuit_breaker_timeout: Circuit breaker timeout (default: 300)

    Example:
        >>> manager = FaultToleranceManager({'max_retries': 3, 'retry_delay': 5})
        >>>
        >>> # Handle task failure
        >>> should_retry = manager.handle_task_failure(
        ...     task, FailureType.NETWORK_ERROR, "Connection timeout"
        ... )
        >>>
        >>> # Handle worker failure
        >>> manager.handle_worker_failure('worker-1')
        >>>
        >>> # Reassign tasks
        >>> reassignment = manager.reassign_tasks('worker-1', task_ids, workers)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize fault tolerance manager."""
        config = config or {}

        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 5.0)
        self.health_check_interval = config.get("health_check_interval", 30.0)
        self.checkpoint_interval = config.get("checkpoint_interval", 10)
        self.circuit_breaker_threshold = config.get("circuit_breaker_threshold", 3)
        self.circuit_breaker_timeout = config.get("circuit_breaker_timeout", 300.0)

        # Task retry tracking
        self.task_retries: Dict[str, int] = {}
        self.task_failures: Dict[str, List[FailureEvent]] = {}
        self.failed_tasks: List[Task] = []

        # Worker failure tracking
        self.worker_failures: Dict[str, List[FailureEvent]] = {}
        self.failed_workers: set = set()

        # Circuit breakers per worker
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Recovery tracking
        self.recovery_attempts: Dict[str, int] = {}

        logger.info(
            f"Initialized FaultToleranceManager "
            f"(max_retries={self.max_retries}, retry_delay={self.retry_delay}s)"
        )

    def handle_task_failure(self, task: Task, failure_type: FailureType, error: str) -> bool:
        """
        Handle task failure with automatic retry logic.

        Args:
            task: Failed task
            failure_type: Type of failure
            error: Error message

        Returns:
            True if task should be retried, False otherwise
        """
        task_id = task.task_id

        # Record failure event
        if task_id not in self.task_failures:
            self.task_failures[task_id] = []

        self.task_failures[task_id].append(
            FailureEvent(
                timestamp=time.time(),
                failure_type=failure_type,
                task_id=task_id,
                worker_id=task.worker_id,
                details=error,
            )
        )

        # Increment retry count
        self.task_retries[task_id] = self.task_retries.get(task_id, 0) + 1
        retry_count = self.task_retries[task_id]

        logger.warning(
            f"Task {task_id} failed: {failure_type.value} - {error} "
            f"(retry {retry_count}/{self.max_retries})"
        )

        # Check if should retry
        if retry_count < self.max_retries:
            # Exponential backoff
            delay = self.retry_delay * (2 ** (retry_count - 1))
            logger.info(f"Will retry task {task_id} after {delay:.1f}s")

            # Note: Actual sleep happens in the caller
            return True
        else:
            logger.error(f"Task {task_id} exceeded max retries, marking as failed")
            self.failed_tasks.append(task)
            return False

    def handle_worker_failure(
        self, worker_id: str, failure_type: FailureType = FailureType.WORKER_CRASH
    ) -> None:
        """
        Handle worker failure.

        1. Record failure event
        2. Update circuit breaker
        3. Mark worker as failed

        Args:
            worker_id: Failed worker ID
            failure_type: Type of failure
        """
        logger.error(f"Worker {worker_id} failed: {failure_type.value}")

        # Record failure event
        if worker_id not in self.worker_failures:
            self.worker_failures[worker_id] = []

        self.worker_failures[worker_id].append(
            FailureEvent(
                timestamp=time.time(),
                failure_type=failure_type,
                worker_id=worker_id,
            )
        )

        # Mark as failed
        self.failed_workers.add(worker_id)

        # Update circuit breaker
        breaker = self._get_circuit_breaker(worker_id)
        breaker.record_failure()

        # Check failure rate
        recent_failures = self._count_recent_failures(worker_id, window=3600)

        if recent_failures > self.circuit_breaker_threshold:
            logger.warning(
                f"Worker {worker_id} has {recent_failures} failures in last hour, "
                f"circuit breaker: {breaker.get_state()}"
            )

    def handle_worker_recovery(self, worker_id: str) -> None:
        """
        Handle worker recovery.

        Args:
            worker_id: Recovered worker ID
        """
        logger.info(f"Worker {worker_id} recovered")

        # Remove from failed set
        self.failed_workers.discard(worker_id)

        # Record success in circuit breaker
        if worker_id in self.circuit_breakers:
            self.circuit_breakers[worker_id].record_success()

    def reassign_tasks(
        self,
        failed_worker_id: str,
        task_ids: List[str],
        available_workers: List[WorkerInfo],
    ) -> Dict[str, str]:
        """
        Reassign tasks from failed worker to healthy workers.

        Args:
            failed_worker_id: Worker that failed
            task_ids: Tasks to reassign
            available_workers: Available workers

        Returns:
            Mapping of task_id -> new_worker_id
        """
        reassignment = {}

        # Filter healthy workers
        healthy_workers = [
            w
            for w in available_workers
            if w.worker_id != failed_worker_id
            and w.status != "dead"
            and not self.is_worker_unhealthy(w.worker_id)
        ]

        if not healthy_workers:
            logger.error("No healthy workers available for task reassignment")
            return reassignment

        # Round-robin assignment
        worker_idx = 0
        for task_id in task_ids:
            worker = healthy_workers[worker_idx % len(healthy_workers)]
            reassignment[task_id] = worker.worker_id

            logger.info(
                f"Reassigned task {task_id} from {failed_worker_id} " f"to {worker.worker_id}"
            )

            worker_idx += 1

        return reassignment

    def is_worker_unhealthy(self, worker_id: str) -> bool:
        """
        Check if worker is unhealthy.

        Args:
            worker_id: Worker ID

        Returns:
            True if worker should not receive tasks
        """
        # Check if in failed set
        if worker_id in self.failed_workers:
            return True

        # Check circuit breaker
        if worker_id in self.circuit_breakers:
            return self.circuit_breakers[worker_id].is_open()

        return False

    def recover_from_checkpoint(self, checkpoint: Dict[str, Any], optimizer: Any) -> int:
        """
        Recover experiment state from checkpoint.

        Args:
            checkpoint: Checkpoint dictionary
            optimizer: Optimizer to restore

        Returns:
            Generation to resume from
        """
        generation = checkpoint.get("generation", 0)

        logger.info(f"Recovering from checkpoint at generation {generation}")

        # Restore optimizer state
        optimizer_state = checkpoint.get("optimizer_state", {})
        if hasattr(optimizer, "load_state"):
            optimizer.load_state(optimizer_state)

        # Restore population
        population_data = checkpoint.get("population", [])
        if population_data and hasattr(optimizer, "population"):
            try:
                population = [Individual.from_dict(ind_dict) for ind_dict in population_data]
                optimizer.population = population
                logger.info(f"Restored population of {len(population)} individuals")
            except Exception as e:
                logger.warning(f"Failed to restore population: {e}")

        # Record recovery
        exp_id = checkpoint.get("experiment_id", "unknown")
        self.recovery_attempts[exp_id] = self.recovery_attempts.get(exp_id, 0) + 1

        logger.info(f"Recovery complete, resuming from generation {generation}")

        return generation

    def _get_circuit_breaker(self, worker_id: str) -> CircuitBreaker:
        """Get or create circuit breaker for worker."""
        if worker_id not in self.circuit_breakers:
            self.circuit_breakers[worker_id] = CircuitBreaker(
                failure_threshold=self.circuit_breaker_threshold,
                timeout=self.circuit_breaker_timeout,
            )

        return self.circuit_breakers[worker_id]

    def _count_recent_failures(self, worker_id: str, window: float = 3600.0) -> int:
        """
        Count worker failures within time window.

        Args:
            worker_id: Worker ID
            window: Time window in seconds

        Returns:
            Number of failures
        """
        if worker_id not in self.worker_failures:
            return 0

        current_time = time.time()
        recent = [
            f for f in self.worker_failures[worker_id] if (current_time - f.timestamp) < window
        ]

        return len(recent)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get fault tolerance statistics.

        Returns:
            Statistics dictionary
        """
        total_task_failures = sum(len(events) for events in self.task_failures.values())
        total_worker_failures = sum(len(events) for events in self.worker_failures.values())

        circuit_breaker_states = {
            wid: breaker.get_state() for wid, breaker in self.circuit_breakers.items()
        }

        return {
            "total_task_failures": total_task_failures,
            "total_worker_failures": total_worker_failures,
            "failed_tasks": len(self.failed_tasks),
            "failed_workers": len(self.failed_workers),
            "circuit_breakers": circuit_breaker_states,
            "tasks_with_retries": len(self.task_retries),
            "recovery_attempts": dict(self.recovery_attempts),
        }

    def reset(self) -> None:
        """Reset all fault tolerance state."""
        self.task_retries.clear()
        self.task_failures.clear()
        self.failed_tasks.clear()
        self.worker_failures.clear()
        self.failed_workers.clear()
        self.circuit_breakers.clear()
        self.recovery_attempts.clear()

        logger.info("Reset fault tolerance state")
