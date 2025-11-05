"""Master node for distributed architecture search.

The master coordinates optimization across multiple worker nodes, distributing
evaluation tasks and aggregating results.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import threading
import time
import uuid
from concurrent import futures
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional

try:
    import grpc

    from morphml.distributed.proto import worker_pb2, worker_pb2_grpc

    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

from morphml.core.graph import ModelGraph
from morphml.exceptions import DistributedError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class WorkerInfo:
    """Worker node metadata and state."""

    worker_id: str
    host: str
    port: int
    num_gpus: int
    gpu_ids: List[int] = field(default_factory=list)
    status: str = "idle"  # 'idle', 'busy', 'dead'
    last_heartbeat: float = field(default_factory=time.time)
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_alive(self, timeout: float = 30.0) -> bool:
        """Check if worker is alive based on heartbeat."""
        return (time.time() - self.last_heartbeat) < timeout

    def is_available(self) -> bool:
        """Check if worker is available for new tasks."""
        return self.status == "idle" and self.is_alive()


@dataclass
class Task:
    """Evaluation task for distributed execution."""

    task_id: str
    architecture: ModelGraph
    status: str = "pending"  # 'pending', 'running', 'completed', 'failed'
    worker_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    num_retries: int = 0
    max_retries: int = 3

    def duration(self) -> Optional[float]:
        """Get task execution duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.num_retries < self.max_retries


class MasterNode:
    """
    Master node for distributed NAS.

    Coordinates architecture search across multiple worker nodes:
    1. Distributes evaluation tasks to workers
    2. Collects and aggregates results
    3. Monitors worker health via heartbeat
    4. Handles worker failures and task reassignment
    5. Manages optimization state

    Args:
        optimizer: Base optimizer (GA, BO, etc.)
        config: Master configuration
            - host: Master host (default: '0.0.0.0')
            - port: Master port (default: 50051)
            - num_workers: Expected number of workers
            - heartbeat_interval: Heartbeat check interval (seconds, default: 10)
            - task_timeout: Task timeout (seconds, default: 3600)
            - max_retries: Maximum task retries (default: 3)

    Example:
        >>> from morphml.optimizers import GeneticAlgorithm
        >>> optimizer = GeneticAlgorithm(space, population_size=50)
        >>> master = MasterNode(optimizer, {'port': 50051, 'num_workers': 4})
        >>> master.start()
        >>> best = master.run_experiment(num_generations=100)
        >>> master.stop()
    """

    def __init__(self, optimizer: Any, config: Dict[str, Any]):
        """Initialize master node."""
        if not GRPC_AVAILABLE:
            raise DistributedError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )

        self.optimizer = optimizer
        self.config = config

        # Server configuration
        self.host = config.get("host", "0.0.0.0")
        self.port = config.get("port", 50051)
        self.num_workers = config.get("num_workers", 4)
        self.heartbeat_interval = config.get("heartbeat_interval", 10)
        self.task_timeout = config.get("task_timeout", 3600)
        self.max_retries = config.get("max_retries", 3)

        # Worker registry
        self.workers: Dict[str, WorkerInfo] = {}
        self.worker_lock = threading.Lock()

        # Task management
        self.pending_tasks: Queue = Queue()
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        self.task_lock = threading.Lock()

        # gRPC server
        self.server: Optional[grpc.Server] = None
        self.master_id = str(uuid.uuid4())[:8]

        # State
        self.running = False
        self.total_evaluations = 0

        logger.info(f"Initialized MasterNode (id={self.master_id}) on {self.host}:{self.port}")

    def start(self) -> None:
        """Start master node server."""
        logger.info(f"Starting master node on {self.host}:{self.port}")

        # Create gRPC server
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=20),
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ],
        )

        # Add servicer
        worker_pb2_grpc.add_MasterServiceServicer_to_server(MasterServicer(self), self.server)

        # Start server
        self.server.add_insecure_port(f"{self.host}:{self.port}")
        self.server.start()

        self.running = True

        # Start background threads
        self._start_heartbeat_monitor()
        self._start_task_dispatcher()

        logger.info(f"Master node started successfully (id={self.master_id})")

    def stop(self) -> None:
        """Stop master node gracefully."""
        logger.info("Stopping master node")
        self.running = False

        if self.server:
            self.server.stop(grace=5)

        logger.info("Master node stopped")

    def register_worker(self, worker_id: str, worker_info: Dict[str, Any]) -> bool:
        """
        Register a worker node.

        Args:
            worker_id: Unique worker identifier
            worker_info: Worker metadata

        Returns:
            True if registration successful
        """
        with self.worker_lock:
            self.workers[worker_id] = WorkerInfo(
                worker_id=worker_id,
                host=worker_info["host"],
                port=worker_info["port"],
                num_gpus=worker_info.get("num_gpus", 1),
                gpu_ids=worker_info.get("gpu_ids", []),
                status="idle",
                last_heartbeat=time.time(),
                metadata=worker_info.get("metadata", {}),
            )

            logger.info(
                f"Worker registered: {worker_id} "
                f"({worker_info['host']}:{worker_info['port']}, "
                f"GPUs: {worker_info.get('num_gpus', 1)})"
            )

        return True

    def update_heartbeat(self, worker_id: str, status: str) -> bool:
        """
        Update worker heartbeat.

        Args:
            worker_id: Worker identifier
            status: Worker status ('idle', 'busy', 'error')

        Returns:
            True if worker found and updated
        """
        with self.worker_lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.last_heartbeat = time.time()
                worker.status = status
                return True

        logger.warning(f"Heartbeat from unknown worker: {worker_id}")
        return False

    def submit_task(self, architecture: ModelGraph, task_id: Optional[str] = None) -> str:
        """
        Submit architecture evaluation task.

        Args:
            architecture: ModelGraph to evaluate
            task_id: Optional task ID (generated if not provided)

        Returns:
            Task ID
        """
        if task_id is None:
            task_id = str(uuid.uuid4())

        task = Task(
            task_id=task_id,
            architecture=architecture,
            status="pending",
            created_at=time.time(),
            max_retries=self.max_retries,
        )

        self.pending_tasks.put(task)

        logger.debug(f"Task submitted: {task_id}")

        return task_id

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Get result for task (blocking).

        Args:
            task_id: Task identifier
            timeout: Maximum wait time (seconds)

        Returns:
            Task result dictionary or None if timeout/failed
        """
        start_time = time.time()

        while True:
            with self.task_lock:
                # Check completed
                if task_id in self.completed_tasks:
                    task = self.completed_tasks[task_id]
                    return task.result

                # Check failed
                if task_id in self.failed_tasks:
                    logger.warning(f"Task {task_id} failed: {self.failed_tasks[task_id].error}")
                    return None

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Task {task_id} timeout after {timeout}s")
                return None

            time.sleep(0.1)

    def run_experiment(
        self, num_generations: int = 100, callback: Optional[Callable] = None
    ) -> List[ModelGraph]:
        """
        Run full distributed NAS experiment.

        Args:
            num_generations: Number of optimization generations
            callback: Optional callback(generation, stats) called each generation

        Returns:
            List of best architectures found

        Example:
            >>> def progress(gen, stats):
            ...     print(f"Gen {gen}: best={stats['best_fitness']:.4f}")
            >>> best = master.run_experiment(100, callback=progress)
        """
        logger.info(
            f"Starting distributed experiment "
            f"({num_generations} generations, {self.num_workers} workers)"
        )

        # Wait for workers
        self._wait_for_workers(timeout=300)

        # Initialize optimizer
        logger.info("Initializing population")
        if hasattr(self.optimizer, "initialize_population"):
            self.optimizer.initialize_population()

        # Evolution loop
        for generation in range(num_generations):
            logger.info(f"Generation {generation + 1}/{num_generations}")

            # Get individuals to evaluate
            if hasattr(self.optimizer, "population"):
                individuals = self.optimizer.population.get_unevaluated()
            else:
                # Fallback for optimizers without population
                individuals = []

            if not individuals:
                logger.warning(f"No individuals to evaluate in generation {generation}")
                continue

            # Submit tasks
            task_mapping = []
            for individual in individuals:
                task_id = self.submit_task(individual.graph)
                task_mapping.append((individual, task_id))

            # Wait for results
            for individual, task_id in task_mapping:
                result = self.get_result(task_id, timeout=self.task_timeout)

                if result is None:
                    logger.warning(f"Task {task_id} failed")
                    individual.set_fitness(0.0)
                else:
                    fitness = result.get("fitness", result.get("val_accuracy", 0.0))
                    individual.set_fitness(fitness)
                    self.total_evaluations += 1

            # Optimizer step (if applicable)
            if hasattr(self.optimizer, "evolve"):
                self.optimizer.evolve()

            # Statistics
            stats = self._get_generation_stats()
            logger.info(
                f"Gen {generation + 1}: "
                f"best={stats['best_fitness']:.4f}, "
                f"avg={stats['avg_fitness']:.4f}, "
                f"workers={len(self.workers)}/{self.num_workers}"
            )

            # Callback
            if callback:
                callback(generation + 1, stats)

        # Return best architectures
        logger.info(f"Experiment complete. Total evaluations: {self.total_evaluations}")

        if hasattr(self.optimizer, "population"):
            return self.optimizer.population.get_best_individuals(10)
        elif hasattr(self.optimizer, "best_individual"):
            return [self.optimizer.best_individual]
        else:
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get master node statistics."""
        with self.worker_lock:
            alive_workers = sum(1 for w in self.workers.values() if w.is_alive())
            busy_workers = sum(1 for w in self.workers.values() if w.status == "busy")

        with self.task_lock:
            pending = self.pending_tasks.qsize()
            running = len(self.running_tasks)
            completed = len(self.completed_tasks)
            failed = len(self.failed_tasks)

        return {
            "workers_total": len(self.workers),
            "workers_alive": alive_workers,
            "workers_busy": busy_workers,
            "tasks_pending": pending,
            "tasks_running": running,
            "tasks_completed": completed,
            "tasks_failed": failed,
            "total_evaluations": self.total_evaluations,
        }

    def _wait_for_workers(self, timeout: float = 300) -> None:
        """Wait for expected number of workers to connect."""
        logger.info(f"Waiting for {self.num_workers} workers (timeout: {timeout}s)...")

        start_time = time.time()
        while len(self.workers) < self.num_workers:
            if (time.time() - start_time) > timeout:
                raise DistributedError(
                    f"Only {len(self.workers)}/{self.num_workers} workers connected "
                    f"after {timeout}s"
                )

            time.sleep(1)

        logger.info(f"All {self.num_workers} workers connected")

    def _start_heartbeat_monitor(self) -> None:
        """Start background thread to monitor worker heartbeats."""

        def monitor() -> None:
            while self.running:
                time.time()

                with self.worker_lock:
                    for worker_id, worker in list(self.workers.items()):
                        # Check heartbeat timeout
                        if not worker.is_alive():
                            logger.warning(f"Worker {worker_id} heartbeat timeout")
                            worker.status = "dead"

                            # Reassign tasks
                            self._reassign_worker_tasks(worker_id)

                time.sleep(self.heartbeat_interval)

        thread = threading.Thread(target=monitor, daemon=True, name="HeartbeatMonitor")
        thread.start()
        logger.debug("Heartbeat monitor started")

    def _start_task_dispatcher(self) -> None:
        """Start background thread to dispatch tasks to workers."""

        def dispatch() -> None:
            while self.running:
                try:
                    # Get pending task
                    task = self.pending_tasks.get(timeout=1.0)

                    # Find available worker
                    worker = self._find_available_worker()

                    if worker:
                        # Dispatch task
                        self._dispatch_task_to_worker(task, worker)
                    else:
                        # No workers available, requeue
                        self.pending_tasks.put(task)
                        time.sleep(0.5)

                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Task dispatcher error: {e}")

        thread = threading.Thread(target=dispatch, daemon=True, name="TaskDispatcher")
        thread.start()
        logger.debug("Task dispatcher started")

    def _find_available_worker(self) -> Optional[WorkerInfo]:
        """Find an idle worker."""
        with self.worker_lock:
            for worker in self.workers.values():
                if worker.is_available():
                    return worker
        return None

    def _dispatch_task_to_worker(self, task: Task, worker: WorkerInfo) -> None:
        """Dispatch task to specific worker."""
        logger.debug(f"Dispatching task {task.task_id} to worker {worker.worker_id}")

        # Update task status
        task.status = "running"
        task.worker_id = worker.worker_id
        task.started_at = time.time()

        with self.task_lock:
            self.running_tasks[task.task_id] = task

        # Update worker status
        worker.status = "busy"
        worker.current_task = task.task_id

        # Send task via gRPC (async)
        def send_task() -> None:
            try:
                channel = grpc.insecure_channel(f"{worker.host}:{worker.port}")
                stub = worker_pb2_grpc.WorkerServiceStub(channel)

                request = worker_pb2.EvaluateRequest(
                    task_id=task.task_id, architecture=task.architecture.to_json()
                )

                # Note: This is async, result comes via SubmitResult RPC
                stub.Evaluate(request, timeout=self.task_timeout)

            except grpc.RpcError as e:
                logger.error(f"Failed to dispatch task {task.task_id}: {e}")
                self._handle_task_failure(task.task_id, str(e))

        # Run in thread to avoid blocking
        threading.Thread(target=send_task, daemon=True).start()

    def _handle_task_result(self, task_id: str, result: Dict[str, Any], duration: float) -> None:
        """Handle task result from worker."""
        with self.task_lock:
            if task_id not in self.running_tasks:
                logger.warning(f"Received result for unknown task: {task_id}")
                return

            task = self.running_tasks.pop(task_id)
            task.status = "completed"
            task.completed_at = time.time()
            task.result = result

            self.completed_tasks[task_id] = task

        # Update worker
        if task.worker_id:
            with self.worker_lock:
                if task.worker_id in self.workers:
                    worker = self.workers[task.worker_id]
                    worker.status = "idle"
                    worker.current_task = None
                    worker.tasks_completed += 1

        logger.debug(
            f"Task {task_id} completed in {duration:.2f}s "
            f"(fitness: {result.get('fitness', 'N/A')})"
        )

    def _handle_task_failure(self, task_id: str, error: str) -> None:
        """Handle task failure."""
        with self.task_lock:
            if task_id in self.running_tasks:
                task = self.running_tasks.pop(task_id)
            else:
                logger.warning(f"Failure for unknown task: {task_id}")
                return

            task.error = error
            task.num_retries += 1

            # Retry if possible
            if task.can_retry():
                logger.warning(
                    f"Task {task_id} failed (retry {task.num_retries}/{task.max_retries}): {error}"
                )
                task.status = "pending"
                task.worker_id = None
                self.pending_tasks.put(task)
            else:
                logger.error(f"Task {task_id} failed permanently after {task.num_retries} retries")
                task.status = "failed"
                self.failed_tasks[task_id] = task

        # Update worker
        if task.worker_id:
            with self.worker_lock:
                if task.worker_id in self.workers:
                    worker = self.workers[task.worker_id]
                    worker.status = "idle"
                    worker.current_task = None
                    worker.tasks_failed += 1

    def _reassign_worker_tasks(self, worker_id: str) -> None:
        """Reassign tasks from dead worker."""
        with self.task_lock:
            tasks_to_reassign = []

            for _task_id, task in list(self.running_tasks.items()):
                if task.worker_id == worker_id:
                    tasks_to_reassign.append(task)

            for task in tasks_to_reassign:
                logger.warning(f"Reassigning task {task.task_id} from dead worker {worker_id}")
                self.running_tasks.pop(task.task_id)
                task.status = "pending"
                task.worker_id = None
                task.num_retries += 1

                if task.can_retry():
                    self.pending_tasks.put(task)
                else:
                    task.status = "failed"
                    self.failed_tasks[task.task_id] = task

    def _get_generation_stats(self) -> Dict[str, float]:
        """Get statistics for current generation."""
        if hasattr(self.optimizer, "population"):
            evaluated = [ind for ind in self.optimizer.population.individuals if ind.is_evaluated()]

            if evaluated:
                fitnesses = [ind.fitness for ind in evaluated]
                return {
                    "best_fitness": max(fitnesses),
                    "avg_fitness": sum(fitnesses) / len(fitnesses),
                    "min_fitness": min(fitnesses),
                }

        return {"best_fitness": 0.0, "avg_fitness": 0.0, "min_fitness": 0.0}


class MasterServicer(worker_pb2_grpc.MasterServiceServicer):
    """gRPC servicer for master node."""

    def __init__(self, master: MasterNode):
        """Initialize servicer."""
        self.master = master

    def RegisterWorker(
        self, request: worker_pb2.RegisterRequest, context: grpc.ServicerContext
    ) -> worker_pb2.RegisterResponse:
        """Handle worker registration."""
        try:
            worker_info = {
                "host": request.host,
                "port": request.port,
                "num_gpus": request.num_gpus,
                "gpu_ids": list(request.gpu_ids),
                "metadata": dict(request.metadata),
            }

            success = self.master.register_worker(request.worker_id, worker_info)

            return worker_pb2.RegisterResponse(
                success=success,
                message="Worker registered successfully",
                master_id=self.master.master_id,
            )

        except Exception as e:
            logger.error(f"Worker registration failed: {e}")
            return worker_pb2.RegisterResponse(
                success=False, message=str(e), master_id=self.master.master_id
            )

    def Heartbeat(
        self, request: worker_pb2.HeartbeatRequest, context: grpc.ServicerContext
    ) -> worker_pb2.HeartbeatResponse:
        """Handle worker heartbeat."""
        success = self.master.update_heartbeat(request.worker_id, request.status)

        return worker_pb2.HeartbeatResponse(
            acknowledged=success, should_continue=self.master.running
        )

    def SubmitResult(
        self, request: worker_pb2.ResultRequest, context: grpc.ServicerContext
    ) -> worker_pb2.ResultResponse:
        """Handle task result submission."""
        try:
            if request.success:
                result = dict(request.metrics)
                self.master._handle_task_result(request.task_id, result, request.duration)
            else:
                self.master._handle_task_failure(request.task_id, request.error)

            return worker_pb2.ResultResponse(acknowledged=True, message="Result received")

        except Exception as e:
            logger.error(f"Failed to handle result: {e}")
            return worker_pb2.ResultResponse(acknowledged=False, message=str(e))

    def RequestTask(
        self, request: worker_pb2.TaskRequest, context: grpc.ServicerContext
    ) -> worker_pb2.TaskResponse:
        """Handle task request from worker (pull model)."""
        # Pull model implementation for future use
        return worker_pb2.TaskResponse(has_task=False, tasks=[])
