"""Worker node for distributed architecture evaluation.

Workers execute architecture evaluation tasks assigned by the master node.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import socket
import threading
import time
import uuid
from concurrent import futures
from typing import Any, Dict, Optional

try:
    import grpc

    from morphml.distributed.proto import worker_pb2, worker_pb2_grpc

    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    # Create stub modules when grpc is not available
    class _StubModule:
        def __getattr__(self, name):
            raise ImportError("grpc is not installed. Install with: pip install grpcio grpcio-tools")
    
    worker_pb2 = _StubModule()
    worker_pb2_grpc = _StubModule()
    grpc = _StubModule()

from morphml.core.graph import ModelGraph
from morphml.exceptions import DistributedError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class WorkerNode:
    """
    Worker node for distributed architecture evaluation.

    Responsibilities:
    1. Register with master node
    2. Receive evaluation tasks
    3. Train and evaluate architectures
    4. Send results back to master
    5. Send periodic heartbeat
    6. Handle graceful shutdown

    Args:
        config: Worker configuration
            - worker_id: Unique worker ID (generated if not provided)
            - master_host: Master node hostname/IP (required)
            - master_port: Master node port (default: 50051)
            - port: Worker port (default: 50052)
            - num_gpus: Number of GPUs available (default: 1)
            - gpu_ids: Specific GPU IDs to use (default: range(num_gpus))
            - heartbeat_interval: Heartbeat interval (seconds, default: 10)
            - evaluator: Custom evaluation function

    Example:
        >>> worker = WorkerNode({
        ...     'master_host': 'localhost',
        ...     'master_port': 50051,
        ...     'port': 50052,
        ...     'num_gpus': 1,
        ...     'evaluator': my_eval_function
        ... })
        >>> worker.start()
        >>> # Worker runs until stopped
        >>> worker.stop()
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize worker node."""
        if not GRPC_AVAILABLE:
            raise DistributedError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )

        self.config = config

        # Worker identification
        self.worker_id = config.get("worker_id", str(uuid.uuid4()))
        self.master_host = config["master_host"]
        self.master_port = config.get("master_port", 50051)
        self.port = config.get("port", 50052)

        # GPU configuration
        self.num_gpus = config.get("num_gpus", 1)
        self.gpu_ids = config.get("gpu_ids", list(range(self.num_gpus)))

        # Evaluation configuration
        self.evaluator = config.get("evaluator")
        self.heartbeat_interval = config.get("heartbeat_interval", 10)

        # State
        self.running = False
        self.current_task_id: Optional[str] = None
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.start_time = time.time()

        # gRPC server
        self.server: Optional[grpc.Server] = None

        logger.info(
            f"Initialized WorkerNode (id={self.worker_id[:12]}) " f"with {self.num_gpus} GPU(s)"
        )

    def start(self) -> None:
        """Start worker node."""
        logger.info(f"Starting worker {self.worker_id[:12]} on port {self.port}")

        # Register with master
        self._register_with_master()

        # Start gRPC server
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=5),
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ],
        )

        # Add servicer
        worker_pb2_grpc.add_WorkerServiceServicer_to_server(WorkerServicer(self), self.server)

        # Start server
        self.server.add_insecure_port(f"[::]:{self.port}")
        self.server.start()

        self.running = True
        self.start_time = time.time()

        # Start heartbeat
        self._start_heartbeat()

        logger.info(f"Worker {self.worker_id[:12]} started successfully")

    def stop(self) -> None:
        """Stop worker gracefully."""
        logger.info(f"Stopping worker {self.worker_id[:12]}")
        self.running = False

        if self.server:
            self.server.stop(grace=5)

        logger.info(f"Worker {self.worker_id[:12]} stopped")

    def wait_for_shutdown(self) -> None:
        """Block until worker is stopped."""
        if self.server:
            self.server.wait_for_termination()

    def evaluate_architecture(
        self,
        architecture: ModelGraph,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate architecture.

        Args:
            architecture: ModelGraph to evaluate
            config: Evaluation configuration (epochs, batch_size, etc.)

        Returns:
            Dictionary with metrics (accuracy, params, latency, etc.)

        Example:
            >>> result = worker.evaluate_architecture(graph)
            >>> print(result['val_accuracy'])
        """
        logger.info(f"Evaluating architecture on worker {self.worker_id[:12]}")

        start_time = time.time()

        try:
            # Use custom evaluator if provided
            if self.evaluator:
                result = self.evaluator(architecture)

                # Ensure result is a dictionary
                if not isinstance(result, dict):
                    result = {"fitness": float(result)}

            else:
                # Default evaluation (heuristic)
                result = self._default_evaluation(architecture, config or {})

            # Add metadata
            result["worker_id"] = self.worker_id
            result["evaluation_time"] = time.time() - start_time
            result["gpu_id"] = self.gpu_ids[0] if self.gpu_ids else -1

            logger.debug(
                f"Evaluation complete in {result['evaluation_time']:.2f}s: "
                f"fitness={result.get('fitness', 'N/A')}"
            )

            return result

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def _default_evaluation(
        self, architecture: ModelGraph, config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Default heuristic evaluation when no evaluator provided.

        This is a fast proxy evaluation based on architecture properties.
        For actual training, provide a custom evaluator.
        """
        from morphml.evaluation import HeuristicEvaluator

        evaluator = HeuristicEvaluator()
        fitness = evaluator(architecture)

        # Estimate other metrics
        params = architecture.estimate_parameters()
        depth = len(list(architecture.topological_sort()))

        return {
            "fitness": fitness,
            "val_accuracy": fitness,
            "params": params,
            "depth": depth,
        }

    def _register_with_master(self) -> None:
        """Register with master node."""
        logger.info(f"Registering with master at {self.master_host}:{self.master_port}")

        max_retries = 10
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                channel = grpc.insecure_channel(f"{self.master_host}:{self.master_port}")
                stub = worker_pb2_grpc.MasterServiceStub(channel)

                request = worker_pb2.RegisterRequest(
                    worker_id=self.worker_id,
                    host=socket.gethostname(),
                    port=self.port,
                    num_gpus=self.num_gpus,
                    gpu_ids=self.gpu_ids,
                )

                response = stub.RegisterWorker(request, timeout=10)

                if response.success:
                    logger.info(f"Successfully registered with master (id={response.master_id})")
                    return
                else:
                    raise DistributedError(f"Registration failed: {response.message}")

            except grpc.RpcError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Registration attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    raise DistributedError(
                        f"Failed to register with master after {max_retries} attempts"
                    )

    def _start_heartbeat(self) -> None:
        """Start periodic heartbeat to master."""

        def heartbeat_loop() -> None:
            while self.running:
                try:
                    channel = grpc.insecure_channel(f"{self.master_host}:{self.master_port}")
                    stub = worker_pb2_grpc.MasterServiceStub(channel)

                    # Determine status
                    status = "busy" if self.current_task_id else "idle"

                    # Create metrics
                    metrics = worker_pb2.WorkerMetrics(
                        cpu_usage=self._get_cpu_usage(),
                        memory_usage=self._get_memory_usage(),
                        gpu_usage=self._get_gpu_usage(),
                        gpu_memory=self._get_gpu_memory(),
                        tasks_completed=self.tasks_completed,
                        tasks_failed=self.tasks_failed,
                    )

                    request = worker_pb2.HeartbeatRequest(
                        worker_id=self.worker_id,
                        status=status,
                        current_task_id=self.current_task_id or "",
                        metrics=metrics,
                    )

                    response = stub.Heartbeat(request, timeout=5)

                    # Check if master wants us to shutdown
                    if not response.should_continue:
                        logger.info("Master requested shutdown")
                        self.running = False
                        break

                except grpc.RpcError as e:
                    logger.error(f"Heartbeat failed: {e}")

                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")

                time.sleep(self.heartbeat_interval)

        thread = threading.Thread(target=heartbeat_loop, daemon=True, name="HeartbeatThread")
        thread.start()
        logger.debug("Heartbeat thread started")

    def _submit_result(
        self,
        task_id: str,
        success: bool,
        metrics: Dict[str, float],
        error: str = "",
        duration: float = 0.0,
    ) -> None:
        """Submit task result to master."""
        try:
            channel = grpc.insecure_channel(f"{self.master_host}:{self.master_port}")
            stub = worker_pb2_grpc.MasterServiceStub(channel)

            request = worker_pb2.ResultRequest(
                task_id=task_id,
                worker_id=self.worker_id,
                success=success,
                metrics=metrics,
                error=error,
                duration=duration,
            )

            response = stub.SubmitResult(request, timeout=10)

            if not response.acknowledged:
                logger.warning(f"Master did not acknowledge result for task {task_id}")

        except grpc.RpcError as e:
            logger.error(f"Failed to submit result for task {task_id}: {e}")

    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            import psutil

            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0

    def _get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        try:
            import psutil

            return psutil.virtual_memory().percent
        except Exception:
            return 0.0

    def _get_gpu_usage(self) -> float:
        """Get GPU usage percentage."""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_ids[0])
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(utilization.gpu)
        except Exception:
            return 0.0

    def _get_gpu_memory(self) -> float:
        """Get GPU memory usage percentage."""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_ids[0])
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return (mem_info.used / mem_info.total) * 100
        except Exception:
            return 0.0

    def get_status(self) -> Dict[str, Any]:
        """Get worker status."""
        return {
            "worker_id": self.worker_id,
            "status": "busy" if self.current_task_id else "idle",
            "current_task": self.current_task_id,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "uptime_seconds": time.time() - self.start_time,
            "cpu_usage": self._get_cpu_usage(),
            "memory_usage": self._get_memory_usage(),
            "gpu_usage": self._get_gpu_usage(),
            "gpu_memory": self._get_gpu_memory(),
        }


if GRPC_AVAILABLE:
    class WorkerServicer(worker_pb2_grpc.WorkerServiceServicer):
        """gRPC servicer for worker node."""

        def __init__(self, worker: WorkerNode):
            """Initialize servicer."""
            self.worker = worker

        def Evaluate(
            self, request: worker_pb2.EvaluateRequest, context: grpc.ServicerContext
        ) -> worker_pb2.EvaluateResponse:
            """Handle evaluation task."""
            task_id = request.task_id

            logger.info(f"Received evaluation task: {task_id}")

            self.worker.current_task_id = task_id
            start_time = time.time()

            try:
                # Deserialize architecture
                architecture = ModelGraph.from_json(request.architecture)

                # Evaluate
                result = self.worker.evaluate_architecture(architecture)

                duration = time.time() - start_time

                # Update stats
                self.worker.tasks_completed += 1
                self.worker.current_task_id = None

                # Submit result to master
                self.worker._submit_result(
                    task_id=task_id,
                    success=True,
                    metrics=result,
                    duration=duration,
                )

                # Return response
                return worker_pb2.EvaluateResponse(
                    task_id=task_id,
                    success=True,
                    metrics=result,
                    error="",
                    duration=duration,
                )

            except Exception as e:
                duration = time.time() - start_time
                error_msg = str(e)

                logger.error(f"Evaluation failed for task {task_id}: {error_msg}")

                # Update stats
                self.worker.tasks_failed += 1
                self.worker.current_task_id = None

                # Submit failure to master
                self.worker._submit_result(
                    task_id=task_id,
                    success=False,
                    metrics={},
                    error=error_msg,
                    duration=duration,
                )

                return worker_pb2.EvaluateResponse(
                    task_id=task_id,
                    success=False,
                    metrics={},
                    error=error_msg,
                    duration=duration,
                )

        def GetStatus(
            self, request: worker_pb2.StatusRequest, context: grpc.ServicerContext
        ) -> worker_pb2.StatusResponse:
            """Handle status request."""
            status = self.worker.get_status()

            metrics = worker_pb2.WorkerMetrics(
                cpu_usage=status["cpu_usage"],
                memory_usage=status["memory_usage"],
                gpu_usage=status["gpu_usage"],
                gpu_memory=status["gpu_memory"],
                tasks_completed=status["tasks_completed"],
                tasks_failed=status["tasks_failed"],
            )

            return worker_pb2.StatusResponse(
                status=status["status"],
                current_task_id=status["current_task"] or "",
                metrics=metrics,
                uptime_seconds=int(status["uptime_seconds"]),
            )

        def Shutdown(
            self, request: worker_pb2.ShutdownRequest, context: grpc.ServicerContext
        ) -> worker_pb2.ShutdownResponse:
            """Handle shutdown request."""
            logger.info(
                f"Shutdown requested (graceful={request.graceful}) for worker {request.worker_id}"
            )

            if request.graceful:
                # Wait for current task to finish
                while self.worker.current_task_id:
                    time.sleep(1)

            # Stop worker
            self.worker.stop()

            return worker_pb2.ShutdownResponse(acknowledged=True)

        def CancelTask(
            self, request: worker_pb2.CancelRequest, context: grpc.ServicerContext
        ) -> worker_pb2.CancelResponse:
            """Handle task cancellation."""
            task_id = request.task_id

            if self.worker.current_task_id == task_id:
                logger.warning(f"Cancelling task {task_id}")
                # Note: Actual cancellation would require more complex logic
                # For now, just clear the task ID
                self.worker.current_task_id = None

                return worker_pb2.CancelResponse(success=True, message="Task cancelled")
            else:
                return worker_pb2.CancelResponse(
                    success=False, message=f"Task {task_id} not running on this worker"
                )
