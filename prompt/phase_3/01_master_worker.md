# Component 1: Master-Worker Architecture

**Duration:** Weeks 1-2  
**LOC Target:** ~5,000  
**Dependencies:** Phase 1-2 complete

---

## 🎯 Objective

Implement distributed master-worker architecture for parallel architecture evaluation:
1. **Master Node** - Coordinates search, distributes tasks
2. **Worker Nodes** - Evaluate architectures in parallel
3. **Communication** - gRPC or ZMQ for message passing
4. **Resource Management** - Track worker availability and load

---

## 📋 Architecture Overview

```
┌─────────────────────────────┐
│       Master Node           │
│  - Optimizer (GA/BO/etc)    │
│  - Task Queue               │
│  - Result Aggregator        │
│  - Worker Registry          │
└─────────────────────────────┘
        │ ↓ Tasks
        │ ↑ Results
   ┌────┴────┬────────┬────────┐
   ↓         ↓        ↓        ↓
┌──────┐  ┌──────┐ ┌──────┐  ┌──────┐
│Worker│  │Worker│ │Worker│  │Worker│
│  1   │  │  2   │ │  3   │  │  N   │
│ GPU  │  │ GPU  │ │ GPU  │  │ GPU  │
└──────┘  └──────┘ └──────┘  └──────┘
```

---

## 📋 Files to Create

### 1. `distributed/master.py` (~2,000 LOC)

**`MasterNode` class:**

```python
import grpc
from concurrent import futures
from typing import List, Dict, Optional, Callable
from queue import Queue, Empty
import threading

from morphml.distributed.proto import worker_pb2, worker_pb2_grpc
from morphml.core.optimizer import BaseOptimizer
from morphml.core.graph import ModelGraph

class MasterNode:
    """
    Master node for distributed architecture search.
    
    Responsibilities:
    1. Run optimization algorithm (GA, BO, etc.)
    2. Distribute architecture evaluation tasks to workers
    3. Collect results and update optimizer
    4. Monitor worker health
    5. Handle failures and task reassignment
    
    Config:
        host: Master host address (default: '0.0.0.0')
        port: Master port (default: 50051)
        num_workers: Expected number of workers
        heartbeat_interval: Worker heartbeat check (seconds, default: 10)
        task_timeout: Task timeout (seconds, default: 3600)
    """
    
    def __init__(self, optimizer: BaseOptimizer, config: Dict[str, Any]):
        self.optimizer = optimizer
        self.config = config
        
        self.host = config.get('host', '0.0.0.0')
        self.port = config.get('port', 50051)
        self.num_workers = config.get('num_workers', 4)
        self.heartbeat_interval = config.get('heartbeat_interval', 10)
        self.task_timeout = config.get('task_timeout', 3600)
        
        # Worker registry
        self.workers: Dict[str, WorkerInfo] = {}
        self.worker_lock = threading.Lock()
        
        # Task management
        self.pending_tasks: Queue = Queue()
        self.running_tasks: Dict[str, Task] = {}  # task_id -> Task
        self.completed_tasks: Dict[str, Task] = {}
        self.task_lock = threading.Lock()
        
        # gRPC server
        self.server = None
        
        # State
        self.running = False
    
    def start(self):
        """Start master node server."""
        logger.info(f"Starting master node on {self.host}:{self.port}")
        
        # Create gRPC server
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        worker_pb2_grpc.add_MasterServiceServicer_to_server(
            MasterServicer(self), self.server
        )
        self.server.add_insecure_port(f'{self.host}:{self.port}')
        self.server.start()
        
        self.running = True
        
        # Start background threads
        self._start_heartbeat_monitor()
        self._start_task_dispatcher()
        
        logger.info("Master node started successfully")
    
    def stop(self):
        """Stop master node."""
        logger.info("Stopping master node")
        self.running = False
        
        if self.server:
            self.server.stop(grace=5)
        
        logger.info("Master node stopped")
    
    def register_worker(self, worker_id: str, worker_info: Dict[str, Any]):
        """
        Register a worker.
        
        Args:
            worker_id: Unique worker identifier
            worker_info: Worker metadata (host, port, GPUs, etc.)
        """
        with self.worker_lock:
            self.workers[worker_id] = WorkerInfo(
                worker_id=worker_id,
                host=worker_info['host'],
                port=worker_info['port'],
                num_gpus=worker_info.get('num_gpus', 1),
                status='idle',
                last_heartbeat=time.time()
            )
            
            logger.info(f"Worker registered: {worker_id} ({worker_info['host']}:{worker_info['port']})")
    
    def submit_task(self, architecture: ModelGraph, task_id: Optional[str] = None) -> str:
        """
        Submit architecture evaluation task.
        
        Returns:
            task_id
        """
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            architecture=architecture,
            status='pending',
            created_at=time.time()
        )
        
        with self.task_lock:
            self.pending_tasks.put(task)
        
        logger.debug(f"Task submitted: {task_id}")
        
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[Dict]:
        """
        Get result for task (blocking).
        
        Args:
            task_id: Task ID
            timeout: Maximum wait time (seconds)
        
        Returns:
            Task result or None if timeout
        """
        start_time = time.time()
        
        while True:
            with self.task_lock:
                if task_id in self.completed_tasks:
                    task = self.completed_tasks[task_id]
                    return task.result
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            time.sleep(0.1)
    
    def run_experiment(self, num_generations: int = 100) -> List[ModelGraph]:
        """
        Run full NAS experiment.
        
        Args:
            num_generations: Number of optimization generations
        
        Returns:
            List of best architectures found
        """
        logger.info(f"Starting distributed experiment ({num_generations} generations)")
        
        # Wait for workers
        self._wait_for_workers()
        
        # Initialize optimizer population
        population = self.optimizer.initialize()
        
        for generation in range(num_generations):
            logger.info(f"Generation {generation}/{num_generations}")
            
            # Submit evaluation tasks
            task_ids = []
            for individual in population:
                task_id = self.submit_task(individual.genome)
                task_ids.append((individual, task_id))
            
            # Wait for results
            for individual, task_id in task_ids:
                result = self.get_result(task_id, timeout=self.task_timeout)
                
                if result is None:
                    logger.warning(f"Task {task_id} timeout")
                    individual.fitness = 0.0
                else:
                    individual.fitness = result['fitness']
            
            # Optimizer step (selection, crossover, mutation)
            population = self.optimizer.step(population)
            
            # Log progress
            best_fitness = max(ind.fitness for ind in population)
            avg_fitness = np.mean([ind.fitness for ind in population])
            
            logger.info(
                f"Gen {generation}: "
                f"best={best_fitness:.4f}, "
                f"avg={avg_fitness:.4f}, "
                f"workers={len(self.workers)}"
            )
        
        # Return best individuals
        population.sort(key=lambda x: x.fitness, reverse=True)
        return [ind.genome for ind in population[:10]]
    
    def _wait_for_workers(self, timeout: float = 300):
        """Wait for workers to connect."""
        logger.info(f"Waiting for {self.num_workers} workers...")
        
        start_time = time.time()
        while len(self.workers) < self.num_workers:
            if (time.time() - start_time) > timeout:
                raise TimeoutError(f"Only {len(self.workers)}/{self.num_workers} workers connected")
            
            time.sleep(1)
        
        logger.info(f"All {self.num_workers} workers connected")
    
    def _start_heartbeat_monitor(self):
        """Monitor worker heartbeats."""
        def monitor():
            while self.running:
                current_time = time.time()
                
                with self.worker_lock:
                    for worker_id, worker in list(self.workers.items()):
                        # Check if heartbeat is stale
                        if (current_time - worker.last_heartbeat) > (self.heartbeat_interval * 3):
                            logger.warning(f"Worker {worker_id} heartbeat timeout")
                            worker.status = 'dead'
                            
                            # Reassign tasks from dead worker
                            self._reassign_worker_tasks(worker_id)
                
                time.sleep(self.heartbeat_interval)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def _start_task_dispatcher(self):
        """Dispatch tasks to available workers."""
        def dispatch():
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
                        # No workers available, re-queue
                        self.pending_tasks.put(task)
                        time.sleep(0.5)
                
                except Empty:
                    continue
        
        thread = threading.Thread(target=dispatch, daemon=True)
        thread.start()
    
    def _find_available_worker(self) -> Optional[WorkerInfo]:
        """Find idle worker."""
        with self.worker_lock:
            for worker in self.workers.values():
                if worker.status == 'idle':
                    return worker
        return None
    
    def _dispatch_task_to_worker(self, task: Task, worker: WorkerInfo):
        """Dispatch task to specific worker."""
        logger.debug(f"Dispatching task {task.task_id} to worker {worker.worker_id}")
        
        # Update task status
        task.status = 'running'
        task.worker_id = worker.worker_id
        task.started_at = time.time()
        
        with self.task_lock:
            self.running_tasks[task.task_id] = task
        
        # Update worker status
        worker.status = 'busy'
        worker.current_task = task.task_id
        
        # Send task via gRPC
        try:
            channel = grpc.insecure_channel(f'{worker.host}:{worker.port}')
            stub = worker_pb2_grpc.WorkerServiceStub(channel)
            
            request = worker_pb2.EvaluateRequest(
                task_id=task.task_id,
                architecture=task.architecture.to_json()
            )
            
            response = stub.Evaluate(request, timeout=self.task_timeout)
            
            # Handle result
            self._handle_task_result(task.task_id, response)
            
        except grpc.RpcError as e:
            logger.error(f"Failed to dispatch task {task.task_id}: {e}")
            self._handle_task_failure(task.task_id)


@dataclass
class WorkerInfo:
    """Worker metadata."""
    worker_id: str
    host: str
    port: int
    num_gpus: int
    status: str  # 'idle', 'busy', 'dead'
    last_heartbeat: float
    current_task: Optional[str] = None


@dataclass
class Task:
    """Evaluation task."""
    task_id: str
    architecture: ModelGraph
    status: str  # 'pending', 'running', 'completed', 'failed'
    created_at: float
    worker_id: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict] = None
```

---

### 2. `distributed/worker.py` (~1,500 LOC)

**`WorkerNode` class:**

```python
class WorkerNode:
    """
    Worker node for architecture evaluation.
    
    Responsibilities:
    1. Register with master
    2. Receive evaluation tasks
    3. Train and evaluate architectures
    4. Send results back to master
    5. Send heartbeat to master
    
    Config:
        worker_id: Unique worker identifier
        master_host: Master node host
        master_port: Master node port
        port: Worker port (default: 50052)
        num_gpus: Number of GPUs (default: 1)
        gpu_ids: Specific GPU IDs to use
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        self.worker_id = config.get('worker_id', str(uuid.uuid4()))
        self.master_host = config['master_host']
        self.master_port = config['master_port']
        self.port = config.get('port', 50052)
        self.num_gpus = config.get('num_gpus', 1)
        self.gpu_ids = config.get('gpu_ids', list(range(self.num_gpus)))
        
        # State
        self.running = False
        self.current_task = None
        
        # gRPC server
        self.server = None
    
    def start(self):
        """Start worker node."""
        logger.info(f"Starting worker {self.worker_id}")
        
        # Register with master
        self._register_with_master()
        
        # Start gRPC server
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
        worker_pb2_grpc.add_WorkerServiceServicer_to_server(
            WorkerServicer(self), self.server
        )
        self.server.add_insecure_port(f'[::]:{self.port}')
        self.server.start()
        
        self.running = True
        
        # Start heartbeat
        self._start_heartbeat()
        
        logger.info(f"Worker {self.worker_id} started on port {self.port}")
    
    def stop(self):
        """Stop worker."""
        logger.info(f"Stopping worker {self.worker_id}")
        self.running = False
        
        if self.server:
            self.server.stop(grace=5)
    
    def evaluate_architecture(
        self,
        architecture: ModelGraph,
        dataset: DataLoader,
        num_epochs: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate architecture.
        
        Returns:
            Dictionary with metrics (accuracy, latency, params, etc.)
        """
        logger.info(f"Evaluating architecture on worker {self.worker_id}")
        
        # Select GPU
        gpu_id = self.gpu_ids[0]
        device = torch.device(f'cuda:{gpu_id}')
        
        # Build model
        model = build_model_from_graph(architecture)
        model = model.to(device)
        
        # Train
        trainer = Trainer(model, dataset, device=device)
        results = trainer.train(num_epochs)
        
        # Evaluate metrics
        metrics = {
            'val_accuracy': results['val_accuracy'],
            'test_accuracy': results.get('test_accuracy', 0.0),
            'params': count_parameters(model),
            'latency': measure_latency(model, device),
            'worker_id': self.worker_id
        }
        
        return metrics
    
    def _register_with_master(self):
        """Register with master node."""
        channel = grpc.insecure_channel(f'{self.master_host}:{self.master_port}')
        stub = worker_pb2_grpc.MasterServiceStub(channel)
        
        request = worker_pb2.RegisterRequest(
            worker_id=self.worker_id,
            host=socket.gethostname(),
            port=self.port,
            num_gpus=self.num_gpus
        )
        
        response = stub.RegisterWorker(request)
        
        if response.success:
            logger.info(f"Successfully registered with master")
        else:
            raise RuntimeError(f"Failed to register with master: {response.message}")
    
    def _start_heartbeat(self):
        """Send periodic heartbeat to master."""
        def heartbeat():
            while self.running:
                try:
                    channel = grpc.insecure_channel(f'{self.master_host}:{self.master_port}')
                    stub = worker_pb2_grpc.MasterServiceStub(channel)
                    
                    request = worker_pb2.HeartbeatRequest(
                        worker_id=self.worker_id,
                        status='busy' if self.current_task else 'idle'
                    )
                    
                    stub.Heartbeat(request)
                    
                except grpc.RpcError as e:
                    logger.error(f"Heartbeat failed: {e}")
                
                time.sleep(10)  # Heartbeat every 10 seconds
        
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
```

---

### 3. `distributed/proto/worker.proto` (~300 LOC)

**gRPC protocol definition:**

```protobuf
syntax = "proto3";

package morphml;

// Master service (called by workers)
service MasterService {
    rpc RegisterWorker(RegisterRequest) returns (RegisterResponse);
    rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);
    rpc SubmitResult(ResultRequest) returns (ResultResponse);
}

// Worker service (called by master)
service WorkerService {
    rpc Evaluate(EvaluateRequest) returns (EvaluateResponse);
    rpc GetStatus(StatusRequest) returns (StatusResponse);
    rpc Shutdown(ShutdownRequest) returns (ShutdownResponse);
}

message RegisterRequest {
    string worker_id = 1;
    string host = 2;
    int32 port = 3;
    int32 num_gpus = 4;
}

message RegisterResponse {
    bool success = 1;
    string message = 2;
}

message HeartbeatRequest {
    string worker_id = 1;
    string status = 2;  // 'idle', 'busy'
}

message HeartbeatResponse {
    bool acknowledged = 1;
}

message EvaluateRequest {
    string task_id = 1;
    string architecture = 2;  // JSON serialized
}

message EvaluateResponse {
    string task_id = 1;
    bool success = 2;
    map<string, double> metrics = 3;
    string error = 4;
}

message ResultRequest {
    string task_id = 1;
    string worker_id = 2;
    map<string, double> metrics = 3;
}

message ResultResponse {
    bool acknowledged = 1;
}
```

---

## 🧪 Tests

**`test_master_worker.py`:**
```python
def test_worker_registration():
    """Test worker registration with master."""
    master = MasterNode(optimizer, {'port': 50051})
    master.start()
    
    worker = WorkerNode({
        'master_host': 'localhost',
        'master_port': 50051,
        'port': 50052
    })
    worker.start()
    
    time.sleep(2)  # Wait for registration
    
    assert len(master.workers) == 1
    
    worker.stop()
    master.stop()
```

---

## ✅ Deliverables

- [ ] Master node with task distribution
- [ ] Worker node with evaluation
- [ ] gRPC communication protocol
- [ ] Worker registry and health monitoring
- [ ] Task queue management
- [ ] Tests for distributed execution

---

**Next:** `02_task_scheduling.md`
