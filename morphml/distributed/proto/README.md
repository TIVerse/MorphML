# Protocol Buffer Stubs for gRPC Communication

This directory contains Protocol Buffer definitions and generated Python stubs for distributed communication between Master and Worker nodes.

## Files

- **`worker.proto`** - Protocol Buffer definition (source)
- **`worker_pb2.py`** - Generated message classes (auto-generated)
- **`worker_pb2_grpc.py`** - Generated gRPC service stubs (auto-generated)
- **`__init__.py`** - Package initialization with exports

## Services Defined

### MasterService (called by workers)
- `RegisterWorker()` - Register a new worker with the master
- `Heartbeat()` - Send periodic heartbeat to indicate worker is alive
- `SubmitResult()` - Submit evaluation results back to master
- `RequestTask()` - Request tasks from master (pull model)

### WorkerService (called by master)
- `Evaluate()` - Evaluate an architecture (push model)
- `GetStatus()` - Get current worker status
- `Shutdown()` - Gracefully shutdown worker
- `CancelTask()` - Cancel a running task

## Message Types

### Registration
- `RegisterRequest` / `RegisterResponse`

### Heartbeat
- `HeartbeatRequest` / `HeartbeatResponse`
- `WorkerMetrics` - CPU, memory, GPU usage stats

### Task Evaluation
- `EvaluateRequest` / `EvaluateResponse`
- `EvaluationConfig` - Training configuration
- `TaskRequest` / `TaskResponse`

### Results
- `ResultRequest` / `ResultResponse`

### Control
- `StatusRequest` / `StatusResponse`
- `ShutdownRequest` / `ShutdownResponse`
- `CancelRequest` / `CancelResponse`

## Regenerating Stubs

If you modify `worker.proto`, regenerate the Python stubs:

### Option 1: Using the provided script (Recommended)

```bash
python scripts/compile_protos.py
```

### Option 2: Manual compilation

```bash
python -m grpc_tools.protoc \
  -I morphml/distributed/proto \
  --python_out=morphml/distributed/proto \
  --grpc_python_out=morphml/distributed/proto \
  --pyi_out=morphml/distributed/proto \
  morphml/distributed/proto/worker.proto
```

**Note:** Requires `grpcio-tools` to be installed:
```bash
pip install grpcio-tools
# or with poetry
poetry install --extras distributed
```

## Usage

### Importing in Python

```python
from morphml.distributed.proto import (
    worker_pb2,
    worker_pb2_grpc,
    RegisterRequest,
    EvaluateRequest,
    # ... other message types
)
```

### Master Node Example

```python
import grpc
from morphml.distributed.proto import worker_pb2_grpc, RegisterResponse

class MyMasterServicer(worker_pb2_grpc.MasterServiceServicer):
    def RegisterWorker(self, request, context):
        print(f"Worker {request.worker_id} registered from {request.host}:{request.port}")
        return RegisterResponse(
            success=True,
            message="Registration successful",
            master_id="master-001"
        )

# Create server
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
worker_pb2_grpc.add_MasterServiceServicer_to_server(MyMasterServicer(), server)
server.add_insecure_port('[::]:50051')
server.start()
```

### Worker Node Example

```python
import grpc
from morphml.distributed.proto import worker_pb2, worker_pb2_grpc

# Connect to master
channel = grpc.insecure_channel('localhost:50051')
stub = worker_pb2_grpc.MasterServiceStub(channel)

# Register with master
request = worker_pb2.RegisterRequest(
    worker_id='worker-1',
    host='localhost',
    port=50052,
    num_gpus=2,
    gpu_ids=[0, 1]
)
response = stub.RegisterWorker(request)
print(f"Registration: {response.success} - {response.message}")
```

## Dependencies

Required packages:
- `grpcio>=1.54.0` - gRPC runtime
- `protobuf>=4.23.0` - Protocol Buffer runtime
- `grpcio-tools>=1.54.0` - For compilation (development only)

Install with:
```bash
pip install grpcio protobuf grpcio-tools
```

Or using poetry:
```bash
poetry install --extras distributed
```

## Communication Flow

```
Worker                          Master
  |                               |
  |-- RegisterWorker() ---------->|
  |<-- RegisterResponse ----------|
  |                               |
  |-- Heartbeat() (every 10s) -->|
  |<-- HeartbeatResponse ---------|
  |                               |
  |<-- Evaluate(task) ------------|  (push model)
  |-- SubmitResult(metrics) ----->|
  |                               |
  |-- RequestTask() ------------->|  (pull model)
  |<-- TaskResponse(tasks) -------|
  |                               |
```

## Protocol Details

- **Serialization:** Protocol Buffers v3 (binary)
- **Transport:** gRPC over HTTP/2
- **Default Ports:**
  - Master: 50051
  - Worker: 50052+
- **Timeout:** Configurable per-request
- **Compression:** Optional (gzip)

## Troubleshooting

### Import Errors

If you see:
```
ImportError: cannot import name 'worker_pb2' from 'morphml.distributed.proto'
```

**Solution:** Generate the protobuf files:
```bash
python scripts/compile_protos.py
```

### gRPC Connection Issues

If workers can't connect to master:
1. Check firewall settings
2. Verify master is running and listening
3. Check network connectivity: `telnet master-host 50051`
4. Review master logs for errors

### Version Incompatibility

Ensure compatible versions:
```bash
pip list | grep grpc
pip list | grep protobuf
```

Recommended versions:
- grpcio: ≥1.54.0, <2.0.0
- protobuf: ≥4.23.0, <5.0.0

## Learn More

- [gRPC Python Documentation](https://grpc.io/docs/languages/python/)
- [Protocol Buffers Guide](https://developers.google.com/protocol-buffers)
- [MorphML Distributed Module](/morphml/distributed/)

---

**Author:** Eshan Roy <eshanized@proton.me>  
**Organization:** TONMOY INFRASTRUCTURE & VISION  
**Auto-generated:** Do not edit `*_pb2.py` files manually
