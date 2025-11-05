"""Generated Protocol Buffer stubs for distributed communication.

Auto-generated files. To regenerate, run:
    python scripts/compile_protos.py

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

try:
    from morphml.distributed.proto.worker_pb2 import (
        CancelRequest,
        CancelResponse,
        EvaluateRequest,
        EvaluateResponse,
        EvaluationConfig,
        HeartbeatRequest,
        HeartbeatResponse,
        RegisterRequest,
        RegisterResponse,
        ResultRequest,
        ResultResponse,
        ShutdownRequest,
        ShutdownResponse,
        StatusRequest,
        StatusResponse,
        TaskRequest,
        TaskResponse,
        WorkerMetrics,
    )
    from morphml.distributed.proto.worker_pb2_grpc import (
        MasterService,
        MasterServiceServicer,
        MasterServiceStub,
        WorkerService,
        WorkerServiceServicer,
        WorkerServiceStub,
        add_MasterServiceServicer_to_server,
        add_WorkerServiceServicer_to_server,
    )

    __all__ = [
        # Messages
        "RegisterRequest",
        "RegisterResponse",
        "HeartbeatRequest",
        "HeartbeatResponse",
        "WorkerMetrics",
        "EvaluateRequest",
        "EvaluateResponse",
        "EvaluationConfig",
        "TaskRequest",
        "TaskResponse",
        "ResultRequest",
        "ResultResponse",
        "StatusRequest",
        "StatusResponse",
        "ShutdownRequest",
        "ShutdownResponse",
        "CancelRequest",
        "CancelResponse",
        # Services
        "MasterServiceStub",
        "MasterServiceServicer",
        "MasterService",
        "add_MasterServiceServicer_to_server",
        "WorkerServiceStub",
        "WorkerServiceServicer",
        "WorkerService",
        "add_WorkerServiceServicer_to_server",
    ]
except ImportError:
    # Protobuf files not generated or grpc not installed
    pass
