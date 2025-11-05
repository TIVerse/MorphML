"""Distributed execution module for MorphML.

This module provides distributed architecture search capabilities with:
- Master-worker coordination
- Task scheduling and distribution
- Fault tolerance and recovery
- Distributed storage and caching

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from morphml.distributed.fault_tolerance import (
    CircuitBreaker,
    FailureEvent,
    FailureType,
    FaultToleranceManager,
)
from morphml.distributed.health_monitor import (
    HealthMetrics,
    HealthMonitor,
    get_system_health,
    is_system_healthy,
)
from morphml.distributed.master import MasterNode, Task, WorkerInfo
from morphml.distributed.resource_manager import (
    GPUAffinityManager,
    ResourceManager,
    TaskRequirements,
    WorkerResources,
)
from morphml.distributed.scheduler import (
    AdaptiveScheduler,
    FIFOScheduler,
    LoadBalancingScheduler,
    PerformanceStats,
    PriorityScheduler,
    RoundRobinScheduler,
    TaskScheduler,
    WorkStealingScheduler,
    create_scheduler,
)
from morphml.distributed.worker import WorkerNode

__all__ = [
    # Core components
    "MasterNode",
    "WorkerNode",
    "WorkerInfo",
    "Task",
    # Schedulers
    "TaskScheduler",
    "FIFOScheduler",
    "PriorityScheduler",
    "LoadBalancingScheduler",
    "WorkStealingScheduler",
    "AdaptiveScheduler",
    "RoundRobinScheduler",
    "PerformanceStats",
    "create_scheduler",
    # Resource management
    "ResourceManager",
    "WorkerResources",
    "TaskRequirements",
    "GPUAffinityManager",
    # Fault tolerance
    "FaultToleranceManager",
    "CircuitBreaker",
    "FailureType",
    "FailureEvent",
    # Health monitoring
    "HealthMonitor",
    "HealthMetrics",
    "get_system_health",
    "is_system_healthy",
]
