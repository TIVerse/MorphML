"""Distributed storage backends for MorphML.

Provides persistent storage for:
- Experiment results (PostgreSQL)
- Fast caching (Redis)
- Model artifacts (S3/MinIO)
- Checkpoints (combined storage)

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from morphml.distributed.storage.artifacts import ArtifactStore
from morphml.distributed.storage.cache import DistributedCache
from morphml.distributed.storage.checkpointing import CheckpointManager
from morphml.distributed.storage.database import (
    Architecture,
    DatabaseManager,
    Experiment,
)

__all__ = [
    # Database
    "DatabaseManager",
    "Experiment",
    "Architecture",
    # Cache
    "DistributedCache",
    # Artifacts
    "ArtifactStore",
    # Checkpointing
    "CheckpointManager",
]
