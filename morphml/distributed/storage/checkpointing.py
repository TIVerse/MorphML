"""Checkpoint management for experiment recovery.

Enables saving and restoring experiment state for fault tolerance.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import pickle
import time
from typing import Any, Dict, List, Optional

from morphml.core.search import Individual
from morphml.distributed.storage.artifacts import ArtifactStore
from morphml.distributed.storage.cache import DistributedCache
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """
    Manage experiment checkpoints for recovery.

    Uses both cache (fast) and artifact storage (persistent):
    - Cache: Recent checkpoints for quick recovery
    - Storage: All checkpoints for long-term persistence

    Args:
        artifact_store: ArtifactStore for persistent storage
        cache: DistributedCache for fast access (optional)
        checkpoint_interval: Save checkpoint every N generations
        max_checkpoints: Maximum checkpoints to keep (0 = unlimited)

    Example:
        >>> manager = CheckpointManager(artifact_store, cache)
        >>>
        >>> # Save checkpoint
        >>> manager.save_checkpoint(
        ...     experiment_id='exp1',
        ...     generation=10,
        ...     optimizer_state={'population_size': 50},
        ...     population=population
        ... )
        >>>
        >>> # Load latest checkpoint
        >>> checkpoint = manager.load_checkpoint('exp1')
        >>> if checkpoint:
        ...     generation = checkpoint['generation']
        ...     optimizer_state = checkpoint['optimizer_state']
    """

    def __init__(
        self,
        artifact_store: ArtifactStore,
        cache: Optional[DistributedCache] = None,
        checkpoint_interval: int = 10,
        max_checkpoints: int = 5,
    ):
        """Initialize checkpoint manager."""
        self.store = artifact_store
        self.cache = cache
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints

        logger.info(
            f"Initialized CheckpointManager "
            f"(interval={checkpoint_interval}, max={max_checkpoints})"
        )

    def save_checkpoint(
        self,
        experiment_id: str,
        generation: int,
        optimizer_state: Dict[str, Any],
        population: Optional[List[Individual]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save experiment checkpoint.

        Args:
            experiment_id: Experiment identifier
            generation: Current generation number
            optimizer_state: Optimizer state dictionary
            population: Current population (optional)
            metadata: Additional metadata (optional)
        """
        checkpoint = {
            "experiment_id": experiment_id,
            "generation": generation,
            "optimizer_state": optimizer_state,
            "population": [ind.to_dict() for ind in population] if population else [],
            "metadata": metadata or {},
            "timestamp": time.time(),
        }

        # Save to cache for fast recovery
        if self.cache:
            try:
                self.cache.set(f"checkpoint:{experiment_id}:latest", checkpoint, ttl=3600)
                self.cache.set(
                    f"checkpoint:{experiment_id}:gen{generation}",
                    checkpoint,
                    ttl=7200,
                )
                logger.debug(f"Cached checkpoint for generation {generation}")
            except Exception as e:
                logger.warning(f"Failed to cache checkpoint: {e}")

        # Save to persistent storage
        try:
            # Serialize checkpoint
            checkpoint_bytes = pickle.dumps(checkpoint)

            # Upload to S3
            s3_key = f"checkpoints/{experiment_id}/gen_{generation:06d}.pkl"
            self.store.upload_bytes(
                checkpoint_bytes,
                s3_key,
                metadata={
                    "experiment_id": experiment_id,
                    "generation": str(generation),
                    "timestamp": str(int(time.time())),
                },
            )

            logger.info(f"Saved checkpoint for {experiment_id} generation {generation}")

            # Clean up old checkpoints
            if self.max_checkpoints > 0:
                self._cleanup_old_checkpoints(experiment_id)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(
        self, experiment_id: str, generation: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint.

        Args:
            experiment_id: Experiment identifier
            generation: Specific generation (None = latest)

        Returns:
            Checkpoint dictionary or None if not found
        """
        # Try cache first (latest only)
        if generation is None and self.cache:
            try:
                checkpoint = self.cache.get(f"checkpoint:{experiment_id}:latest")

                if checkpoint:
                    logger.info("Loaded checkpoint from cache (latest)")
                    return checkpoint
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")

        # Load from persistent storage
        try:
            if generation is not None:
                # Load specific generation
                s3_key = f"checkpoints/{experiment_id}/gen_{generation:06d}.pkl"

                if not self.store.exists(s3_key):
                    logger.warning(f"Checkpoint not found for generation {generation}")
                    return None

                checkpoint_bytes = self.store.download_bytes(s3_key)
                checkpoint = pickle.loads(checkpoint_bytes)

                logger.info(f"Loaded checkpoint for {experiment_id} generation {generation}")

                return checkpoint

            else:
                # Load latest
                checkpoints = self.list_checkpoints(experiment_id)

                if not checkpoints:
                    logger.info(f"No checkpoints found for {experiment_id}")
                    return None

                # Get latest
                latest = checkpoints[-1]
                s3_key = latest["key"]

                checkpoint_bytes = self.store.download_bytes(s3_key)
                checkpoint = pickle.loads(checkpoint_bytes)

                logger.info(
                    f"Loaded latest checkpoint for {experiment_id} "
                    f"(generation {checkpoint['generation']})"
                )

                return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def list_checkpoints(self, experiment_id: str) -> List[Dict[str, Any]]:
        """
        List all checkpoints for experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            List of checkpoint metadata dictionaries
        """
        prefix = f"checkpoints/{experiment_id}/"

        try:
            checkpoints = self.store.list_objects(prefix)

            # Sort by generation (extracted from key)
            checkpoints.sort(key=lambda x: x["key"])

            return checkpoints

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []

    def delete_checkpoint(self, experiment_id: str, generation: int) -> None:
        """
        Delete specific checkpoint.

        Args:
            experiment_id: Experiment identifier
            generation: Generation number
        """
        s3_key = f"checkpoints/{experiment_id}/gen_{generation:06d}.pkl"

        try:
            self.store.delete(s3_key)
            logger.info(f"Deleted checkpoint for generation {generation}")

        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")

    def delete_all_checkpoints(self, experiment_id: str) -> int:
        """
        Delete all checkpoints for experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Number of checkpoints deleted
        """
        prefix = f"checkpoints/{experiment_id}/"

        try:
            deleted = self.store.delete_prefix(prefix)
            logger.info(f"Deleted {deleted} checkpoints for {experiment_id}")

            # Also clear cache
            if self.cache:
                self.cache.invalidate_pattern(f"checkpoint:{experiment_id}:*")

            return deleted

        except Exception as e:
            logger.error(f"Failed to delete checkpoints: {e}")
            return 0

    def _cleanup_old_checkpoints(self, experiment_id: str) -> None:
        """Remove old checkpoints keeping only max_checkpoints."""
        checkpoints = self.list_checkpoints(experiment_id)

        if len(checkpoints) <= self.max_checkpoints:
            return

        # Delete oldest checkpoints
        num_to_delete = len(checkpoints) - self.max_checkpoints

        for checkpoint in checkpoints[:num_to_delete]:
            try:
                self.store.delete(checkpoint["key"])
                logger.debug(f"Cleaned up old checkpoint: {checkpoint['key']}")
            except Exception as e:
                logger.warning(f"Failed to cleanup checkpoint: {e}")

    def should_checkpoint(self, generation: int) -> bool:
        """
        Check if should save checkpoint at this generation.

        Args:
            generation: Current generation

        Returns:
            True if should checkpoint
        """
        return generation % self.checkpoint_interval == 0

    def get_statistics(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get checkpoint statistics.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Statistics dictionary
        """
        checkpoints = self.list_checkpoints(experiment_id)

        if not checkpoints:
            return {
                "experiment_id": experiment_id,
                "num_checkpoints": 0,
                "total_size_bytes": 0,
            }

        total_size = sum(cp["size"] for cp in checkpoints)

        return {
            "experiment_id": experiment_id,
            "num_checkpoints": len(checkpoints),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "oldest": checkpoints[0]["last_modified"],
            "newest": checkpoints[-1]["last_modified"],
        }
