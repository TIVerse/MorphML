"""Experiment database for meta-learning knowledge base.

Stores and retrieves past experiments for transfer learning.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)

try:
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


@dataclass
class TaskMetadata:
    """
    Metadata describing a machine learning task.

    Used for task similarity computation and warm-starting.

    Attributes:
        task_id: Unique task identifier
        dataset_name: Name of dataset (e.g., 'CIFAR-10')
        num_samples: Number of training samples
        num_classes: Number of output classes
        input_size: Input tensor shape (C, H, W)
        problem_type: Type of problem ('classification', 'detection', 'segmentation')
        metadata: Additional task-specific metadata
    """

    task_id: str
    dataset_name: str
    num_samples: int
    num_classes: int
    input_size: Tuple[int, int, int]  # (C, H, W)
    problem_type: str = "classification"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "dataset_name": self.dataset_name,
            "num_samples": self.num_samples,
            "num_classes": self.num_classes,
            "input_size": list(self.input_size),
            "problem_type": self.problem_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskMetadata":
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            dataset_name=data["dataset_name"],
            num_samples=data["num_samples"],
            num_classes=data["num_classes"],
            input_size=tuple(data["input_size"]),
            problem_type=data.get("problem_type", "classification"),
            metadata=data.get("metadata", {}),
        )


class ExperimentDatabase:
    """
    Database interface for past experiments.

    Provides access to experiment history for meta-learning:
    - Query tasks by similarity
    - Retrieve top architectures
    - Get search trajectories
    - Store new experiments

    Args:
        db_manager: DatabaseManager instance (if using SQL backend)
        storage_path: Path to local storage (if using file backend)

    Example:
        >>> db = ExperimentDatabase(db_manager)
        >>> tasks = db.get_all_tasks()
        >>> archs = db.get_top_architectures('exp1', top_k=10)
    """

    def __init__(self, db_manager: Optional[Any] = None, storage_path: Optional[str] = None):
        """Initialize experiment database."""
        self.db_manager = db_manager
        self.storage_path = storage_path

        # In-memory cache
        self._task_cache: Dict[str, TaskMetadata] = {}
        self._arch_cache: Dict[str, List[ModelGraph]] = {}

        logger.info("Initialized ExperimentDatabase")

    def add_task(self, task: TaskMetadata) -> None:
        """
        Add task metadata to database.

        Args:
            task: Task metadata
        """
        self._task_cache[task.task_id] = task
        logger.debug(f"Added task {task.task_id} to database")

    def get_all_tasks(self) -> List[TaskMetadata]:
        """
        Get all past tasks.

        Returns:
            List of task metadata
        """
        if self.db_manager and DB_AVAILABLE:
            # Query from SQL database
            try:
                experiments = self.db_manager.list_experiments()
                tasks = [self._experiment_to_task_metadata(exp) for exp in experiments]
                return tasks
            except Exception as e:
                logger.warning(f"Failed to query database: {e}")

        # Return from cache
        return list(self._task_cache.values())

    def get_task(self, task_id: str) -> Optional[TaskMetadata]:
        """
        Get specific task metadata.

        Args:
            task_id: Task identifier

        Returns:
            Task metadata or None
        """
        return self._task_cache.get(task_id)

    def get_top_architectures(self, task_id: str, top_k: int = 10) -> List[ModelGraph]:
        """
        Get top-k architectures for a task.

        Args:
            task_id: Task identifier
            top_k: Number of architectures to return

        Returns:
            List of best architectures
        """
        # Check cache first
        if task_id in self._arch_cache:
            cached = self._arch_cache[task_id]
            return cached[:top_k]

        if self.db_manager and DB_AVAILABLE:
            # Query from SQL database
            try:
                best = self.db_manager.get_best_architectures(task_id, top_k=top_k)
                graphs = []

                for arch in best:
                    try:
                        graph_dict = json.loads(arch.architecture_json)
                        graph = ModelGraph.from_dict(graph_dict)
                        graphs.append(graph)
                    except Exception as e:
                        logger.warning(f"Failed to deserialize architecture: {e}")

                # Cache results
                self._arch_cache[task_id] = graphs

                return graphs
            except Exception as e:
                logger.warning(f"Failed to query architectures: {e}")

        return []

    def add_architecture(self, task_id: str, graph: ModelGraph, fitness: float) -> None:
        """
        Add architecture to database.

        Args:
            task_id: Task identifier
            graph: Architecture graph
            fitness: Performance score
        """
        if task_id not in self._arch_cache:
            self._arch_cache[task_id] = []

        self._arch_cache[task_id].append(graph)

        # Sort by fitness (need to store fitness too)
        # For now just append
        logger.debug(f"Added architecture to task {task_id}")

    def _experiment_to_task_metadata(self, experiment: Any) -> TaskMetadata:
        """Convert database Experiment to TaskMetadata."""
        try:
            config = json.loads(experiment.config) if hasattr(experiment, "config") else {}
            (json.loads(experiment.search_space) if hasattr(experiment, "search_space") else {})

            # Extract metadata
            return TaskMetadata(
                task_id=str(experiment.id),
                dataset_name=experiment.name,
                num_samples=config.get("num_samples", 50000),
                num_classes=config.get("num_classes", 10),
                input_size=tuple(config.get("input_size", [3, 32, 32])),
                problem_type=config.get("problem_type", "classification"),
                metadata={
                    "optimizer": experiment.optimizer,
                    "status": experiment.status,
                    "num_evaluations": experiment.num_evaluations,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to convert experiment: {e}")
            # Return default
            return TaskMetadata(
                task_id=str(experiment.id),
                dataset_name=experiment.name,
                num_samples=50000,
                num_classes=10,
                input_size=(3, 32, 32),
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "num_tasks": len(self._task_cache),
            "num_cached_architectures": sum(len(archs) for archs in self._arch_cache.values()),
        }
