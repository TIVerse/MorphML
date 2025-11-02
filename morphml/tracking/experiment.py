"""Experiment tracking for reproducibility and analysis."""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from morphml.core.graph import ModelGraph
from morphml.core.search import Individual
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class Experiment:
    """Single experiment run."""

    def __init__(self, name: str, config: Dict[str, Any], experiment_id: Optional[str] = None):
        """Initialize experiment."""
        self.name = name
        self.config = config
        self.id = experiment_id or self._generate_id()
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.metrics: Dict[str, List[Any]] = {}
        self.artifacts: Dict[str, str] = {}
        self.best_result: Optional[Dict[str, Any]] = None
        self.status = "running"

    def _generate_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.name}_{timestamp}"

    def log_metric(self, key: str, value: Any, step: Optional[int] = None) -> None:
        """Log a metric value."""
        if key not in self.metrics:
            self.metrics[key] = []

        entry = {"value": value}
        if step is not None:
            entry["step"] = step
        entry["timestamp"] = time.time()

        self.metrics[key].append(entry)

    def log_artifact(self, key: str, path: str) -> None:
        """Log an artifact path."""
        self.artifacts[key] = path

    def set_best_result(self, result: Dict[str, Any]) -> None:
        """Set best result."""
        self.best_result = result

    def finish(self, status: str = "completed") -> None:
        """Mark experiment as finished."""
        self.end_time = datetime.now()
        self.status = status
        logger.info(f"Experiment {self.id} finished with status: {status}")

    def get_duration(self) -> float:
        """Get experiment duration in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "config": self.config,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.get_duration(),
            "status": self.status,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "best_result": self.best_result,
        }


class ExperimentTracker:
    """
    Track multiple experiments.

    Example:
        >>> tracker = ExperimentTracker("./experiments")
        >>> exp = tracker.create_experiment("GA_CIFAR10", config={...})
        >>> exp.log_metric("fitness", 0.85, step=10)
        >>> exp.finish()
        >>> tracker.save_experiment(exp)
    """

    def __init__(self, base_dir: str = "./experiments"):
        """
        Initialize tracker.

        Args:
            base_dir: Base directory for experiments
        """
        self.base_dir = base_dir
        self.experiments: Dict[str, Experiment] = {}

        os.makedirs(base_dir, exist_ok=True)
        logger.info(f"ExperimentTracker initialized at {base_dir}")

    def create_experiment(
        self, name: str, config: Dict[str, Any], experiment_id: Optional[str] = None
    ) -> Experiment:
        """
        Create new experiment.

        Args:
            name: Experiment name
            config: Configuration dict
            experiment_id: Optional custom ID

        Returns:
            Experiment instance
        """
        exp = Experiment(name, config, experiment_id)
        self.experiments[exp.id] = exp

        # Create experiment directory
        exp_dir = os.path.join(self.base_dir, exp.id)
        os.makedirs(exp_dir, exist_ok=True)

        logger.info(f"Created experiment: {exp.id}")
        return exp

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self.experiments.get(experiment_id)

    def save_experiment(self, experiment: Experiment) -> None:
        """Save experiment to disk."""
        exp_dir = os.path.join(self.base_dir, experiment.id)
        os.makedirs(exp_dir, exist_ok=True)

        # Save metadata
        metadata_path = os.path.join(exp_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2)

        logger.info(f"Saved experiment {experiment.id} to {exp_dir}")

    def load_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Load experiment from disk."""
        exp_dir = os.path.join(self.base_dir, experiment_id)
        metadata_path = os.path.join(exp_dir, "metadata.json")

        if not os.path.exists(metadata_path):
            logger.warning(f"Experiment {experiment_id} not found")
            return None

        with open(metadata_path, "r") as f:
            data = json.load(f)

        exp = Experiment(data["name"], data["config"], data["id"])
        exp.start_time = datetime.fromisoformat(data["start_time"])
        if data["end_time"]:
            exp.end_time = datetime.fromisoformat(data["end_time"])
        exp.status = data["status"]
        exp.metrics = data["metrics"]
        exp.artifacts = data["artifacts"]
        exp.best_result = data["best_result"]

        self.experiments[exp.id] = exp
        logger.info(f"Loaded experiment {experiment_id}")

        return exp

    def list_experiments(self) -> List[str]:
        """List all experiment IDs."""
        experiments = []

        if not os.path.exists(self.base_dir):
            return experiments

        for item in os.listdir(self.base_dir):
            exp_dir = os.path.join(self.base_dir, item)
            if os.path.isdir(exp_dir):
                metadata_path = os.path.join(exp_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    experiments.append(item)

        return experiments

    def compare_experiments(self, experiment_ids: List[str], metric: str) -> Dict[str, List[float]]:
        """
        Compare experiments on a metric.

        Args:
            experiment_ids: List of experiment IDs
            metric: Metric name to compare

        Returns:
            Dictionary mapping experiment IDs to metric values
        """
        results = {}

        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            if not exp:
                exp = self.load_experiment(exp_id)

            if exp and metric in exp.metrics:
                values = [entry["value"] for entry in exp.metrics[metric]]
                results[exp_id] = values

        return results

    def get_best_experiment(self, metric: str = "best_fitness") -> Optional[Experiment]:
        """Get experiment with best result."""
        best_exp = None
        best_value = float("-inf")

        for exp_id in self.list_experiments():
            exp = self.get_experiment(exp_id)
            if not exp:
                exp = self.load_experiment(exp_id)

            if exp and exp.best_result and metric in exp.best_result:
                value = exp.best_result[metric]
                if value > best_value:
                    best_value = value
                    best_exp = exp

        return best_exp

    def export_summary(self, output_path: str) -> None:
        """Export summary of all experiments."""
        summary = {
            "base_dir": self.base_dir,
            "total_experiments": len(self.list_experiments()),
            "experiments": [],
        }

        for exp_id in self.list_experiments():
            exp = self.get_experiment(exp_id)
            if not exp:
                exp = self.load_experiment(exp_id)

            if exp:
                summary["experiments"].append(
                    {
                        "id": exp.id,
                        "name": exp.name,
                        "status": exp.status,
                        "duration": exp.get_duration(),
                        "best_result": exp.best_result,
                    }
                )

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Experiment summary exported to {output_path}")

    def clean_failed_experiments(self) -> int:
        """Remove failed experiments."""
        removed = 0

        for exp_id in self.list_experiments():
            exp = self.get_experiment(exp_id)
            if not exp:
                exp = self.load_experiment(exp_id)

            if exp and exp.status == "failed":
                exp_dir = os.path.join(self.base_dir, exp_id)
                import shutil

                shutil.rmtree(exp_dir)

                if exp_id in self.experiments:
                    del self.experiments[exp_id]

                removed += 1

        logger.info(f"Removed {removed} failed experiments")
        return removed


class RunContext:
    """Context manager for experiment runs."""

    def __init__(self, tracker: ExperimentTracker, name: str, config: Dict[str, Any]):
        """Initialize run context."""
        self.tracker = tracker
        self.name = name
        self.config = config
        self.experiment: Optional[Experiment] = None

    def __enter__(self) -> Experiment:
        """Enter context."""
        self.experiment = self.tracker.create_experiment(self.name, self.config)
        return self.experiment

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if self.experiment:
            if exc_type is None:
                self.experiment.finish("completed")
            else:
                self.experiment.finish("failed")
                logger.error(f"Experiment failed: {exc_val}")

            self.tracker.save_experiment(self.experiment)

        return False
