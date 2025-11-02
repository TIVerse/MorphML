"""Metric logging utilities."""

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

from morphml.logging_config import get_logger

logger = get_logger(__name__)


class MetricLogger:
    """
    Log metrics during optimization.

    Example:
        >>> logger = MetricLogger()
        >>> logger.log("fitness", 0.85, step=10)
        >>> logger.log("diversity", 0.45, step=10)
        >>> logger.save("metrics.json")
    """

    def __init__(self):
        """Initialize metric logger."""
        self.metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.step_counter = 0

    def log(self, metric_name: str, value: Any, step: Optional[int] = None) -> None:
        """
        Log a metric value.

        Args:
            metric_name: Name of metric
            value: Metric value
            step: Optional step/generation number
        """
        if step is None:
            step = self.step_counter
            self.step_counter += 1

        self.metrics[metric_name].append({"step": step, "value": value})

    def log_dict(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name to value
            step: Optional step number
        """
        for name, value in metrics.items():
            self.log(name, value, step)

    def get_metric(self, metric_name: str) -> List[Dict[str, Any]]:
        """Get all values for a metric."""
        return self.metrics.get(metric_name, [])

    def get_last_value(self, metric_name: str) -> Optional[Any]:
        """Get last logged value for a metric."""
        values = self.metrics.get(metric_name, [])
        return values[-1]["value"] if values else None

    def get_values(self, metric_name: str) -> List[Any]:
        """Get all values for a metric (without step info)."""
        return [entry["value"] for entry in self.metrics.get(metric_name, [])]

    def get_steps(self, metric_name: str) -> List[int]:
        """Get all steps for a metric."""
        return [entry["step"] for entry in self.metrics.get(metric_name, [])]

    def clear(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()
        self.step_counter = 0

    def save(self, filepath: str) -> None:
        """Save metrics to JSON file."""
        with open(filepath, "w") as f:
            json.dump(dict(self.metrics), f, indent=2)

        logger.info(f"Metrics saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load metrics from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        self.metrics = defaultdict(list, data)

        # Update step counter
        max_step = 0
        for entries in self.metrics.values():
            for entry in entries:
                max_step = max(max_step, entry.get("step", 0))

        self.step_counter = max_step + 1

        logger.info(f"Metrics loaded from {filepath}")

    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics for all metrics."""
        import statistics

        summary = {}

        for metric_name, entries in self.metrics.items():
            values = [e["value"] for e in entries if isinstance(e["value"], (int, float))]

            if values:
                summary[metric_name] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "last": values[-1],
                }

        return summary

    def print_summary(self) -> None:
        """Print summary of all metrics."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("METRIC SUMMARY")
        print("=" * 60)

        for metric_name, stats in sorted(summary.items()):
            print(f"\n{metric_name}:")
            print(f"  Count:  {stats['count']}")
            print(f"  Mean:   {stats['mean']:.4f}")
            print(f"  Median: {stats['median']:.4f}")
            print(f"  Std:    {stats['std']:.4f}")
            print(f"  Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Last:   {stats['last']:.4f}")

        print("=" * 60)


class CSVLogger(MetricLogger):
    """Log metrics to CSV file."""

    def __init__(self, filepath: str):
        """
        Initialize CSV logger.

        Args:
            filepath: Path to CSV file
        """
        super().__init__()
        self.filepath = filepath
        self.file = None
        self.writer = None
        self.headers_written = False

    def __enter__(self):
        """Enter context."""
        self.file = open(self.filepath, "w", newline="")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if self.file:
            self.file.close()
        return False

    def log_dict(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to CSV."""
        super().log_dict(metrics, step)

        if not self.file:
            return

        # Write headers if needed
        if not self.headers_written:
            import csv

            self.writer = csv.DictWriter(self.file, fieldnames=["step"] + list(metrics.keys()))
            self.writer.writeheader()
            self.headers_written = True

        # Write row
        if self.writer:
            row = {"step": step if step is not None else self.step_counter - 1}
            row.update(metrics)
            self.writer.writerow(row)
            self.file.flush()


class TensorBoardLogger(MetricLogger):
    """Log metrics to TensorBoard."""

    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: TensorBoard log directory
        """
        super().__init__()
        self.log_dir = log_dir
        self.writer = None

        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard logging to {log_dir}")
        except ImportError:
            logger.warning("tensorboard not available")

    def log(self, metric_name: str, value: Any, step: Optional[int] = None) -> None:
        """Log metric to TensorBoard."""
        super().log(metric_name, value, step)

        if self.writer and isinstance(value, (int, float)):
            step = step if step is not None else self.step_counter - 1
            self.writer.add_scalar(metric_name, value, step)

    def log_histogram(self, name: str, values: List[float], step: int) -> None:
        """Log histogram to TensorBoard."""
        if self.writer:
            import numpy as np

            self.writer.add_histogram(name, np.array(values), step)

    def close(self) -> None:
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()


class WandbLogger(MetricLogger):
    """Log metrics to Weights & Biases."""

    def __init__(self, project: str, name: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize W&B logger.

        Args:
            project: W&B project name
            name: Run name
            config: Configuration dict
        """
        super().__init__()
        self.project = project
        self.run = None

        try:
            import wandb

            self.run = wandb.init(project=project, name=name, config=config)
            logger.info(f"W&B logging to project {project}")
        except ImportError:
            logger.warning("wandb not available")

    def log_dict(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B."""
        super().log_dict(metrics, step)

        if self.run:
            import wandb

            wandb.log(metrics, step=step)

    def finish(self) -> None:
        """Finish W&B run."""
        if self.run:
            import wandb

            wandb.finish()


class MultiLogger(MetricLogger):
    """Log to multiple backends simultaneously."""

    def __init__(self, loggers: List[MetricLogger]):
        """
        Initialize multi-logger.

        Args:
            loggers: List of logger instances
        """
        super().__init__()
        self.loggers = loggers

    def log(self, metric_name: str, value: Any, step: Optional[int] = None) -> None:
        """Log to all loggers."""
        super().log(metric_name, value, step)

        for logger_instance in self.loggers:
            logger_instance.log(metric_name, value, step)

    def log_dict(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log dict to all loggers."""
        super().log_dict(metrics, step)

        for logger_instance in self.loggers:
            logger_instance.log_dict(metrics, step)
