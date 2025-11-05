"""OpenML integration for benchmarking.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Dict, List, Tuple

import numpy as np

from morphml.logging_config import get_logger

logger = get_logger(__name__)

# Check if OpenML is available
try:
    import openml

    OPENML_AVAILABLE = True
except ImportError:
    OPENML_AVAILABLE = False
    logger.warning("OpenML not available. Install with: pip install openml")


class OpenMLSuite:
    """
    OpenML benchmark suite for NAS evaluation.

    Provides access to curated machine learning datasets from OpenML.

    Example:
        >>> suite = OpenMLSuite()
        >>> task = suite.get_task(3)  # CIFAR-10
        >>> X_train, y_train, X_test, y_test = suite.load_task_data(task)
    """

    # Curated task IDs for benchmarking
    BENCHMARK_TASKS = {
        "mnist": 3573,
        "fashion_mnist": 146825,
        "cifar10": 167124,
        "svhn": 168757,
    }

    def __init__(self):
        """Initialize OpenML suite."""
        if not OPENML_AVAILABLE:
            raise ImportError(
                "OpenML required for benchmark suite. " "Install with: pip install openml"
            )

        self.tasks = {}
        logger.info("Initialized OpenML benchmark suite")

    def get_task(self, task_id: int):
        """
        Get an OpenML task.

        Args:
            task_id: OpenML task ID

        Returns:
            OpenML task object
        """
        if task_id in self.tasks:
            return self.tasks[task_id]

        logger.info(f"Fetching OpenML task {task_id}...")
        task = openml.tasks.get_task(task_id)
        self.tasks[task_id] = task

        return task

    def get_task_by_name(self, name: str):
        """
        Get a benchmark task by name.

        Args:
            name: Task name (e.g., 'mnist', 'cifar10')

        Returns:
            OpenML task object
        """
        task_id = self.BENCHMARK_TASKS.get(name.lower())
        if task_id is None:
            raise ValueError(f"Unknown benchmark task: {name}")

        return self.get_task(task_id)

    def load_task_data(
        self, task, normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data for a task.

        Args:
            task: OpenML task object
            normalize: Whether to normalize features

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        logger.info(f"Loading data for task {task.task_id}...")

        dataset = task.get_dataset()
        X, y, _, _ = dataset.get_data(target=task.target_name)

        X = np.array(X)
        y = np.array(y)

        if normalize and X.dtype in [np.float32, np.float64]:
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # Split train/test (80/20)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"Loaded task data: train={X_train.shape}, test={X_test.shape}")

        return X_train, y_train, X_test, y_test

    def list_benchmark_tasks(self) -> List[str]:
        """List available benchmark tasks."""
        return list(self.BENCHMARK_TASKS.keys())

    def get_task_info(self, task_name: str) -> Dict:
        """
        Get information about a benchmark task.

        Args:
            task_name: Task name

        Returns:
            Dictionary with task metadata
        """
        task_id = self.BENCHMARK_TASKS.get(task_name)
        if task_id is None:
            return {}

        try:
            task = self.get_task(task_id)
            dataset = task.get_dataset()

            return {
                "name": task_name,
                "task_id": task_id,
                "dataset_name": dataset.name,
                "num_instances": dataset.qualities.get("NumberOfInstances"),
                "num_features": dataset.qualities.get("NumberOfFeatures"),
                "num_classes": dataset.qualities.get("NumberOfClasses"),
            }
        except Exception as e:
            logger.error(f"Error fetching task info: {e}")
            return {"name": task_name, "task_id": task_id, "error": str(e)}


def run_openml_benchmark(optimizer, task_name: str, evaluator, num_runs: int = 5) -> List[Dict]:
    """
    Run optimizer on an OpenML benchmark task.

    Args:
        optimizer: Optimizer instance
        task_name: OpenML task name
        evaluator: Evaluator function
        num_runs: Number of independent runs

    Returns:
        List of result dictionaries
    """
    if not OPENML_AVAILABLE:
        raise ImportError("OpenML required")

    OpenMLSuite()

    logger.info(f"Running benchmark: {task_name} with {num_runs} runs")

    results = []

    for run in range(num_runs):
        logger.info(f"Run {run + 1}/{num_runs}")

        # Reset optimizer for each run
        if hasattr(optimizer, "reset"):
            optimizer.reset()

        # Run optimization
        best = optimizer.optimize(evaluator)

        result = {
            "run": run + 1,
            "task": task_name,
            "best_fitness": best.fitness if hasattr(best, "fitness") else 0.0,
            "history": optimizer.get_history() if hasattr(optimizer, "get_history") else [],
        }

        results.append(result)

    logger.info(f"Benchmark complete: {task_name}")

    return results
