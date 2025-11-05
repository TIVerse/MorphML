"""Benchmark suite for NAS optimizers."""

from morphml.benchmarks.comparator import ConvergenceAnalyzer, OptimizerComparator

# Phase 2: Extended benchmarking
from morphml.benchmarks.datasets import DatasetLoader, get_dataset_info
from morphml.benchmarks.metrics import (
    compare_optimizers,
    compute_all_metrics,
    convergence_rate,
    final_best_fitness,
    sample_efficiency,
)
from morphml.benchmarks.problems import (
    ComplexProblem,
    ConstrainedProblem,
    MultiModalProblem,
    SimpleProblem,
    get_all_problems,
)
from morphml.benchmarks.suite import BenchmarkResult, BenchmarkSuite

# OpenML integration (optional)
try:
    from morphml.benchmarks.openml_suite import OpenMLSuite, run_openml_benchmark

    _OPENML_AVAILABLE = True
except ImportError:
    OpenMLSuite = None
    run_openml_benchmark = None
    _OPENML_AVAILABLE = False

__all__ = [
    # Core
    "BenchmarkSuite",
    "BenchmarkResult",
    "OptimizerComparator",
    "ConvergenceAnalyzer",
    "SimpleProblem",
    "ComplexProblem",
    "MultiModalProblem",
    "ConstrainedProblem",
    "get_all_problems",
    # Phase 2: Datasets
    "DatasetLoader",
    "get_dataset_info",
    # Phase 2: Metrics
    "sample_efficiency",
    "convergence_rate",
    "final_best_fitness",
    "compute_all_metrics",
    "compare_optimizers",
    # Phase 2: OpenML (if available)
    "OpenMLSuite",
    "run_openml_benchmark",
]
