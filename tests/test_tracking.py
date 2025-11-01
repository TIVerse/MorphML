"""Tests for experiment tracking system."""

import json
import os
import tempfile

import pytest

from morphml.tracking import ExperimentTracker, MetricLogger, Reporter
from morphml.tracking.experiment import Experiment, RunContext
from morphml.tracking.logger import CSVLogger


class TestExperiment:
    """Test Experiment class."""

    def test_experiment_creation(self) -> None:
        """Test creating experiment."""
        exp = Experiment("test_exp", {"param1": 10})
        
        assert exp.name == "test_exp"
        assert exp.config["param1"] == 10
        assert exp.status == "running"

    def test_log_metric(self) -> None:
        """Test logging metrics."""
        exp = Experiment("test", {})
        
        exp.log_metric("fitness", 0.85, step=1)
        exp.log_metric("fitness", 0.90, step=2)
        
        assert "fitness" in exp.metrics
        assert len(exp.metrics["fitness"]) == 2

    def test_log_artifact(self) -> None:
        """Test logging artifacts."""
        exp = Experiment("test", {})
        
        exp.log_artifact("model", "/path/to/model.pt")
        
        assert "model" in exp.artifacts
        assert exp.artifacts["model"] == "/path/to/model.pt"

    def test_set_best_result(self) -> None:
        """Test setting best result."""
        exp = Experiment("test", {})
        
        exp.set_best_result({"fitness": 0.95, "accuracy": 0.92})
        
        assert exp.best_result is not None
        assert exp.best_result["fitness"] == 0.95

    def test_finish_experiment(self) -> None:
        """Test finishing experiment."""
        exp = Experiment("test", {})
        
        exp.finish("completed")
        
        assert exp.status == "completed"
        assert exp.end_time is not None

    def test_get_duration(self) -> None:
        """Test getting duration."""
        exp = Experiment("test", {})
        
        import time
        time.sleep(0.1)
        
        duration = exp.get_duration()
        
        assert duration > 0.0

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        exp = Experiment("test", {"param": 1})
        exp.log_metric("fitness", 0.8)
        exp.finish()
        
        data = exp.to_dict()
        
        assert "id" in data
        assert "name" in data
        assert "config" in data
        assert "metrics" in data
        assert "status" in data


class TestExperimentTracker:
    """Test ExperimentTracker."""

    def test_tracker_creation(self) -> None:
        """Test creating tracker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            
            assert tracker.base_dir == tmpdir
            assert os.path.exists(tmpdir)

    def test_create_experiment(self) -> None:
        """Test creating experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            
            exp = tracker.create_experiment("test", {"param": 10})
            
            assert exp.name == "test"
            assert exp.id in tracker.experiments

    def test_save_experiment(self) -> None:
        """Test saving experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            exp = tracker.create_experiment("test", {})
            
            exp.log_metric("fitness", 0.85)
            exp.finish()
            
            tracker.save_experiment(exp)
            
            # Check file exists
            exp_dir = os.path.join(tmpdir, exp.id)
            metadata_path = os.path.join(exp_dir, "metadata.json")
            
            assert os.path.exists(metadata_path)

    def test_load_experiment(self) -> None:
        """Test loading experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            
            # Create and save
            exp = tracker.create_experiment("test", {"param": 1})
            exp.log_metric("fitness", 0.85)
            exp.finish()
            tracker.save_experiment(exp)
            
            exp_id = exp.id
            
            # Clear and load
            tracker.experiments.clear()
            loaded_exp = tracker.load_experiment(exp_id)
            
            assert loaded_exp is not None
            assert loaded_exp.name == "test"
            assert "fitness" in loaded_exp.metrics

    def test_list_experiments(self) -> None:
        """Test listing experiments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            
            exp1 = tracker.create_experiment("test1", {})
            exp2 = tracker.create_experiment("test2", {})
            
            tracker.save_experiment(exp1)
            tracker.save_experiment(exp2)
            
            experiments = tracker.list_experiments()
            
            assert len(experiments) == 2
            assert exp1.id in experiments
            assert exp2.id in experiments

    def test_compare_experiments(self) -> None:
        """Test comparing experiments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            
            exp1 = tracker.create_experiment("test1", {})
            exp1.log_metric("fitness", 0.85)
            exp1.log_metric("fitness", 0.90)
            
            exp2 = tracker.create_experiment("test2", {})
            exp2.log_metric("fitness", 0.80)
            exp2.log_metric("fitness", 0.88)
            
            comparison = tracker.compare_experiments([exp1.id, exp2.id], "fitness")
            
            assert exp1.id in comparison
            assert exp2.id in comparison
            assert len(comparison[exp1.id]) == 2

    def test_get_best_experiment(self) -> None:
        """Test getting best experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            
            exp1 = tracker.create_experiment("test1", {})
            exp1.set_best_result({"best_fitness": 0.85})
            exp1.finish()
            tracker.save_experiment(exp1)
            
            exp2 = tracker.create_experiment("test2", {})
            exp2.set_best_result({"best_fitness": 0.92})
            exp2.finish()
            tracker.save_experiment(exp2)
            
            best = tracker.get_best_experiment("best_fitness")
            
            assert best is not None
            assert best.id == exp2.id

    def test_export_summary(self) -> None:
        """Test exporting summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            
            exp = tracker.create_experiment("test", {})
            exp.set_best_result({"fitness": 0.9})
            exp.finish()
            tracker.save_experiment(exp)
            
            summary_path = os.path.join(tmpdir, "summary.json")
            tracker.export_summary(summary_path)
            
            assert os.path.exists(summary_path)
            
            with open(summary_path, 'r') as f:
                data = json.load(f)
            
            assert "total_experiments" in data
            assert data["total_experiments"] == 1


class TestRunContext:
    """Test RunContext."""

    def test_run_context_success(self) -> None:
        """Test successful run context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            
            with RunContext(tracker, "test", {"param": 1}) as exp:
                exp.log_metric("fitness", 0.85)
                exp.set_best_result({"fitness": 0.85})
            
            # Should be saved automatically
            experiments = tracker.list_experiments()
            assert len(experiments) == 1

    def test_run_context_failure(self) -> None:
        """Test failed run context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            
            try:
                with RunContext(tracker, "test", {}) as exp:
                    exp.log_metric("fitness", 0.85)
                    raise ValueError("Simulated error")
            except ValueError:
                pass
            
            # Load experiment
            experiments = tracker.list_experiments()
            exp_id = experiments[0]
            exp = tracker.load_experiment(exp_id)
            
            # Should be marked as failed
            assert exp.status == "failed"


class TestMetricLogger:
    """Test MetricLogger."""

    def test_logger_creation(self) -> None:
        """Test creating logger."""
        logger = MetricLogger()
        
        assert logger is not None
        assert len(logger.metrics) == 0

    def test_log_metric(self) -> None:
        """Test logging single metric."""
        logger = MetricLogger()
        
        logger.log("fitness", 0.85, step=1)
        logger.log("fitness", 0.90, step=2)
        
        values = logger.get_values("fitness")
        
        assert len(values) == 2
        assert values[0] == 0.85
        assert values[1] == 0.90

    def test_log_dict(self) -> None:
        """Test logging dict."""
        logger = MetricLogger()
        
        logger.log_dict({"fitness": 0.85, "accuracy": 0.92}, step=1)
        
        assert "fitness" in logger.metrics
        assert "accuracy" in logger.metrics

    def test_get_last_value(self) -> None:
        """Test getting last value."""
        logger = MetricLogger()
        
        logger.log("fitness", 0.85)
        logger.log("fitness", 0.90)
        
        last = logger.get_last_value("fitness")
        
        assert last == 0.90

    def test_save_and_load(self) -> None:
        """Test saving and loading."""
        logger = MetricLogger()
        logger.log("fitness", 0.85)
        logger.log("accuracy", 0.92)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            logger.save(filepath)
            
            # Load into new logger
            new_logger = MetricLogger()
            new_logger.load(filepath)
            
            assert "fitness" in new_logger.metrics
            assert "accuracy" in new_logger.metrics
            assert new_logger.get_last_value("fitness") == 0.85
        
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_get_summary(self) -> None:
        """Test getting summary."""
        logger = MetricLogger()
        
        for i in range(10):
            logger.log("fitness", 0.5 + i * 0.05)
        
        summary = logger.get_summary()
        
        assert "fitness" in summary
        assert "mean" in summary["fitness"]
        assert "std" in summary["fitness"]
        assert summary["fitness"]["count"] == 10

    def test_clear(self) -> None:
        """Test clearing metrics."""
        logger = MetricLogger()
        logger.log("fitness", 0.85)
        
        logger.clear()
        
        assert len(logger.metrics) == 0


class TestCSVLogger:
    """Test CSVLogger."""

    def test_csv_logger(self) -> None:
        """Test CSV logging."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            with CSVLogger(filepath) as logger:
                logger.log_dict({"fitness": 0.85, "accuracy": 0.92}, step=1)
                logger.log_dict({"fitness": 0.90, "accuracy": 0.94}, step=2)
            
            # Check file exists and has content
            assert os.path.exists(filepath)
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Should have header + 2 data rows
            assert len(lines) >= 3
        
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestReporter:
    """Test Reporter."""

    def test_reporter_creation(self) -> None:
        """Test creating reporter."""
        reporter = Reporter()
        
        assert reporter is not None

    def test_generate_markdown_report(self) -> None:
        """Test Markdown report."""
        exp = Experiment("test", {"param": 1})
        exp.log_metric("fitness", 0.85)
        exp.set_best_result({"fitness": 0.90})
        exp.finish()
        
        reporter = Reporter()
        report = reporter.generate_markdown_report(exp)
        
        assert "# Experiment Report" in report
        assert "test" in report
        assert "Configuration" in report

    def test_generate_html_report(self) -> None:
        """Test HTML report."""
        exp = Experiment("test", {"param": 1})
        exp.set_best_result({"fitness": 0.90})
        exp.finish()
        
        reporter = Reporter()
        report = reporter.generate_html_report(exp)
        
        assert "<!DOCTYPE html>" in report
        assert "<html>" in report
        assert "test" in report

    def test_generate_comparison_report(self) -> None:
        """Test comparison report."""
        exp1 = Experiment("test1", {})
        exp1.set_best_result({"best_fitness": 0.85})
        exp1.finish()
        
        exp2 = Experiment("test2", {})
        exp2.set_best_result({"best_fitness": 0.92})
        exp2.finish()
        
        reporter = Reporter()
        report = reporter.generate_comparison_report([exp1, exp2], "best_fitness")
        
        assert "# Experiment Comparison" in report
        assert "test1" in report
        assert "test2" in report

    def test_save_report(self) -> None:
        """Test saving report."""
        exp = Experiment("test", {})
        exp.finish()
        
        reporter = Reporter()
        report = reporter.generate_markdown_report(exp)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            filepath = f.name
        
        try:
            reporter.save_report(report, filepath)
            
            assert os.path.exists(filepath)
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            assert "# Experiment Report" in content
        
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_generate_latex_report(self) -> None:
        """Test LaTeX report."""
        exp = Experiment("test", {})
        exp.set_best_result({"fitness": 0.90, "accuracy": 0.92})
        exp.finish()
        
        reporter = Reporter()
        report = reporter.generate_latex_report(exp)
        
        assert "\\documentclass" in report
        assert "\\begin{document}" in report
        assert "\\end{document}" in report


def test_tracking_integration() -> None:
    """Integration test for tracking system."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create tracker
        tracker = ExperimentTracker(tmpdir)
        
        # Run multiple experiments
        for i in range(3):
            with RunContext(tracker, f"experiment_{i}", {"run": i}) as exp:
                # Simulate optimization
                for step in range(10):
                    fitness = 0.5 + step * 0.05 + i * 0.01
                    exp.log_metric("fitness", fitness, step=step)
                    exp.log_metric("diversity", 0.8 - step * 0.02, step=step)
                
                # Set best result
                exp.set_best_result({"fitness": 0.95 + i * 0.01})
        
        # List all experiments
        experiments = tracker.list_experiments()
        assert len(experiments) == 3
        
        # Get best
        best = tracker.get_best_experiment("fitness")
        assert best is not None
        
        # Export summary
        summary_path = os.path.join(tmpdir, "summary.json")
        tracker.export_summary(summary_path)
        assert os.path.exists(summary_path)
        
        # Generate reports
        reporter = Reporter()
        for exp_id in experiments:
            exp = tracker.load_experiment(exp_id)
            report = reporter.generate_markdown_report(exp)
            assert len(report) > 0


def test_metric_logger_workflow() -> None:
    """Test metric logger workflow."""
    logger = MetricLogger()
    
    # Simulate optimization
    for gen in range(20):
        best = 0.5 + gen * 0.02
        mean = best - 0.1
        worst = best - 0.2
        diversity = 0.8 - gen * 0.01
        
        logger.log_dict({
            "best_fitness": best,
            "mean_fitness": mean,
            "worst_fitness": worst,
            "diversity": diversity
        }, step=gen)
    
    # Get summary
    summary = logger.get_summary()
    
    assert "best_fitness" in summary
    assert "diversity" in summary
    assert summary["best_fitness"]["count"] == 20
    
    # Save
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        filepath = f.name
    
    try:
        logger.save(filepath)
        assert os.path.exists(filepath)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
