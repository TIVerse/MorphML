"""Tests for benchmarking metrics.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import pytest

from morphml.benchmarks.metrics import (
    area_under_curve,
    compare_optimizers,
    compute_all_metrics,
    convergence_rate,
    final_best_fitness,
    sample_efficiency,
    stability_score,
    success_rate,
    time_efficiency,
)


class TestMetricFunctions:
    """Test metric calculation functions."""

    def test_sample_efficiency(self):
        """Test sample efficiency calculation."""
        history = [
            {"best_fitness": 0.5},
            {"best_fitness": 0.7},
            {"best_fitness": 0.85},
            {"best_fitness": 0.9},
        ]

        # Should reach 0.8 at iteration 3
        efficiency = sample_efficiency(history, target_fitness=0.8)
        assert efficiency == 3

        # Should not reach 0.95
        efficiency = sample_efficiency(history, target_fitness=0.95)
        assert efficiency == -1

    def test_convergence_rate(self):
        """Test convergence rate calculation."""
        history = [
            {"best_fitness": 0.5},
            {"best_fitness": 0.6},
            {"best_fitness": 0.7},
            {"best_fitness": 0.8},
        ]

        rate = convergence_rate(history)
        assert rate > 0
        assert rate == 0.1  # Constant improvement of 0.1

    def test_final_best_fitness(self):
        """Test final best fitness extraction."""
        history = [
            {"best_fitness": 0.5},
            {"best_fitness": 0.7},
            {"best_fitness": 0.9},
            {"best_fitness": 0.85},  # Can decrease
        ]

        best = final_best_fitness(history)
        assert best == 0.9

    def test_time_efficiency(self):
        """Test time efficiency calculation."""
        history = [
            {"best_fitness": 0.5, "time_elapsed": 1.0},
            {"best_fitness": 0.7, "time_elapsed": 2.5},
            {"best_fitness": 0.85, "time_elapsed": 4.0},
        ]

        time_eff = time_efficiency(history, target_fitness=0.8)
        assert time_eff == 4.0

        # Should not reach 0.9
        time_eff = time_efficiency(history, target_fitness=0.9)
        assert time_eff == -1.0

    def test_stability_score(self):
        """Test stability score across multiple runs."""
        # High stability (low variance)
        histories1 = [
            [{"best_fitness": 0.90}],
            [{"best_fitness": 0.91}],
            [{"best_fitness": 0.89}],
        ]

        stability1 = stability_score(histories1)
        assert 0.9 < stability1 <= 1.0

        # Low stability (high variance)
        histories2 = [
            [{"best_fitness": 0.5}],
            [{"best_fitness": 0.9}],
            [{"best_fitness": 0.3}],
        ]

        stability2 = stability_score(histories2)
        assert stability2 < stability1

    def test_area_under_curve(self):
        """Test AUC calculation."""
        history = [
            {"best_fitness": 0.5},
            {"best_fitness": 0.6},
            {"best_fitness": 0.7},
            {"best_fitness": 0.8},
        ]

        auc = area_under_curve(history)
        assert auc > 0
        # AUC should be approximately (0.5+0.6+0.7+0.8)*1 = 2.6 (trapezoid rule)
        assert 2.0 < auc < 3.0

    def test_success_rate(self):
        """Test success rate calculation."""
        histories = [
            [{"best_fitness": 0.85}],  # Success
            [{"best_fitness": 0.90}],  # Success
            [{"best_fitness": 0.75}],  # Failure
            [{"best_fitness": 0.95}],  # Success
        ]

        rate = success_rate(histories, target_fitness=0.8)
        assert rate == 0.75  # 3 out of 4 succeeded

    def test_compute_all_metrics(self):
        """Test computing all metrics at once."""
        history = [
            {"best_fitness": 0.5},
            {"best_fitness": 0.7},
            {"best_fitness": 0.85},
            {"best_fitness": 0.9},
        ]

        metrics = compute_all_metrics(history, target_fitness=0.8)

        assert "final_best_fitness" in metrics
        assert "convergence_rate" in metrics
        assert "auc" in metrics
        assert "num_evaluations" in metrics
        assert "sample_efficiency" in metrics
        assert "time_efficiency" in metrics

        assert metrics["final_best_fitness"] == 0.9
        assert metrics["num_evaluations"] == 4
        assert metrics["sample_efficiency"] == 3

    def test_compare_optimizers(self):
        """Test optimizer comparison."""
        results = {
            "Optimizer1": [
                {"best_fitness": 0.8},
                {"best_fitness": 0.85},
                {"best_fitness": 0.9},
            ],
            "Optimizer2": [
                {"best_fitness": 0.7},
                {"best_fitness": 0.75},
                {"best_fitness": 0.8},
            ],
        }

        comparison = compare_optimizers(results, target_fitness=0.75)

        assert "Optimizer1" in comparison
        assert "Optimizer2" in comparison

        assert comparison["Optimizer1"]["final_best_fitness"] == 0.9
        assert comparison["Optimizer2"]["final_best_fitness"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
