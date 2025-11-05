"""Tests for Gaussian Process optimizer.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import numpy as np
import pytest

from morphml.core.dsl import create_cnn_space
from morphml.optimizers.bayesian import GaussianProcessOptimizer


class TestGaussianProcessOptimizer:
    """Test suite for GP optimizer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.search_space = create_cnn_space(num_classes=10)
        self.config = {
            "n_initial_points": 5,
            "max_iterations": 20,
            "acquisition": "ei",
            "kernel": "matern",
        }

    def test_initialization(self):
        """Test GP optimizer initialization."""
        optimizer = GaussianProcessOptimizer(self.search_space, self.config)

        assert optimizer.n_initial_points == 5
        assert optimizer.max_iterations == 20
        assert optimizer.acquisition == "ei"
        assert optimizer.X_sample is None
        assert optimizer.y_sample is None

    def test_initial_sampling(self):
        """Test initial random sampling."""
        optimizer = GaussianProcessOptimizer(self.search_space, self.config)

        # Dummy evaluator
        def evaluator(graph):
            return np.random.rand()

        # Should create initial samples
        optimizer._initial_sample(evaluator)

        assert optimizer.X_sample is not None
        assert optimizer.y_sample is not None
        assert len(optimizer.X_sample) == 5
        assert len(optimizer.y_sample) == 5

    def test_fit_surrogate(self):
        """Test GP surrogate fitting."""
        optimizer = GaussianProcessOptimizer(self.search_space, self.config)

        # Create dummy data
        optimizer.X_sample = np.random.rand(10, 60)
        optimizer.y_sample = np.random.rand(10)

        # Should fit without error
        optimizer._fit_surrogate()

        assert optimizer.gp is not None

    def test_acquisition_optimization(self):
        """Test acquisition function optimization."""
        optimizer = GaussianProcessOptimizer(self.search_space, self.config)

        # Setup dummy GP
        optimizer.X_sample = np.random.rand(10, 60)
        optimizer.y_sample = np.random.rand(10)
        optimizer._fit_surrogate()

        # Should find next point
        next_x = optimizer._optimize_acquisition()

        assert next_x is not None
        assert len(next_x) == 60
        assert np.all((next_x >= 0) & (next_x <= 1))

    def test_optimize_simple(self):
        """Test basic optimization run."""
        optimizer = GaussianProcessOptimizer(
            self.search_space, {"n_initial_points": 3, "max_iterations": 5}
        )

        # Simple quadratic evaluator
        def evaluator(graph):
            # Return random score for testing
            return np.random.rand()

        best = optimizer.optimize(evaluator)

        assert best is not None
        assert hasattr(best, "fitness")
        assert best.fitness >= 0

    def test_get_history(self):
        """Test history tracking."""
        optimizer = GaussianProcessOptimizer(
            self.search_space, {"n_initial_points": 2, "max_iterations": 3}
        )

        def evaluator(graph):
            return np.random.rand()

        optimizer.optimize(evaluator)
        history = optimizer.get_history()

        assert len(history) > 0
        assert "iteration" in history[0]
        assert "best_fitness" in history[0]


class TestAcquisitionFunctions:
    """Test acquisition functions."""

    def test_expected_improvement(self):
        """Test EI acquisition function."""
        from morphml.optimizers.bayesian.acquisition import expected_improvement

        # Dummy GP predictions
        mu = np.array([0.5, 0.7, 0.3])
        sigma = np.array([0.1, 0.15, 0.05])
        f_best = 0.6

        ei = expected_improvement(mu, sigma, f_best)

        assert len(ei) == 3
        assert np.all(ei >= 0)
        # Higher mu or sigma should give higher EI
        assert ei[1] > ei[2]  # Higher mu

    def test_upper_confidence_bound(self):
        """Test UCB acquisition function."""
        from morphml.optimizers.bayesian.acquisition import upper_confidence_bound

        mu = np.array([0.5, 0.7, 0.3])
        sigma = np.array([0.1, 0.15, 0.05])

        ucb = upper_confidence_bound(mu, sigma, kappa=2.0)

        assert len(ucb) == 3
        # UCB should be mu + kappa * sigma
        expected_0 = 0.5 + 2.0 * 0.1
        assert np.isclose(ucb[0], expected_0)


def test_convenience_function():
    """Test optimize_with_gp convenience function."""
    from morphml.optimizers.bayesian import optimize_with_gp

    space = create_cnn_space(num_classes=10)

    def evaluator(graph):
        return np.random.rand()

    best = optimize_with_gp(space, evaluator, n_iterations=5, verbose=False)

    assert best is not None
    assert hasattr(best, "fitness")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
