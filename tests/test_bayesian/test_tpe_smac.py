"""Tests for TPE and SMAC optimizers.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import numpy as np
import pytest

from morphml.core.dsl import create_cnn_space
from morphml.optimizers.bayesian import TPEOptimizer, SMACOptimizer


class TestTPEOptimizer:
    """Test suite for TPE optimizer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.search_space = create_cnn_space(num_classes=10)
        self.config = {
            'n_initial_points': 3,
            'max_iterations': 10,
            'gamma': 0.25
        }

    def test_initialization(self):
        """Test TPE optimizer initialization."""
        optimizer = TPEOptimizer(
            self.search_space,
            self.config
        )

        assert optimizer.n_initial_points == 3
        assert optimizer.max_iterations == 10
        assert optimizer.gamma == 0.25
        assert len(optimizer.X_sample) == 0
        assert len(optimizer.y_sample) == 0

    def test_initial_sampling(self):
        """Test initial random sampling."""
        optimizer = TPEOptimizer(
            self.search_space,
            self.config
        )

        # Dummy evaluator
        def evaluator(graph):
            return np.random.rand()

        # Should create initial samples
        optimizer._initial_sample(evaluator)

        assert len(optimizer.X_sample) == 3
        assert len(optimizer.y_sample) == 3

    def test_kde_fitting(self):
        """Test KDE fitting for TPE."""
        optimizer = TPEOptimizer(
            self.search_space,
            self.config
        )

        # Create dummy data
        optimizer.X_sample = [np.random.rand(60) for _ in range(10)]
        optimizer.y_sample = np.random.rand(10)

        # Should fit without error
        try:
            optimizer._fit_models()
            # If we have sklearn, this should work
        except ImportError:
            pytest.skip("sklearn not available for KDE")

    def test_optimize_simple(self):
        """Test basic TPE optimization run."""
        optimizer = TPEOptimizer(
            self.search_space,
            {'n_initial_points': 3, 'max_iterations': 5}
        )

        # Simple evaluator
        def evaluator(graph):
            return np.random.rand()

        best = optimizer.optimize(evaluator)

        assert best is not None
        assert hasattr(best, 'fitness')
        assert best.fitness >= 0

    def test_get_history(self):
        """Test history tracking."""
        optimizer = TPEOptimizer(
            self.search_space,
            {'n_initial_points': 2, 'max_iterations': 3}
        )

        def evaluator(graph):
            return np.random.rand()

        optimizer.optimize(evaluator)
        history = optimizer.get_history()

        assert len(history) > 0
        assert 'iteration' in history[0]
        assert 'best_fitness' in history[0]


class TestSMACOptimizer:
    """Test suite for SMAC optimizer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.search_space = create_cnn_space(num_classes=10)
        self.config = {
            'n_initial_points': 3,
            'max_iterations': 10,
            'n_estimators': 10
        }

    def test_initialization(self):
        """Test SMAC optimizer initialization."""
        optimizer = SMACOptimizer(
            self.search_space,
            self.config
        )

        assert optimizer.n_initial_points == 3
        assert optimizer.max_iterations == 10
        assert optimizer.n_estimators == 10
        assert len(optimizer.X_sample) == 0
        assert len(optimizer.y_sample) == 0

    def test_initial_sampling(self):
        """Test initial random sampling."""
        optimizer = SMACOptimizer(
            self.search_space,
            self.config
        )

        # Dummy evaluator
        def evaluator(graph):
            return np.random.rand()

        # Should create initial samples
        optimizer._initial_sample(evaluator)

        assert len(optimizer.X_sample) == 3
        assert len(optimizer.y_sample) == 3

    def test_random_forest_fitting(self):
        """Test Random Forest surrogate fitting."""
        optimizer = SMACOptimizer(
            self.search_space,
            self.config
        )

        # Create dummy data
        optimizer.X_sample = [np.random.rand(60) for _ in range(10)]
        optimizer.y_sample = np.random.rand(10)

        # Should fit without error
        optimizer._fit_surrogate()

        assert optimizer.rf is not None

    def test_acquisition_optimization(self):
        """Test acquisition function optimization."""
        optimizer = SMACOptimizer(
            self.search_space,
            self.config
        )

        # Setup dummy RF
        optimizer.X_sample = [np.random.rand(60) for _ in range(10)]
        optimizer.y_sample = np.random.rand(10)
        optimizer._fit_surrogate()

        # Should find next point
        next_x = optimizer._optimize_acquisition()

        assert next_x is not None
        assert len(next_x) == 60

    def test_optimize_simple(self):
        """Test basic SMAC optimization run."""
        optimizer = SMACOptimizer(
            self.search_space,
            {'n_initial_points': 3, 'max_iterations': 5, 'n_estimators': 5}
        )

        # Simple evaluator
        def evaluator(graph):
            return np.random.rand()

        best = optimizer.optimize(evaluator)

        assert best is not None
        assert hasattr(best, 'fitness')
        assert best.fitness >= 0

    def test_get_history(self):
        """Test history tracking."""
        optimizer = SMACOptimizer(
            self.search_space,
            {'n_initial_points': 2, 'max_iterations': 3, 'n_estimators': 5}
        )

        def evaluator(graph):
            return np.random.rand()

        optimizer.optimize(evaluator)
        history = optimizer.get_history()

        assert len(history) > 0
        assert 'iteration' in history[0]
        assert 'best_fitness' in history[0]


def test_tpe_convenience_function():
    """Test optimize_with_tpe convenience function."""
    from morphml.optimizers.bayesian import optimize_with_tpe

    space = create_cnn_space(num_classes=10)

    def evaluator(graph):
        return np.random.rand()

    best = optimize_with_tpe(
        space,
        evaluator,
        n_iterations=5,
        verbose=False
    )

    assert best is not None
    assert hasattr(best, 'fitness')


def test_smac_convenience_function():
    """Test optimize_with_smac convenience function."""
    from morphml.optimizers.bayesian import optimize_with_smac

    space = create_cnn_space(num_classes=10)

    def evaluator(graph):
        return np.random.rand()

    best = optimize_with_smac(
        space,
        evaluator,
        n_iterations=5,
        verbose=False
    )

    assert best is not None
    assert hasattr(best, 'fitness')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
