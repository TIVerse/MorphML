"""Tests for CMA-ES optimizer.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import numpy as np
import pytest

from morphml.core.dsl import create_cnn_space
from morphml.optimizers.evolutionary import CMAESOptimizer


class TestCMAESOptimizer:
    """Test suite for CMA-ES optimizer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.search_space = create_cnn_space(num_classes=10)
        self.config = {"population_size": 10, "num_generations": 5, "sigma0": 0.3}

    def test_initialization(self):
        """Test CMA-ES initialization."""
        optimizer = CMAESOptimizer(self.search_space, self.config)

        assert optimizer.population_size == 10
        assert optimizer.num_generations == 5
        assert optimizer.sigma == 0.3

    def test_optimization_run(self):
        """Test basic CMA-ES optimization."""
        optimizer = CMAESOptimizer(self.search_space, {"population_size": 8, "num_generations": 3})

        def evaluator(graph):
            return np.random.rand()

        best = optimizer.optimize(evaluator)

        assert best is not None
        assert hasattr(best, "fitness")
        assert best.fitness > 0

    def test_covariance_adaptation(self):
        """Test that covariance matrix is adapted."""
        optimizer = CMAESOptimizer(self.search_space, {"population_size": 8, "num_generations": 3})

        def evaluator(graph):
            return np.random.rand()

        optimizer.optimize(evaluator)

        # Check that covariance matrix has been updated
        assert optimizer.C is not None
        assert optimizer.C.shape[0] > 0

    def test_get_history(self):
        """Test history tracking."""
        optimizer = CMAESOptimizer(self.search_space, {"population_size": 6, "num_generations": 3})

        def evaluator(graph):
            return np.random.rand()

        optimizer.optimize(evaluator)
        history = optimizer.get_history()

        assert len(history) > 0
        assert "generation" in history[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
