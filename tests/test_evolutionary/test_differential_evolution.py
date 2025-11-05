"""Tests for Differential Evolution optimizer.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import numpy as np
import pytest

from morphml.core.dsl import create_cnn_space
from morphml.optimizers.evolutionary import DifferentialEvolutionOptimizer


class TestDifferentialEvolutionOptimizer:
    """Test suite for DE optimizer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.search_space = create_cnn_space(num_classes=10)
        self.config = {
            "population_size": 10,
            "num_generations": 5,
            "mutation_factor": 0.8,
            "crossover_rate": 0.7,
        }

    def test_initialization(self):
        """Test DE initialization."""
        optimizer = DifferentialEvolutionOptimizer(self.search_space, self.config)

        assert optimizer.population_size == 10
        assert optimizer.num_generations == 5
        assert optimizer.F == 0.8
        assert optimizer.CR == 0.7

    def test_mutation_strategies(self):
        """Test different mutation strategies."""
        strategies = ["rand/1", "best/1", "rand/2"]

        for strategy in strategies:
            optimizer = DifferentialEvolutionOptimizer(
                self.search_space,
                {"population_size": 8, "num_generations": 2, "strategy": strategy},
            )

            def evaluator(graph):
                return np.random.rand()

            best = optimizer.optimize(evaluator)
            assert best is not None

    def test_optimization_run(self):
        """Test basic DE optimization."""
        optimizer = DifferentialEvolutionOptimizer(
            self.search_space, {"population_size": 8, "num_generations": 3}
        )

        def evaluator(graph):
            return np.random.rand()

        best = optimizer.optimize(evaluator)

        assert best is not None
        assert hasattr(best, "fitness")
        assert best.fitness > 0

    def test_get_history(self):
        """Test history tracking."""
        optimizer = DifferentialEvolutionOptimizer(
            self.search_space, {"population_size": 6, "num_generations": 3}
        )

        def evaluator(graph):
            return np.random.rand()

        optimizer.optimize(evaluator)
        history = optimizer.get_history()

        assert len(history) > 0
        assert "generation" in history[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
