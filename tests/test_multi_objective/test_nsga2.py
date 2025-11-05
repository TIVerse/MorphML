"""Tests for NSGA-II optimizer.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import numpy as np
import pytest

from morphml.core.dsl import create_cnn_space
from morphml.optimizers.multi_objective import (
    MultiObjectiveIndividual,
    NSGA2Optimizer,
)


class TestMultiObjectiveIndividual:
    """Test MultiObjectiveIndividual class."""

    def test_dominance_strict(self):
        """Test strict dominance."""
        objectives = [{"name": "f1", "maximize": True}, {"name": "f2", "maximize": True}]

        # ind1 dominates ind2 (better in both)
        ind1 = MultiObjectiveIndividual(genome=None, objectives={"f1": 0.9, "f2": 0.8})
        ind2 = MultiObjectiveIndividual(genome=None, objectives={"f1": 0.5, "f2": 0.6})

        assert ind1.dominates(ind2, objectives)
        assert not ind2.dominates(ind1, objectives)

    def test_dominance_partial(self):
        """Test partial dominance."""
        objectives = [{"name": "f1", "maximize": True}, {"name": "f2", "maximize": True}]

        # ind1 better in f1, ind2 better in f2 -> no dominance
        ind1 = MultiObjectiveIndividual(genome=None, objectives={"f1": 0.9, "f2": 0.5})
        ind2 = MultiObjectiveIndividual(genome=None, objectives={"f1": 0.5, "f2": 0.9})

        assert not ind1.dominates(ind2, objectives)
        assert not ind2.dominates(ind1, objectives)

    def test_dominance_mixed_objectives(self):
        """Test dominance with mixed maximize/minimize."""
        objectives = [
            {"name": "accuracy", "maximize": True},
            {"name": "latency", "maximize": False},  # Minimize
        ]

        # ind1: higher accuracy, lower latency -> dominates
        ind1 = MultiObjectiveIndividual(genome=None, objectives={"accuracy": 0.9, "latency": 10.0})
        ind2 = MultiObjectiveIndividual(genome=None, objectives={"accuracy": 0.8, "latency": 15.0})

        assert ind1.dominates(ind2, objectives)


class TestNSGA2Optimizer:
    """Test NSGA-II optimizer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.search_space = create_cnn_space(num_classes=10)
        self.config = {
            "population_size": 20,
            "num_generations": 5,
            "objectives": [{"name": "f1", "maximize": True}, {"name": "f2", "maximize": True}],
        }

    def test_initialization(self):
        """Test NSGA-II initialization."""
        optimizer = NSGA2Optimizer(self.search_space, self.config)

        assert optimizer.pop_size == 20
        assert optimizer.num_generations == 5
        assert len(optimizer.objectives) == 2
        assert len(optimizer.population) == 0

    def test_population_initialization(self):
        """Test population initialization."""
        optimizer = NSGA2Optimizer(self.search_space, self.config)
        optimizer.initialize_population()

        assert len(optimizer.population) == 20
        for ind in optimizer.population:
            assert isinstance(ind, MultiObjectiveIndividual)
            assert ind.genome is not None

    def test_fast_non_dominated_sort(self):
        """Test fast non-dominated sorting."""
        optimizer = NSGA2Optimizer(self.search_space, self.config)

        # Create test population with known dominance
        population = [
            MultiObjectiveIndividual(None, {"f1": 0.9, "f2": 0.9}),  # Front 0
            MultiObjectiveIndividual(None, {"f1": 0.8, "f2": 0.8}),  # Front 1
            MultiObjectiveIndividual(None, {"f1": 0.7, "f2": 0.7}),  # Front 2
            MultiObjectiveIndividual(None, {"f1": 0.9, "f2": 0.5}),  # Front 0
            MultiObjectiveIndividual(None, {"f1": 0.5, "f2": 0.9}),  # Front 0
        ]

        fronts = optimizer.fast_non_dominated_sort(population)

        assert len(fronts) > 0
        # First front should have non-dominated solutions
        assert len(fronts[0]) > 0
        # All in front 0 should have rank 0
        for ind in fronts[0]:
            assert ind.rank == 0

    def test_crowding_distance(self):
        """Test crowding distance calculation."""
        optimizer = NSGA2Optimizer(self.search_space, self.config)

        # Create test front
        front = [
            MultiObjectiveIndividual(None, {"f1": 0.1, "f2": 0.9}),
            MultiObjectiveIndividual(None, {"f1": 0.5, "f2": 0.5}),
            MultiObjectiveIndividual(None, {"f1": 0.9, "f2": 0.1}),
        ]

        optimizer.calculate_crowding_distance(front)

        # Boundary points should have infinite distance
        assert front[0].crowding_distance == float("inf") or front[2].crowding_distance == float(
            "inf"
        )

        # Middle point should have finite distance
        assert front[1].crowding_distance < float("inf")
        assert front[1].crowding_distance > 0

    def test_optimize_simple(self):
        """Test basic NSGA-II optimization."""
        optimizer = NSGA2Optimizer(
            self.search_space,
            {
                "population_size": 10,
                "num_generations": 3,
                "objectives": [{"name": "f1", "maximize": True}, {"name": "f2", "maximize": True}],
            },
        )

        # Simple multi-objective evaluator
        def evaluator(graph):
            return {"f1": np.random.rand(), "f2": np.random.rand()}

        pareto_front = optimizer.optimize(evaluator)

        assert len(pareto_front) > 0
        assert all(ind.rank == 0 for ind in pareto_front)

        # Test non-domination within front
        for i, ind1 in enumerate(pareto_front):
            for j, ind2 in enumerate(pareto_front):
                if i != j:
                    # No solution should dominate another in the front
                    assert not ind1.dominates(ind2, optimizer.objectives)


class TestQualityIndicators:
    """Test quality indicators."""

    def test_hypervolume_2d(self):
        """Test 2D hypervolume calculation."""
        from morphml.optimizers.multi_objective import QualityIndicators

        pareto_front = [
            MultiObjectiveIndividual(None, {"f1": 0.8, "f2": 0.3}),
            MultiObjectiveIndividual(None, {"f1": 0.5, "f2": 0.6}),
            MultiObjectiveIndividual(None, {"f1": 0.3, "f2": 0.8}),
        ]

        qi = QualityIndicators()
        hv = qi.hypervolume(pareto_front, reference_point=np.array([0, 0]))

        assert hv > 0
        assert np.isfinite(hv)

    def test_spacing(self):
        """Test spacing metric."""
        from morphml.optimizers.multi_objective import QualityIndicators

        # Uniformly spaced front
        pareto_front = [
            MultiObjectiveIndividual(None, {"f1": 0.2, "f2": 0.8}),
            MultiObjectiveIndividual(None, {"f1": 0.5, "f2": 0.5}),
            MultiObjectiveIndividual(None, {"f1": 0.8, "f2": 0.2}),
        ]

        qi = QualityIndicators()
        spacing = qi.spacing(pareto_front)

        assert spacing >= 0
        assert np.isfinite(spacing)


def test_convenience_function():
    """Test optimize_with_nsga2 convenience function."""
    from morphml.optimizers.multi_objective import optimize_with_nsga2

    space = create_cnn_space(num_classes=10)

    def evaluator(graph):
        return {"f1": np.random.rand(), "f2": np.random.rand()}

    pareto_front = optimize_with_nsga2(
        space,
        evaluator,
        objectives=[{"name": "f1", "maximize": True}, {"name": "f2", "maximize": True}],
        population_size=10,
        num_generations=3,
        verbose=False,
    )

    assert len(pareto_front) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
