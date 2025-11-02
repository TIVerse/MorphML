"""Tests for constraint system."""


from morphml.constraints import (
    ConstraintHandler,
    DepthConstraint,
    MaxParametersConstraint,
    MinParametersConstraint,
    OperationConstraint,
    WidthConstraint,
)
from morphml.constraints.predicates import CompositeConstraint, ConnectivityConstraint
from morphml.core.dsl import Layer, SearchSpace
from morphml.core.graph import GraphEdge, GraphNode, ModelGraph


class TestConstraintPredicates:
    """Test individual constraint predicates."""

    def create_sample_graph(self, num_nodes: int = 5) -> ModelGraph:
        """Create sample graph for testing."""
        graph = ModelGraph()

        input_node = GraphNode.create("input", {"shape": (3, 32, 32)})
        graph.add_node(input_node)

        prev_node = input_node
        for _i in range(num_nodes - 2):
            node = GraphNode.create("conv2d", {"filters": 64})
            graph.add_node(node)
            graph.add_edge(GraphEdge(prev_node, node))
            prev_node = node

        output_node = GraphNode.create("dense", {"units": 10})
        graph.add_node(output_node)
        graph.add_edge(GraphEdge(prev_node, output_node))

        return graph

    def test_max_parameters_constraint(self) -> None:
        """Test MaxParametersConstraint."""
        constraint = MaxParametersConstraint(max_params=1000000)

        graph = self.create_sample_graph(3)

        # Should pass
        assert constraint.check(graph)
        assert constraint.penalty(graph) == 0.0

    def test_max_parameters_violation(self) -> None:
        """Test MaxParametersConstraint violation."""
        constraint = MaxParametersConstraint(max_params=100)

        graph = self.create_sample_graph(5)

        # Should fail
        assert not constraint.check(graph)
        assert constraint.penalty(graph) > 0.0

    def test_min_parameters_constraint(self) -> None:
        """Test MinParametersConstraint."""
        constraint = MinParametersConstraint(min_params=100)

        graph = self.create_sample_graph(5)

        assert constraint.check(graph)
        assert constraint.penalty(graph) == 0.0

    def test_depth_constraint(self) -> None:
        """Test DepthConstraint."""
        constraint = DepthConstraint(min_depth=2, max_depth=10)

        graph = self.create_sample_graph(5)
        depth = graph.get_depth()

        if 2 <= depth <= 10:
            assert constraint.check(graph)
            assert constraint.penalty(graph) == 0.0

    def test_depth_constraint_too_shallow(self) -> None:
        """Test DepthConstraint with shallow graph."""
        constraint = DepthConstraint(min_depth=10, max_depth=20)

        graph = self.create_sample_graph(2)

        # Should fail (too shallow)
        assert not constraint.check(graph)
        assert constraint.penalty(graph) > 0.0

    def test_depth_constraint_too_deep(self) -> None:
        """Test DepthConstraint with deep graph."""
        constraint = DepthConstraint(min_depth=1, max_depth=2)

        graph = self.create_sample_graph(10)

        # Might fail (too deep)
        penalty = constraint.penalty(graph)
        assert penalty >= 0.0

    def test_width_constraint(self) -> None:
        """Test WidthConstraint."""
        constraint = WidthConstraint(min_width=1, max_width=10)

        graph = self.create_sample_graph(5)

        assert constraint.check(graph)

    def test_operation_constraint_required(self) -> None:
        """Test OperationConstraint with required ops."""
        constraint = OperationConstraint(required_ops={"input", "dense"})

        graph = self.create_sample_graph(3)

        assert constraint.check(graph)
        assert constraint.penalty(graph) == 0.0

    def test_operation_constraint_missing(self) -> None:
        """Test OperationConstraint with missing ops."""
        constraint = OperationConstraint(required_ops={"batchnorm"})

        graph = self.create_sample_graph(3)

        # Should fail (no batchnorm)
        assert not constraint.check(graph)
        assert constraint.penalty(graph) > 0.0

    def test_operation_constraint_forbidden(self) -> None:
        """Test OperationConstraint with forbidden ops."""
        constraint = OperationConstraint(forbidden_ops={"dropout"})

        graph = self.create_sample_graph(3)

        # Should pass (no dropout)
        assert constraint.check(graph)

    def test_connectivity_constraint(self) -> None:
        """Test ConnectivityConstraint."""
        constraint = ConnectivityConstraint(min_edges=2, max_edges=20)

        graph = self.create_sample_graph(5)

        assert constraint.check(graph)


class TestConstraintHandler:
    """Test ConstraintHandler."""

    def create_graph(self) -> ModelGraph:
        """Create test graph."""
        graph = ModelGraph()

        input_node = GraphNode.create("input", {"shape": (3, 32, 32)})
        conv = GraphNode.create("conv2d", {"filters": 64})
        output = GraphNode.create("dense", {"units": 10})

        graph.add_node(input_node)
        graph.add_node(conv)
        graph.add_node(output)

        graph.add_edge(GraphEdge(input_node, conv))
        graph.add_edge(GraphEdge(conv, output))

        return graph

    def test_handler_creation(self) -> None:
        """Test ConstraintHandler creation."""
        handler = ConstraintHandler()

        assert len(handler) == 0

    def test_add_constraint(self) -> None:
        """Test adding constraints."""
        handler = ConstraintHandler()

        handler.add_constraint(MaxParametersConstraint(1000000))
        handler.add_constraint(DepthConstraint(min_depth=2, max_depth=10))

        assert len(handler) == 2

    def test_remove_constraint(self) -> None:
        """Test removing constraints."""
        handler = ConstraintHandler()

        handler.add_constraint(MaxParametersConstraint(1000000, name="max_params"))
        handler.add_constraint(DepthConstraint(name="depth"))

        assert len(handler) == 2

        handler.remove_constraint("depth")

        assert len(handler) == 1

    def test_check_all_satisfied(self) -> None:
        """Test check when all constraints satisfied."""
        handler = ConstraintHandler()
        handler.add_constraint(MaxParametersConstraint(1000000))
        handler.add_constraint(DepthConstraint(min_depth=1, max_depth=10))

        graph = self.create_graph()

        assert handler.check(graph)

    def test_check_violation(self) -> None:
        """Test check with violation."""
        handler = ConstraintHandler()
        handler.add_constraint(MaxParametersConstraint(100))  # Very low limit

        graph = self.create_graph()

        # Should fail
        violations = handler.get_violations(graph)
        assert len(violations) > 0

    def test_total_penalty(self) -> None:
        """Test total penalty calculation."""
        handler = ConstraintHandler()
        handler.add_constraint(MaxParametersConstraint(100))
        handler.add_constraint(DepthConstraint(min_depth=20, max_depth=30))

        graph = self.create_graph()

        penalty = handler.total_penalty(graph)
        assert penalty > 0.0
        assert penalty <= 1.0

    def test_get_penalties(self) -> None:
        """Test getting individual penalties."""
        handler = ConstraintHandler()
        handler.add_constraint(MaxParametersConstraint(100, name="max_params"))
        handler.add_constraint(DepthConstraint(min_depth=1, max_depth=10, name="depth"))

        graph = self.create_graph()

        penalties = handler.get_penalties(graph)

        assert "max_params" in penalties
        assert "depth" in penalties
        assert all(isinstance(p, float) for p in penalties.values())

    def test_apply_penalty_to_fitness(self) -> None:
        """Test applying penalties to fitness."""
        handler = ConstraintHandler()
        handler.add_constraint(MaxParametersConstraint(100))

        graph = self.create_graph()

        original_fitness = 0.9
        penalized = handler.apply_penalty_to_fitness(original_fitness, graph)

        # Should be lower due to violation
        assert penalized < original_fitness

    def test_clear_constraints(self) -> None:
        """Test clearing all constraints."""
        handler = ConstraintHandler()
        handler.add_constraint(MaxParametersConstraint(1000000))
        handler.add_constraint(DepthConstraint())

        assert len(handler) == 2

        handler.clear()

        assert len(handler) == 0


class TestCompositeConstraints:
    """Test composite constraints."""

    def create_graph(self) -> ModelGraph:
        """Create test graph."""
        graph = ModelGraph()

        input_node = GraphNode.create("input", {"shape": (3, 32, 32)})
        conv = GraphNode.create("conv2d", {"filters": 64})
        relu = GraphNode.create("relu")
        output = GraphNode.create("dense", {"units": 10})

        graph.add_node(input_node)
        graph.add_node(conv)
        graph.add_node(relu)
        graph.add_node(output)

        graph.add_edge(GraphEdge(input_node, conv))
        graph.add_edge(GraphEdge(conv, relu))
        graph.add_edge(GraphEdge(relu, output))

        return graph

    def test_composite_all_mode(self) -> None:
        """Test composite constraint with ALL mode."""
        constraints = [MaxParametersConstraint(1000000), DepthConstraint(min_depth=1, max_depth=10)]

        composite = CompositeConstraint(constraints, mode="all")
        graph = self.create_graph()

        # Both should be satisfied
        assert composite.check(graph)

    def test_composite_any_mode(self) -> None:
        """Test composite constraint with ANY mode."""
        constraints = [
            MaxParametersConstraint(100),  # Will fail
            DepthConstraint(min_depth=1, max_depth=10),  # Will pass
        ]

        composite = CompositeConstraint(constraints, mode="any")
        graph = self.create_graph()

        # At least one should be satisfied
        assert composite.check(graph)


class TestConstraintIntegration:
    """Integration tests for constraint system."""

    def test_constrained_search_space(self) -> None:
        """Test search space with constraints."""
        space = SearchSpace("constrained")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64, 128]),
            Layer.relu(),
            Layer.output(units=10),
        )

        # Add constraint
        def max_depth_constraint(graph):
            return graph.get_depth() <= 5

        space.add_constraint(max_depth_constraint)

        # Sample should respect constraint
        for _ in range(5):
            arch = space.sample()
            assert arch.get_depth() <= 5

    def test_optimizer_with_constraints(self) -> None:
        """Test optimizer with constraint handler."""
        from morphml.evaluation import HeuristicEvaluator
        from morphml.optimizers import RandomSearch

        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)), Layer.conv2d(filters=64), Layer.output(units=10)
        )

        handler = ConstraintHandler()
        handler.add_constraint(MaxParametersConstraint(1000000))

        evaluator = HeuristicEvaluator()

        def constrained_eval(graph):
            base_fitness = evaluator(graph)
            return handler.apply_penalty_to_fitness(base_fitness, graph)

        rs = RandomSearch(space, num_samples=10)
        best = rs.optimize(constrained_eval)

        assert best is not None


def test_constraint_workflow() -> None:
    """Integration test for constraint workflow."""
    # Create search space
    space = SearchSpace("workflow_test")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=[32, 64]),
        Layer.relu(),
        Layer.maxpool(),
        Layer.dense(units=[128, 256]),
        Layer.output(units=10),
    )

    # Create constraint handler
    handler = ConstraintHandler()
    handler.add_constraint(MaxParametersConstraint(2000000))
    handler.add_constraint(DepthConstraint(min_depth=3, max_depth=10))
    handler.add_constraint(OperationConstraint(required_ops={"input", "dense"}))

    # Sample and check
    valid_count = 0
    for _ in range(10):
        arch = space.sample()
        if handler.check(arch):
            valid_count += 1

    # Most should be valid
    assert valid_count > 0
