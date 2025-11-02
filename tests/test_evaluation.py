"""Tests for evaluation systems."""


from morphml.core.dsl import Layer, SearchSpace
from morphml.core.graph import GraphEdge, GraphNode, ModelGraph
from morphml.evaluation import HeuristicEvaluator


class TestHeuristicEvaluator:
    """Test HeuristicEvaluator."""

    def create_small_graph(self) -> ModelGraph:
        """Create small test graph."""
        graph = ModelGraph()

        input_node = GraphNode.create("input", {"shape": (3, 32, 32)})
        conv = GraphNode.create("conv2d", {"filters": 32})
        output = GraphNode.create("dense", {"units": 10})

        graph.add_node(input_node)
        graph.add_node(conv)
        graph.add_node(output)

        graph.add_edge(GraphEdge(input_node, conv))
        graph.add_edge(GraphEdge(conv, output))

        return graph

    def create_large_graph(self) -> ModelGraph:
        """Create larger test graph."""
        graph = ModelGraph()

        input_node = GraphNode.create("input", {"shape": (3, 32, 32)})
        graph.add_node(input_node)

        prev = input_node
        for i in range(10):
            node = GraphNode.create("conv2d", {"filters": 64 * (i + 1)})
            graph.add_node(node)
            graph.add_edge(GraphEdge(prev, node))
            prev = node

        output = GraphNode.create("dense", {"units": 1000})
        graph.add_node(output)
        graph.add_edge(GraphEdge(prev, output))

        return graph

    def test_evaluator_creation(self) -> None:
        """Test creating evaluator."""
        evaluator = HeuristicEvaluator()

        assert evaluator.param_weight == 0.3
        assert evaluator.depth_weight == 0.3

    def test_evaluator_custom_weights(self) -> None:
        """Test custom weights."""
        evaluator = HeuristicEvaluator(
            param_weight=0.5,
            depth_weight=0.2,
            width_weight=0.2,
            connectivity_weight=0.1
        )

        assert evaluator.param_weight == 0.5
        assert evaluator.depth_weight == 0.2

    def test_parameter_score(self) -> None:
        """Test parameter scoring."""
        evaluator = HeuristicEvaluator(target_params=1000000)

        small_graph = self.create_small_graph()
        score = evaluator.parameter_score(small_graph)

        assert 0.0 <= score <= 1.0

    def test_depth_score(self) -> None:
        """Test depth scoring."""
        evaluator = HeuristicEvaluator(target_depth=10)

        graph = self.create_small_graph()
        score = evaluator.depth_score(graph)

        assert 0.0 <= score <= 1.0

    def test_width_score(self) -> None:
        """Test width scoring."""
        evaluator = HeuristicEvaluator()

        graph = self.create_small_graph()
        score = evaluator.width_score(graph)

        assert 0.0 <= score <= 1.0

    def test_connectivity_score(self) -> None:
        """Test connectivity scoring."""
        evaluator = HeuristicEvaluator()

        graph = self.create_small_graph()
        score = evaluator.connectivity_score(graph)

        assert 0.0 <= score <= 1.0

    def test_combined_score(self) -> None:
        """Test combined scoring."""
        evaluator = HeuristicEvaluator()

        graph = self.create_small_graph()
        score = evaluator.combined_score(graph)

        assert 0.0 <= score <= 1.0

    def test_call_method(self) -> None:
        """Test __call__ method."""
        evaluator = HeuristicEvaluator()

        graph = self.create_small_graph()
        score = evaluator(graph)

        assert 0.0 <= score <= 1.0

    def test_get_all_scores(self) -> None:
        """Test getting all scores."""
        evaluator = HeuristicEvaluator()

        graph = self.create_small_graph()
        scores = evaluator.get_all_scores(graph)

        assert "parameter" in scores
        assert "depth" in scores
        assert "width" in scores
        assert "connectivity" in scores
        assert "combined" in scores

        assert all(0.0 <= s <= 1.0 for s in scores.values())

    def test_small_vs_large_graph(self) -> None:
        """Test scoring difference between small and large graphs."""
        evaluator = HeuristicEvaluator(target_params=1000000)

        small = self.create_small_graph()
        large = self.create_large_graph()

        small_score = evaluator(small)
        large_score = evaluator(large)

        # Both should produce valid scores
        assert 0.0 <= small_score <= 1.0
        assert 0.0 <= large_score <= 1.0

    def test_evaluator_consistency(self) -> None:
        """Test evaluator consistency."""
        evaluator = HeuristicEvaluator()

        graph = self.create_small_graph()

        # Multiple evaluations should give same result
        score1 = evaluator(graph)
        score2 = evaluator(graph)
        score3 = evaluator(graph)

        assert score1 == score2 == score3

    def test_evaluator_with_search_space(self) -> None:
        """Test evaluator with sampled architectures."""
        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64], kernel_size=3),
            Layer.relu(),
            Layer.output(units=10)
        )

        evaluator = HeuristicEvaluator()

        scores = []
        for _ in range(10):
            arch = space.sample()
            score = evaluator(arch)
            scores.append(score)

        # All should be valid
        assert all(0.0 <= s <= 1.0 for s in scores)

        # Should have some variation
        assert len(set(scores)) > 1

    def test_parameter_weight_impact(self) -> None:
        """Test impact of parameter weight."""
        graph = self.create_large_graph()

        # High parameter weight
        eval_high_param = HeuristicEvaluator(
            param_weight=0.9,
            depth_weight=0.05,
            width_weight=0.025,
            connectivity_weight=0.025
        )

        # Low parameter weight
        eval_low_param = HeuristicEvaluator(
            param_weight=0.1,
            depth_weight=0.3,
            width_weight=0.3,
            connectivity_weight=0.3
        )

        score_high = eval_high_param(graph)
        score_low = eval_low_param(graph)

        # Both valid
        assert 0.0 <= score_high <= 1.0
        assert 0.0 <= score_low <= 1.0

    def test_edge_cases(self) -> None:
        """Test edge cases."""
        evaluator = HeuristicEvaluator()

        # Single node graph
        single_graph = ModelGraph()
        node = GraphNode.create("input", {"shape": (3, 32, 32)})
        single_graph.add_node(node)

        score = evaluator(single_graph)
        assert 0.0 <= score <= 1.0

    def test_zero_parameters(self) -> None:
        """Test with graph having minimal parameters."""
        evaluator = HeuristicEvaluator()

        graph = ModelGraph()
        input_node = GraphNode.create("input", {"shape": (3, 32, 32)})
        relu = GraphNode.create("relu")
        output = GraphNode.create("dense", {"units": 1})

        graph.add_node(input_node)
        graph.add_node(relu)
        graph.add_node(output)
        graph.add_edge(GraphEdge(input_node, relu))
        graph.add_edge(GraphEdge(relu, output))

        score = evaluator(graph)
        assert 0.0 <= score <= 1.0


class TestEvaluatorComparison:
    """Test comparing different evaluator configurations."""

    def create_test_architectures(self) -> list:
        """Create test architectures."""
        architectures = []

        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5]),
            Layer.relu(),
            Layer.maxpool(pool_size=2),
            Layer.dense(units=[128, 256]),
            Layer.output(units=10)
        )

        for _ in range(20):
            architectures.append(space.sample())

        return architectures

    def test_different_target_params(self) -> None:
        """Test with different target parameters."""
        architectures = self.create_test_architectures()

        eval_small = HeuristicEvaluator(target_params=100000)
        eval_large = HeuristicEvaluator(target_params=10000000)

        for arch in architectures[:5]:
            score_small = eval_small(arch)
            score_large = eval_large(arch)

            assert 0.0 <= score_small <= 1.0
            assert 0.0 <= score_large <= 1.0

    def test_evaluator_ranking(self) -> None:
        """Test that evaluator produces consistent rankings."""
        architectures = self.create_test_architectures()
        evaluator = HeuristicEvaluator()

        # Score all architectures
        scored = [(arch, evaluator(arch)) for arch in architectures]

        # Sort by score
        sorted_archs = sorted(scored, key=lambda x: x[1], reverse=True)

        # Best should have higher score than worst
        best_score = sorted_archs[0][1]
        worst_score = sorted_archs[-1][1]

        assert best_score >= worst_score


def test_evaluator_integration() -> None:
    """Integration test for evaluator."""
    from morphml.optimizers import RandomSearch

    # Create search space
    space = SearchSpace("integration")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=[32, 64], kernel_size=3),
        Layer.relu(),
        Layer.maxpool(pool_size=2),
        Layer.dense(units=[128, 256]),
        Layer.dropout(rate=0.5),
        Layer.output(units=10)
    )

    # Use heuristic evaluator
    evaluator = HeuristicEvaluator()

    # Run search
    rs = RandomSearch(space, num_samples=30)
    best = rs.optimize(evaluator)

    # Best should be valid
    assert best is not None
    assert best.is_evaluated()
    assert 0.0 <= best.fitness <= 1.0

    # Get all evaluated and check distribution
    all_evaluated = rs.get_all_evaluated()
    fitnesses = [ind.fitness for ind in all_evaluated]

    # Should have variation
    assert len(set(fitnesses)) > 1
    assert min(fitnesses) >= 0.0
    assert max(fitnesses) <= 1.0
