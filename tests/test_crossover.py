"""Tests for crossover operators."""

import pytest

from morphml.core.crossover import (
    AdaptiveCrossover,
    CrossoverOperator,
    LayerWiseCrossover,
    MultiParentCrossover,
    SinglePointCrossover,
    SubgraphCrossover,
    TwoPointCrossover,
    UniformCrossover,
    get_crossover_operator,
)
from morphml.core.dsl import Layer, SearchSpace
from morphml.core.graph import GraphEdge, GraphNode, ModelGraph


class TestCrossoverOperators:
    """Test crossover operators."""

    def create_sample_graph(self, num_nodes: int = 5) -> ModelGraph:
        """Create sample graph."""
        graph = ModelGraph()
        
        input_node = GraphNode.create("input", {"shape": (3, 32, 32)})
        graph.add_node(input_node)
        
        prev = input_node
        for i in range(num_nodes - 2):
            node = GraphNode.create("conv2d", {"filters": 32 * (i + 1)})
            graph.add_node(node)
            graph.add_edge(GraphEdge(prev, node))
            prev = node
        
        output = GraphNode.create("dense", {"units": 10})
        graph.add_node(output)
        graph.add_edge(GraphEdge(prev, output))
        
        return graph

    def test_single_point_crossover(self) -> None:
        """Test single-point crossover."""
        parent1 = self.create_sample_graph(5)
        parent2 = self.create_sample_graph(6)
        
        operator = SinglePointCrossover()
        child1, child2 = operator.crossover(parent1, parent2)
        
        assert child1 is not None
        assert child2 is not None
        assert len(child1.nodes) > 0
        assert len(child2.nodes) > 0

    def test_two_point_crossover(self) -> None:
        """Test two-point crossover."""
        parent1 = self.create_sample_graph(6)
        parent2 = self.create_sample_graph(7)
        
        operator = TwoPointCrossover()
        child1, child2 = operator.crossover(parent1, parent2)
        
        assert child1 is not None
        assert child2 is not None

    def test_uniform_crossover(self) -> None:
        """Test uniform crossover."""
        parent1 = self.create_sample_graph(5)
        parent2 = self.create_sample_graph(5)
        
        operator = UniformCrossover(swap_probability=0.5)
        child1, child2 = operator.crossover(parent1, parent2)
        
        assert child1 is not None
        assert child2 is not None

    def test_subgraph_crossover(self) -> None:
        """Test subgraph crossover."""
        parent1 = self.create_sample_graph(6)
        parent2 = self.create_sample_graph(6)
        
        operator = SubgraphCrossover()
        child1, child2 = operator.crossover(parent1, parent2)
        
        assert child1 is not None
        assert child2 is not None

    def test_layerwise_crossover(self) -> None:
        """Test layer-wise crossover."""
        parent1 = self.create_sample_graph(5)
        parent2 = self.create_sample_graph(5)
        
        operator = LayerWiseCrossover()
        child1, child2 = operator.crossover(parent1, parent2)
        
        assert child1 is not None
        assert child2 is not None

    def test_adaptive_crossover(self) -> None:
        """Test adaptive crossover."""
        parent1 = self.create_sample_graph(5)
        parent2 = self.create_sample_graph(5)
        
        operator = AdaptiveCrossover()
        child1, child2 = operator.crossover(parent1, parent2)
        
        assert child1 is not None
        assert child2 is not None

    def test_multi_parent_crossover(self) -> None:
        """Test multi-parent crossover."""
        parent1 = self.create_sample_graph(5)
        parent2 = self.create_sample_graph(6)
        parent3 = self.create_sample_graph(5)
        
        operator = MultiParentCrossover(num_parents=3)
        child = operator.crossover_multi([parent1, parent2, parent3])
        
        assert child is not None
        assert len(child.nodes) > 0

    def test_crossover_with_small_graphs(self) -> None:
        """Test crossover with minimal graphs."""
        graph1 = ModelGraph()
        input1 = GraphNode.create("input", {"shape": (3, 32, 32)})
        output1 = GraphNode.create("dense", {"units": 10})
        graph1.add_node(input1)
        graph1.add_node(output1)
        graph1.add_edge(GraphEdge(input1, output1))
        
        graph2 = graph1.clone()
        
        operator = SinglePointCrossover()
        child1, child2 = operator.crossover(graph1, graph2)
        
        assert child1 is not None
        assert child2 is not None

    def test_crossover_preserves_validity(self) -> None:
        """Test that crossover produces valid graphs."""
        parent1 = self.create_sample_graph(5)
        parent2 = self.create_sample_graph(6)
        
        operators = [
            SinglePointCrossover(),
            TwoPointCrossover(),
            UniformCrossover(),
            SubgraphCrossover(),
            LayerWiseCrossover(),
        ]
        
        for operator in operators:
            child1, child2 = operator.crossover(parent1, parent2)
            
            # Check basic validity
            assert len(child1.nodes) > 0
            assert len(child2.nodes) > 0

    def test_get_crossover_operator(self) -> None:
        """Test getting operator by name."""
        operator = get_crossover_operator("single_point")
        
        assert isinstance(operator, SinglePointCrossover)

    def test_get_crossover_operator_with_params(self) -> None:
        """Test getting operator with parameters."""
        operator = get_crossover_operator("uniform", swap_probability=0.7)
        
        assert isinstance(operator, UniformCrossover)
        assert operator.swap_probability == 0.7

    def test_invalid_crossover_operator(self) -> None:
        """Test invalid operator name."""
        with pytest.raises(ValueError):
            get_crossover_operator("invalid_operator")


class TestCrossoverWithSearchSpace:
    """Test crossover with search spaces."""

    def create_space(self) -> SearchSpace:
        """Create test search space."""
        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64], kernel_size=3),
            Layer.relu(),
            Layer.maxpool(pool_size=2),
            Layer.conv2d(filters=[64, 128], kernel_size=3),
            Layer.relu(),
            Layer.dense(units=[128, 256]),
            Layer.output(units=10)
        )
        return space

    def test_crossover_with_sampled_architectures(self) -> None:
        """Test crossover with sampled architectures."""
        space = self.create_space()
        
        parent1 = space.sample()
        parent2 = space.sample()
        
        operator = SinglePointCrossover()
        child1, child2 = operator.crossover(parent1, parent2)
        
        assert child1 is not None
        assert child2 is not None

    def test_multiple_crossovers(self) -> None:
        """Test multiple crossover operations."""
        space = self.create_space()
        
        parents = [space.sample() for _ in range(10)]
        
        operator = TwoPointCrossover()
        
        offspring = []
        for i in range(0, len(parents), 2):
            child1, child2 = operator.crossover(parents[i], parents[i + 1])
            offspring.extend([child1, child2])
        
        assert len(offspring) == 10

    def test_crossover_diversity(self) -> None:
        """Test that crossover produces diverse offspring."""
        space = self.create_space()
        
        parent1 = space.sample()
        parent2 = space.sample()
        
        operator = UniformCrossover(swap_probability=0.5)
        
        offspring = []
        for _ in range(10):
            child1, child2 = operator.crossover(parent1, parent2)
            offspring.extend([child1, child2])
        
        # Check diversity
        node_counts = [len(child.nodes) for child in offspring]
        assert len(set(node_counts)) > 1  # Should have variation


class TestAdaptiveCrossover:
    """Test adaptive crossover behavior."""

    def create_similar_graphs(self) -> tuple:
        """Create similar graphs."""
        graph1 = ModelGraph()
        graph2 = ModelGraph()
        
        for i in range(5):
            node1 = GraphNode.create("conv2d", {"filters": 32})
            node2 = GraphNode.create("conv2d", {"filters": 32})
            graph1.add_node(node1)
            graph2.add_node(node2)
        
        return graph1, graph2

    def create_different_graphs(self) -> tuple:
        """Create different graphs."""
        graph1 = ModelGraph()
        graph2 = ModelGraph()
        
        # Different operations
        node1 = GraphNode.create("conv2d", {"filters": 32})
        node2 = GraphNode.create("dense", {"units": 128})
        node3 = GraphNode.create("maxpool", {"pool_size": 2})
        node4 = GraphNode.create("dropout", {"rate": 0.5})
        
        graph1.add_node(node1)
        graph1.add_node(node3)
        
        graph2.add_node(node2)
        graph2.add_node(node4)
        
        return graph1, graph2

    def test_adaptive_with_similar_parents(self) -> None:
        """Test adaptive crossover with similar parents."""
        parent1, parent2 = self.create_similar_graphs()
        
        operator = AdaptiveCrossover()
        child1, child2 = operator.crossover(parent1, parent2)
        
        assert child1 is not None
        assert child2 is not None

    def test_adaptive_with_different_parents(self) -> None:
        """Test adaptive crossover with different parents."""
        parent1, parent2 = self.create_different_graphs()
        
        operator = AdaptiveCrossover()
        child1, child2 = operator.crossover(parent1, parent2)
        
        assert child1 is not None
        assert child2 is not None


class TestMultiParentCrossover:
    """Test multi-parent crossover."""

    def create_graphs(self, n: int) -> list:
        """Create n graphs."""
        graphs = []
        for i in range(n):
            graph = ModelGraph()
            for j in range(4 + i):
                node = GraphNode.create("conv2d", {"filters": 32 * (j + 1)})
                graph.add_node(node)
            graphs.append(graph)
        return graphs

    def test_three_parent_crossover(self) -> None:
        """Test crossover with 3 parents."""
        parents = self.create_graphs(3)
        
        operator = MultiParentCrossover(num_parents=3)
        child = operator.crossover_multi(parents)
        
        assert child is not None
        assert len(child.nodes) > 0

    def test_five_parent_crossover(self) -> None:
        """Test crossover with 5 parents."""
        parents = self.create_graphs(5)
        
        operator = MultiParentCrossover(num_parents=5)
        child = operator.crossover_multi(parents)
        
        assert child is not None

    def test_single_parent_multi_crossover(self) -> None:
        """Test multi-parent crossover with single parent."""
        parents = self.create_graphs(1)
        
        operator = MultiParentCrossover(num_parents=3)
        child = operator.crossover_multi(parents)
        
        # Should return clone
        assert child is not None
        assert len(child.nodes) == len(parents[0].nodes)


def test_crossover_integration() -> None:
    """Integration test for crossover operators."""
    # Create search space
    space = SearchSpace("integration_test")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5]),
        Layer.relu(),
        Layer.batchnorm(),
        Layer.maxpool(pool_size=[2, 3]),
        Layer.conv2d(filters=[64, 128, 256], kernel_size=3),
        Layer.relu(),
        Layer.dense(units=[128, 256, 512]),
        Layer.dropout(rate=[0.3, 0.5]),
        Layer.output(units=10)
    )
    
    # Sample parents
    parents = [space.sample() for _ in range(10)]
    
    # Test all operators
    operators = {
        "single_point": SinglePointCrossover(),
        "two_point": TwoPointCrossover(),
        "uniform": UniformCrossover(),
        "subgraph": SubgraphCrossover(),
        "layerwise": LayerWiseCrossover(),
        "adaptive": AdaptiveCrossover(),
    }
    
    offspring_by_operator = {}
    
    for name, operator in operators.items():
        offspring = []
        for i in range(0, len(parents), 2):
            child1, child2 = operator.crossover(parents[i], parents[i + 1])
            offspring.extend([child1, child2])
        
        offspring_by_operator[name] = offspring
    
    # Verify all operators produced offspring
    for name, offspring in offspring_by_operator.items():
        assert len(offspring) == 10, f"{name} didn't produce correct number of offspring"
        assert all(len(child.nodes) > 0 for child in offspring), f"{name} produced empty graphs"


def test_crossover_in_ga() -> None:
    """Test using crossover in genetic algorithm."""
    from morphml.evaluation import HeuristicEvaluator
    from morphml.optimizers import GeneticAlgorithm
    
    # Create search space
    space = SearchSpace("ga_crossover_test")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=[32, 64], kernel_size=3),
        Layer.relu(),
        Layer.maxpool(pool_size=2),
        Layer.dense(units=[128, 256]),
        Layer.output(units=10)
    )
    
    # Run GA with custom crossover
    ga = GeneticAlgorithm(
        search_space=space,
        population_size=10,
        num_generations=3,
        crossover_rate=0.8
    )
    
    evaluator = HeuristicEvaluator()
    best = ga.optimize(evaluator)
    
    assert best is not None
    assert best.is_evaluated()
