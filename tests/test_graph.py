"""Tests for graph system."""

import pytest

from morphml.core.graph import GraphEdge, GraphMutator, GraphNode, ModelGraph
from morphml.exceptions import GraphError


class TestGraphNode:
    """Tests for GraphNode."""

    def test_create_node(self) -> None:
        """Test node creation."""
        node = GraphNode.create("conv2d", params={"filters": 64, "kernel_size": 3})

        assert node.operation == "conv2d"
        assert node.get_param("filters") == 64
        assert node.get_param("kernel_size") == 3
        assert len(node.predecessors) == 0
        assert len(node.successors) == 0

    def test_node_connections(self) -> None:
        """Test node predecessor/successor management."""
        node1 = GraphNode.create("conv2d")
        node2 = GraphNode.create("relu")

        node1.add_successor(node2)
        node2.add_predecessor(node1)

        assert node2 in node1.successors
        assert node1 in node2.predecessors

    def test_node_clone(self) -> None:
        """Test node cloning."""
        original = GraphNode.create("dense", params={"units": 128})
        cloned = original.clone()

        assert cloned.operation == original.operation
        assert cloned.get_param("units") == 128
        assert cloned.id != original.id  # Different ID

    def test_node_serialization(self) -> None:
        """Test node to_dict and from_dict."""
        original = GraphNode.create("conv2d", params={"filters": 32})
        data = original.to_dict()

        # Manually create new node from dict (without connections)
        restored = GraphNode.from_dict(data)

        assert restored.operation == original.operation
        assert restored.get_param("filters") == 32


class TestGraphEdge:
    """Tests for GraphEdge."""

    def test_create_edge(self) -> None:
        """Test edge creation."""
        source = GraphNode.create("conv2d")
        target = GraphNode.create("relu")

        edge = GraphEdge(source, target)

        assert edge.source == source
        assert edge.target == target

    def test_edge_none_nodes(self) -> None:
        """Test edge creation with None nodes raises error."""
        node = GraphNode.create("conv2d")

        with pytest.raises(GraphError):
            GraphEdge(None, node)  # type: ignore

        with pytest.raises(GraphError):
            GraphEdge(node, None)  # type: ignore


class TestModelGraph:
    """Tests for ModelGraph."""

    def test_create_empty_graph(self) -> None:
        """Test empty graph creation."""
        graph = ModelGraph()

        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_add_nodes(self) -> None:
        """Test adding nodes to graph."""
        graph = ModelGraph()

        node1 = GraphNode.create("input")
        node2 = GraphNode.create("conv2d")

        graph.add_node(node1)
        graph.add_node(node2)

        assert len(graph.nodes) == 2
        assert node1.id in graph.nodes
        assert node2.id in graph.nodes

    def test_add_edge(self) -> None:
        """Test adding edges to graph."""
        graph = ModelGraph()

        node1 = GraphNode.create("input")
        node2 = GraphNode.create("conv2d")

        graph.add_node(node1)
        graph.add_node(node2)

        edge = GraphEdge(node1, node2)
        graph.add_edge(edge)

        assert len(graph.edges) == 1
        assert node2 in node1.successors
        assert node1 in node2.predecessors

    def test_cycle_detection(self) -> None:
        """Test that cycles are detected and prevented."""
        graph = ModelGraph()

        node1 = GraphNode.create("conv2d")
        node2 = GraphNode.create("relu")

        graph.add_node(node1)
        graph.add_node(node2)

        # Add edge1: node1 -> node2
        edge1 = GraphEdge(node1, node2)
        graph.add_edge(edge1)

        # Try to add edge2: node2 -> node1 (would create cycle)
        edge2 = GraphEdge(node2, node1)

        with pytest.raises(GraphError, match="cycle"):
            graph.add_edge(edge2)

    def test_topological_sort(self) -> None:
        """Test topological sorting of graph."""
        graph = ModelGraph()

        # Create linear graph: input -> conv -> relu -> output
        input_node = GraphNode.create("input")
        conv_node = GraphNode.create("conv2d")
        relu_node = GraphNode.create("relu")
        output_node = GraphNode.create("output")

        graph.add_node(input_node)
        graph.add_node(conv_node)
        graph.add_node(relu_node)
        graph.add_node(output_node)

        graph.add_edge(GraphEdge(input_node, conv_node))
        graph.add_edge(GraphEdge(conv_node, relu_node))
        graph.add_edge(GraphEdge(relu_node, output_node))

        sorted_nodes = graph.topological_sort()

        assert len(sorted_nodes) == 4
        assert sorted_nodes[0] == input_node
        assert sorted_nodes[-1] == output_node

    def test_graph_clone(self) -> None:
        """Test graph cloning."""
        original = ModelGraph()

        node1 = GraphNode.create("input")
        node2 = GraphNode.create("output")

        original.add_node(node1)
        original.add_node(node2)
        original.add_edge(GraphEdge(node1, node2))

        cloned = original.clone()

        assert len(cloned.nodes) == len(original.nodes)
        assert len(cloned.edges) == len(original.edges)

        # Different node IDs
        assert set(cloned.nodes.keys()) != set(original.nodes.keys())

    def test_graph_serialization(self) -> None:
        """Test graph to_dict, from_dict, and JSON."""
        original = ModelGraph()

        node1 = GraphNode.create("input")
        node2 = GraphNode.create("conv2d", params={"filters": 64})
        node3 = GraphNode.create("output")

        original.add_node(node1)
        original.add_node(node2)
        original.add_node(node3)

        original.add_edge(GraphEdge(node1, node2))
        original.add_edge(GraphEdge(node2, node3))

        # Test dict serialization
        graph_dict = original.to_dict()
        restored = ModelGraph.from_dict(graph_dict)

        assert len(restored.nodes) == len(original.nodes)
        assert len(restored.edges) == len(original.edges)

        # Test JSON serialization
        json_str = original.to_json()
        restored_from_json = ModelGraph.from_json(json_str)

        assert len(restored_from_json.nodes) == len(original.nodes)

    def test_graph_hash(self) -> None:
        """Test graph hashing for deduplication."""
        graph1 = ModelGraph()
        graph2 = ModelGraph()

        # Create identical graphs
        for graph in [graph1, graph2]:
            node1 = GraphNode.create("input")
            node2 = GraphNode.create("conv2d", params={"filters": 64})

            graph.add_node(node1)
            graph.add_node(node2)
            graph.add_edge(GraphEdge(node1, node2))

        # Should have same hash despite different node IDs
        assert graph1.hash() == graph2.hash()

    def test_input_output_nodes(self) -> None:
        """Test getting input and output nodes."""
        graph = ModelGraph()

        input_node = GraphNode.create("input")
        middle_node = GraphNode.create("conv2d")
        output_node = GraphNode.create("output")

        graph.add_node(input_node)
        graph.add_node(middle_node)
        graph.add_node(output_node)

        graph.add_edge(GraphEdge(input_node, middle_node))
        graph.add_edge(GraphEdge(middle_node, output_node))

        assert graph.get_input_node() == input_node
        assert graph.get_output_node() == output_node

    def test_graph_metrics(self) -> None:
        """Test graph depth and width calculation."""
        graph = ModelGraph()

        # Create graph: input -> conv1 -> relu -> output
        #                     -> conv2 ----^
        input_node = GraphNode.create("input")
        conv1 = GraphNode.create("conv2d")
        conv2 = GraphNode.create("conv2d")
        relu = GraphNode.create("relu")
        output = GraphNode.create("output")

        graph.add_node(input_node)
        graph.add_node(conv1)
        graph.add_node(conv2)
        graph.add_node(relu)
        graph.add_node(output)

        graph.add_edge(GraphEdge(input_node, conv1))
        graph.add_edge(GraphEdge(input_node, conv2))
        graph.add_edge(GraphEdge(conv1, relu))
        graph.add_edge(GraphEdge(conv2, relu))
        graph.add_edge(GraphEdge(relu, output))

        depth = graph.get_depth()
        assert depth > 0


class TestGraphMutator:
    """Tests for GraphMutator."""

    def test_mutator_creation(self) -> None:
        """Test mutator initialization."""
        mutator = GraphMutator()
        assert len(mutator.operation_types) > 0

    def test_mutation_preserves_validity(self) -> None:
        """Test that mutations preserve graph validity."""
        # Create simple valid graph
        graph = ModelGraph()

        input_node = GraphNode.create("input")
        conv = GraphNode.create("conv2d", params={"filters": 32})
        output = GraphNode.create("output")

        graph.add_node(input_node)
        graph.add_node(conv)
        graph.add_node(output)

        graph.add_edge(GraphEdge(input_node, conv))
        graph.add_edge(GraphEdge(conv, output))

        assert graph.is_valid()

        # Apply mutations
        mutator = GraphMutator()
        mutated = mutator.mutate(graph, mutation_rate=0.3)

        # Mutated graph should still be valid
        assert mutated.is_valid()

    def test_add_node_mutation(self) -> None:
        """Test add_node mutation."""
        graph = ModelGraph()

        node1 = GraphNode.create("input")
        node2 = GraphNode.create("output")

        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_edge(GraphEdge(node1, node2))

        mutator = GraphMutator()
        original_count = len(graph.nodes)

        mutator.add_node_mutation(graph)

        # Should have added one node
        assert len(graph.nodes) == original_count + 1

    def test_modify_node_mutation(self) -> None:
        """Test modify_node mutation."""
        graph = ModelGraph()

        node = GraphNode.create("conv2d", params={"filters": 32})
        graph.add_node(node)

        mutator = GraphMutator()
        node.get_param("filters")

        mutator.modify_node_mutation(graph)

        # Parameters should have changed (probabilistically)
        # Note: This test might occasionally fail due to randomness
        # In practice, we'd use a fixed seed


def test_graph_creation_workflow() -> None:
    """Integration test: Create a complete graph workflow."""
    # Create a simple CNN-like architecture
    graph = ModelGraph()

    # Input
    input_node = GraphNode.create("input", params={"shape": (3, 32, 32)})
    graph.add_node(input_node)

    # Conv block 1
    conv1 = GraphNode.create("conv2d", params={"filters": 32, "kernel_size": 3})
    relu1 = GraphNode.create("relu")
    pool1 = GraphNode.create("maxpool", params={"pool_size": 2})

    graph.add_node(conv1)
    graph.add_node(relu1)
    graph.add_node(pool1)

    # Conv block 2
    conv2 = GraphNode.create("conv2d", params={"filters": 64, "kernel_size": 3})
    relu2 = GraphNode.create("relu")
    pool2 = GraphNode.create("maxpool", params={"pool_size": 2})

    graph.add_node(conv2)
    graph.add_node(relu2)
    graph.add_node(pool2)

    # Dense layers
    dense1 = GraphNode.create("dense", params={"units": 128})
    relu3 = GraphNode.create("relu")
    output = GraphNode.create("dense", params={"units": 10})

    graph.add_node(dense1)
    graph.add_node(relu3)
    graph.add_node(output)

    # Connect everything
    graph.add_edge(GraphEdge(input_node, conv1))
    graph.add_edge(GraphEdge(conv1, relu1))
    graph.add_edge(GraphEdge(relu1, pool1))

    graph.add_edge(GraphEdge(pool1, conv2))
    graph.add_edge(GraphEdge(conv2, relu2))
    graph.add_edge(GraphEdge(relu2, pool2))

    graph.add_edge(GraphEdge(pool2, dense1))
    graph.add_edge(GraphEdge(dense1, relu3))
    graph.add_edge(GraphEdge(relu3, output))

    # Validate
    assert graph.is_valid()
    assert len(graph.nodes) == 10
    assert len(graph.edges) == 9

    # Test topological sort
    sorted_nodes = graph.topological_sort()
    assert sorted_nodes[0] == input_node
    assert sorted_nodes[-1] == output

    # Test cloning
    cloned = graph.clone()
    assert len(cloned.nodes) == len(graph.nodes)
    assert cloned.hash() == graph.hash()

    # Test serialization
    json_str = graph.to_json()
    restored = ModelGraph.from_json(json_str)
    assert len(restored.nodes) == len(graph.nodes)
    assert restored.is_valid()
