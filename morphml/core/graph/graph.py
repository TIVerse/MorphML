"""Model graph representation for neural architectures."""

import hashlib
import json
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from morphml.core.graph.edge import GraphEdge
from morphml.core.graph.node import GraphNode
from morphml.exceptions import GraphError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class ModelGraph:
    """
    Directed Acyclic Graph (DAG) representation of a neural architecture.

    A ModelGraph consists of:
    - Nodes: Operations/layers (conv2d, maxpool, dense, etc.)
    - Edges: Connections between operations
    - Metadata: Additional information about the architecture

    The graph must be a valid DAG (no cycles) and have exactly one
    input node and one output node.

    Attributes:
        nodes: Dictionary mapping node IDs to GraphNode instances
        edges: Dictionary mapping edge IDs to GraphEdge instances
        metadata: Additional metadata

    Example:
        >>> graph = ModelGraph()
        >>> input_node = graph.add_node(GraphNode.create('input'))
        >>> conv = graph.add_node(GraphNode.create('conv2d', {'filters': 64}))
        >>> graph.add_edge(GraphEdge(input_node, conv))
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize empty model graph.

        Args:
            metadata: Optional metadata
        """
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.metadata = metadata or {}

    def add_node(self, node: GraphNode) -> GraphNode:
        """
        Add a node to the graph.

        Args:
            node: GraphNode to add

        Returns:
            The added node

        Raises:
            GraphError: If node with same ID already exists
        """
        if node.id in self.nodes:
            raise GraphError(f"Node with ID {node.id} already exists")

        self.nodes[node.id] = node
        logger.debug(f"Added node: {node.operation} (id={node.id[:8]})")
        return node

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        """
        Add an edge to the graph.

        Automatically updates predecessor/successor relationships.

        Args:
            edge: GraphEdge to add

        Returns:
            The added edge

        Raises:
            GraphError: If edge creates a cycle or nodes not in graph
        """
        # Validate nodes exist
        if edge.source.id not in self.nodes:
            raise GraphError(f"Source node {edge.source.id} not in graph")
        if edge.target.id not in self.nodes:
            raise GraphError(f"Target node {edge.target.id} not in graph")

        # Check for cycles
        if self._would_create_cycle(edge.source, edge.target):
            raise GraphError(
                f"Adding edge from {edge.source.operation} to {edge.target.operation} "
                "would create a cycle"
            )

        # Add edge
        self.edges[edge.id] = edge

        # Update connections
        edge.source.add_successor(edge.target)
        edge.target.add_predecessor(edge.source)

        logger.debug(
            f"Added edge: {edge.source.operation} -> {edge.target.operation} " f"(id={edge.id[:8]})"
        )
        return edge

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node and its connected edges.

        Args:
            node_id: ID of node to remove

        Raises:
            GraphError: If node not found
        """
        if node_id not in self.nodes:
            raise GraphError(f"Node {node_id} not found")

        node = self.nodes[node_id]

        # Remove connected edges
        edges_to_remove = [
            edge_id
            for edge_id, edge in self.edges.items()
            if edge.source.id == node_id or edge.target.id == node_id
        ]

        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)

        # Remove node
        del self.nodes[node_id]
        logger.debug(f"Removed node: {node.operation} (id={node_id[:8]})")

    def remove_edge(self, edge_id: str) -> None:
        """
        Remove an edge from the graph.

        Args:
            edge_id: ID of edge to remove

        Raises:
            GraphError: If edge not found
        """
        if edge_id not in self.edges:
            raise GraphError(f"Edge {edge_id} not found")

        edge = self.edges[edge_id]

        # Update connections
        edge.source.remove_successor(edge.target)
        edge.target.remove_predecessor(edge.source)

        # Remove edge
        del self.edges[edge_id]
        logger.debug(f"Removed edge: {edge_id[:8]}")

    def get_input_nodes(self) -> List[GraphNode]:
        """
        Get all input nodes (nodes with no predecessors).

        Returns:
            List of input nodes
        """
        return [node for node in self.nodes.values() if len(node.predecessors) == 0]

    def get_output_nodes(self) -> List[GraphNode]:
        """
        Get all output nodes (nodes with no successors).

        Returns:
            List of output nodes
        """
        return [node for node in self.nodes.values() if len(node.successors) == 0]

    def get_input_node(self) -> Optional[GraphNode]:
        """Get single input node (returns first if multiple)."""
        inputs = self.get_input_nodes()
        return inputs[0] if inputs else None

    def get_output_node(self) -> Optional[GraphNode]:
        """Get single output node (returns first if multiple)."""
        outputs = self.get_output_nodes()
        return outputs[0] if outputs else None

    def topological_sort(self) -> List[GraphNode]:
        """
        Return nodes in topological order.

        Returns:
            List of nodes in topological order

        Raises:
            GraphError: If graph has cycles
        """
        try:
            nx_graph = self.to_networkx()
            sorted_ids = list(nx.topological_sort(nx_graph))
            return [self.nodes[node_id] for node_id in sorted_ids]
        except nx.NetworkXError as e:
            raise GraphError(f"Graph is not a DAG: {e}") from e

    def is_valid(self) -> bool:
        """
        Check if graph is valid.

        A valid graph:
        - Is a DAG (no cycles)
        - Has at least one input node
        - Has at least one output node
        - All nodes are reachable from input(s)

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check for cycles
            self.topological_sort()

            # Check for input/output nodes
            if not self.get_input_nodes():
                return False
            if not self.get_output_nodes():
                return False

            # Check all nodes are reachable
            if not self._all_nodes_reachable():
                return False

            return True

        except (GraphError, nx.NetworkXError):
            return False

    def is_valid_dag(self) -> bool:
        """
        Check if graph is a valid DAG.

        Alias for is_valid() method.

        Returns:
            True if valid DAG, False otherwise
        """
        return self.is_valid()

    def clone(self) -> "ModelGraph":
        """
        Create a deep copy of this graph.

        Returns:
            Cloned ModelGraph
        """
        # Clone metadata
        cloned = ModelGraph(metadata=self.metadata.copy())

        # Clone nodes (create mapping)
        node_mapping = {}
        for node_id, node in self.nodes.items():
            cloned_node = node.clone()
            cloned.nodes[cloned_node.id] = cloned_node
            node_mapping[node_id] = cloned_node

        # Clone edges (using new nodes)
        for edge in self.edges.values():
            new_source = node_mapping[edge.source.id]
            new_target = node_mapping[edge.target.id]
            cloned_edge = GraphEdge(
                source=new_source,
                target=new_target,
                operation=edge.operation,
                metadata=edge.metadata.copy(),
            )
            cloned.add_edge(cloned_edge)

        return cloned

    def to_networkx(self) -> nx.DiGraph:
        """
        Convert to NetworkX DiGraph.

        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()

        # Add nodes
        for node_id, node in self.nodes.items():
            G.add_node(node_id, **node.to_dict())

        # Add edges
        for edge in self.edges.values():
            G.add_edge(edge.source.id, edge.target.id, **edge.to_dict())

        return G

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize graph to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """
        Serialize graph to JSON string.

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelGraph":
        """
        Deserialize graph from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ModelGraph instance
        """
        graph = cls(metadata=data.get("metadata", {}))

        # Restore nodes
        for node_data in data["nodes"]:
            node = GraphNode.from_dict(node_data)
            graph.nodes[node.id] = node

        # Restore edges
        for edge_data in data["edges"]:
            edge = GraphEdge.from_dict(edge_data, graph.nodes)
            graph.edges[edge.id] = edge

            # Restore connections
            edge.source.add_successor(edge.target)
            edge.target.add_predecessor(edge.source)

        return graph

    @classmethod
    def from_json(cls, json_str: str) -> "ModelGraph":
        """
        Deserialize graph from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            ModelGraph instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def hash(self) -> str:
        """
        Compute hash of graph structure.

        Used for deduplication and caching.

        Returns:
            SHA256 hash of graph
        """
        # Create canonical representation
        canonical = {
            "nodes": sorted(
                [
                    (node.operation, tuple(sorted(node.params.items())))
                    for node in self.nodes.values()
                ]
            ),
            "edges": sorted(
                [(edge.source.operation, edge.target.operation) for edge in self.edges.values()]
            ),
        }

        canonical_str = json.dumps(canonical, sort_keys=True)
        return hashlib.sha256(canonical_str.encode()).hexdigest()

    def get_depth(self) -> int:
        """
        Get maximum depth of graph.

        Returns:
            Maximum depth (longest path from input to output)
        """
        try:
            nx_graph = self.to_networkx()
            length: int = nx.dag_longest_path_length(nx_graph)
            return length
        except nx.NetworkXError:
            return 0

    def get_max_width(self) -> int:
        """
        Get maximum width of graph.

        Returns:
            Maximum number of nodes at any depth level
        """
        # Compute levels
        levels: Dict[int, int] = {}

        def compute_level(node: GraphNode, level: int = 0) -> None:
            levels[level] = levels.get(level, 0) + 1
            for successor in node.successors:
                compute_level(successor, level + 1)

        # Start from input nodes
        for input_node in self.get_input_nodes():
            compute_level(input_node)

        return max(levels.values()) if levels else 0

    def estimate_parameters(self) -> int:
        """
        Estimate number of parameters (simplified).

        Returns:
            Estimated parameter count
        """
        total_params = 0

        for node in self.nodes.values():
            if node.operation == "conv2d":
                filters = node.get_param("filters", 64)
                kernel_size = node.get_param("kernel_size", 3)
                in_channels = node.get_param("in_channels", 3)
                params = in_channels * filters * kernel_size * kernel_size
                total_params += params

            elif node.operation == "dense":
                units = node.get_param("units", 128)
                in_features = node.get_param("in_features", 512)
                params = in_features * units
                total_params += params

        return total_params

    def _would_create_cycle(self, source: GraphNode, target: GraphNode) -> bool:
        """
        Check if adding edge would create a cycle.

        Args:
            source: Source node
            target: Target node

        Returns:
            True if would create cycle, False otherwise
        """
        # DFS from target to see if we can reach source
        visited: Set[str] = set()

        def dfs(node: GraphNode) -> bool:
            if node.id in visited:
                return False
            if node.id == source.id:
                return True

            visited.add(node.id)
            for successor in node.successors:
                if dfs(successor):
                    return True

            return False

        return dfs(target)

    def _all_nodes_reachable(self) -> bool:
        """Check if all nodes are reachable from input nodes."""
        reachable: Set[str] = set()

        def dfs(node: GraphNode) -> None:
            if node.id in reachable:
                return
            reachable.add(node.id)
            for successor in node.successors:
                dfs(successor)

        # DFS from all input nodes
        for input_node in self.get_input_nodes():
            dfs(input_node)

        return len(reachable) == len(self.nodes)

    def __repr__(self) -> str:
        """String representation of graph."""
        return (
            f"ModelGraph(nodes={len(self.nodes)}, edges={len(self.edges)}, "
            f"depth={self.get_depth()}, hash={self.hash()[:8]})"
        )

    def __len__(self) -> int:
        """Return number of nodes."""
        return len(self.nodes)
