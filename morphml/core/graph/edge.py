"""Graph edge representation for neural architecture."""

import uuid
from typing import Any, Dict, Optional

from morphml.core.graph.node import GraphNode
from morphml.exceptions import GraphError


class GraphEdge:
    """
    Represents a connection between two nodes in a neural architecture graph.

    Each edge contains:
    - Unique identifier
    - Source and target nodes
    - Optional operation (for edge-level operations)
    - Metadata

    Attributes:
        id: Unique edge identifier
        source: Source node
        target: Target node
        operation: Optional edge operation
        metadata: Additional metadata

    Example:
        >>> source = GraphNode.create('conv2d')
        >>> target = GraphNode.create('relu')
        >>> edge = GraphEdge(source, target)
    """

    def __init__(
        self,
        source: GraphNode,
        target: GraphNode,
        operation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        edge_id: Optional[str] = None,
    ):
        """
        Initialize graph edge.

        Args:
            source: Source node
            target: Target node
            operation: Optional edge operation
            metadata: Additional metadata
            edge_id: Optional edge ID (auto-generated if None)

        Raises:
            GraphError: If source or target is None
        """
        if source is None or target is None:
            raise GraphError("Source and target nodes cannot be None")

        self.id = edge_id or str(uuid.uuid4())
        self.source = source
        self.target = target
        self.operation = operation
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize edge to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "source_id": self.source.id,
            "target_id": self.target.id,
            "operation": self.operation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], node_map: Dict[str, GraphNode]) -> "GraphEdge":
        """
        Deserialize edge from dictionary.

        Args:
            data: Dictionary representation
            node_map: Mapping of node IDs to GraphNode instances

        Returns:
            GraphEdge instance

        Raises:
            GraphError: If source or target node not found
        """
        source_id = data["source_id"]
        target_id = data["target_id"]

        if source_id not in node_map or target_id not in node_map:
            raise GraphError(f"Source or target node not found: {source_id}, {target_id}")

        return cls(
            source=node_map[source_id],
            target=node_map[target_id],
            operation=data.get("operation"),
            metadata=data.get("metadata", {}),
            edge_id=data["id"],
        )

    def __repr__(self) -> str:
        """String representation of edge."""
        op_str = f", operation={self.operation}" if self.operation else ""
        return (
            f"GraphEdge(id={self.id[:8]}, "
            f"source={self.source.operation}, "
            f"target={self.target.operation}{op_str})"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality based on ID."""
        if not isinstance(other, GraphEdge):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)
