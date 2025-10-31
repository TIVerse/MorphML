"""Graph node representation for neural architecture."""

import uuid
from typing import Any, Dict, List, Optional

from morphml.exceptions import GraphError


class GraphNode:
    """
    Represents a single operation/layer in a neural architecture graph.

    Each node contains:
    - Unique identifier
    - Operation type (conv2d, maxpool, dense, etc.)
    - Operation parameters (filters, kernel_size, etc.)
    - Connections to other nodes (predecessors/successors)
    - Metadata for tracking

    Attributes:
        id: Unique node identifier
        operation: Operation type
        params: Operation parameters
        predecessors: List of predecessor nodes
        successors: List of successor nodes
        metadata: Additional metadata

    Example:
        >>> node = GraphNode.create(
        ...     operation='conv2d',
        ...     params={'filters': 64, 'kernel_size': 3}
        ... )
        >>> node.get_param('filters')
        64
    """

    def __init__(
        self,
        node_id: str,
        operation: str,
        params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize graph node.

        Args:
            node_id: Unique identifier
            operation: Operation type
            params: Operation parameters
            metadata: Additional metadata
        """
        self.id = node_id
        self.operation = operation
        self.params = params or {}
        self.metadata = metadata or {}

        # Connections
        self.predecessors: List[GraphNode] = []
        self.successors: List[GraphNode] = []

    @classmethod
    def create(
        cls,
        operation: str,
        params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "GraphNode":
        """
        Factory method to create a new node with auto-generated ID.

        Args:
            operation: Operation type
            params: Operation parameters
            metadata: Additional metadata

        Returns:
            New GraphNode instance
        """
        node_id = str(uuid.uuid4())
        return cls(node_id, operation, params, metadata)

    def add_predecessor(self, node: "GraphNode") -> None:
        """
        Add a predecessor node.

        Args:
            node: Predecessor node to add
        """
        if node not in self.predecessors:
            self.predecessors.append(node)

    def add_successor(self, node: "GraphNode") -> None:
        """
        Add a successor node.

        Args:
            node: Successor node to add
        """
        if node not in self.successors:
            self.successors.append(node)

    def remove_predecessor(self, node: "GraphNode") -> None:
        """Remove a predecessor node."""
        if node in self.predecessors:
            self.predecessors.remove(node)

    def remove_successor(self, node: "GraphNode") -> None:
        """Remove a successor node."""
        if node in self.successors:
            self.successors.remove(node)

    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Get operation parameter.

        Args:
            key: Parameter key
            default: Default value if key not found

        Returns:
            Parameter value
        """
        return self.params.get(key, default)

    def set_param(self, key: str, value: Any) -> None:
        """
        Set operation parameter.

        Args:
            key: Parameter key
            value: Parameter value
        """
        self.params[key] = value

    def clone(self) -> "GraphNode":
        """
        Create a deep copy of this node.

        Returns:
            Cloned GraphNode (new ID, same operation and params)
        """
        return GraphNode.create(
            operation=self.operation,
            params=self.params.copy(),
            metadata=self.metadata.copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize node to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "operation": self.operation,
            "params": self.params,
            "metadata": self.metadata,
            "predecessor_ids": [p.id for p in self.predecessors],
            "successor_ids": [s.id for s in self.successors],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNode":
        """
        Deserialize node from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            GraphNode instance

        Note:
            Predecessor/successor connections must be restored separately
        """
        return cls(
            node_id=data["id"],
            operation=data["operation"],
            params=data.get("params", {}),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        """String representation of node."""
        return f"GraphNode(id={self.id[:8]}, operation={self.operation}, params={self.params})"

    def __eq__(self, other: object) -> bool:
        """Check equality based on ID."""
        if not isinstance(other, GraphNode):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)
