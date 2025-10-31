"""Search space definition using the MorphML DSL.

This module provides the SearchSpace class for defining neural architecture
search spaces in a Pythonic way.

Example:
    >>> from morphml.core.dsl import SearchSpace, Layer
    >>>
    >>> # Define search space
    >>> space = SearchSpace(name="cifar10_cnn")
    >>>
    >>> # Add layers
    >>> space.add_layer(Layer.input(shape=(3, 32, 32)))
    >>> space.add_layer(Layer.conv2d(filters=[32, 64], kernel_size=[3, 5]))
    >>> space.add_layer(Layer.relu())
    >>> space.add_layer(Layer.maxpool(pool_size=2))
    >>> space.add_layer(Layer.dense(units=[128, 256, 512]))
    >>> space.add_layer(Layer.output(units=10))
    >>>
    >>> # Sample architectures
    >>> arch1 = space.sample()
    >>> arch2 = space.sample()
"""

from typing import Any, Callable, Dict, List, Optional

from morphml.core.dsl.layers import LayerSpec
from morphml.core.graph import GraphEdge, ModelGraph
from morphml.exceptions import SearchSpaceError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class SearchSpace:
    """
    Defines a search space for neural architecture search.

    A SearchSpace consists of:
    - A sequence of layer specifications
    - Constraints on valid architectures
    - Metadata about the space

    Architectures are sampled by:
    1. Sampling parameters for each layer
    2. Connecting layers sequentially
    3. Validating the resulting graph

    Attributes:
        name: Search space name
        layers: List of layer specifications
        constraints: List of constraint functions
        metadata: Additional metadata

    Example:
        >>> space = SearchSpace("my_space")
        >>> space.add_layer(Layer.conv2d(filters=[32, 64]))
        >>> space.add_layer(Layer.dense(units=10))
        >>> architecture = space.sample()
    """

    def __init__(
        self,
        name: str = "search_space",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize search space.

        Args:
            name: Search space name
            metadata: Optional metadata
        """
        self.name = name
        self.layers: List[LayerSpec] = []
        self.constraints: List[Callable[[ModelGraph], bool]] = []
        self.metadata = metadata or {}

        logger.debug(f"Created SearchSpace: {name}")

    def add_layer(self, layer_spec: LayerSpec) -> "SearchSpace":
        """
        Add a layer specification to the search space.

        Args:
            layer_spec: Layer specification to add

        Returns:
            Self (for method chaining)

        Example:
            >>> space.add_layer(Layer.conv2d(filters=64))
            >>> space.add_layer(Layer.relu())
        """
        self.layers.append(layer_spec)
        logger.debug(f"Added layer: {layer_spec.operation}")
        return self

    def add_layers(self, *layer_specs: LayerSpec) -> "SearchSpace":
        """
        Add multiple layer specifications.

        Args:
            *layer_specs: Layer specifications to add

        Returns:
            Self (for method chaining)

        Example:
            >>> space.add_layers(
            ...     Layer.conv2d(filters=64),
            ...     Layer.relu(),
            ...     Layer.maxpool()
            ... )
        """
        for layer_spec in layer_specs:
            self.add_layer(layer_spec)
        return self

    def add_constraint(self, constraint_fn: Callable[[ModelGraph], bool]) -> "SearchSpace":
        """
        Add a constraint function.

        Constraint functions should take a ModelGraph and return bool.

        Args:
            constraint_fn: Function that validates a graph

        Returns:
            Self (for method chaining)

        Example:
            >>> def max_depth(graph):
            ...     return graph.get_depth() <= 20
            >>> space.add_constraint(max_depth)
        """
        self.constraints.append(constraint_fn)
        return self

    def sample(self, max_attempts: int = 100) -> ModelGraph:
        """
        Sample a random architecture from this search space.

        Args:
            max_attempts: Maximum sampling attempts before giving up

        Returns:
            Sampled ModelGraph

        Raises:
            SearchSpaceError: If unable to sample valid architecture

        Example:
            >>> arch = space.sample()
            >>> print(arch)
        """
        for attempt in range(max_attempts):
            try:
                # Create graph
                graph = ModelGraph(metadata={"search_space": self.name})

                # Sample nodes from each layer spec
                nodes = []
                for layer_spec in self.layers:
                    node = layer_spec.sample()
                    graph.add_node(node)
                    nodes.append(node)

                # Connect sequentially
                for i in range(len(nodes) - 1):
                    edge = GraphEdge(nodes[i], nodes[i + 1])
                    graph.add_edge(edge)

                # Validate
                if not graph.is_valid():
                    logger.debug(f"Attempt {attempt + 1}: Invalid graph")
                    continue

                # Check constraints
                if not self._check_constraints(graph):
                    logger.debug(f"Attempt {attempt + 1}: Constraint violation")
                    continue

                logger.debug(f"Successfully sampled architecture (attempt {attempt + 1})")
                return graph

            except Exception as e:
                logger.debug(f"Attempt {attempt + 1} failed: {e}")
                continue

        raise SearchSpaceError(f"Failed to sample valid architecture after {max_attempts} attempts")

    def sample_batch(self, batch_size: int, max_attempts: int = 100) -> List[ModelGraph]:
        """
        Sample multiple architectures.

        Args:
            batch_size: Number of architectures to sample
            max_attempts: Max attempts per architecture

        Returns:
            List of sampled architectures

        Example:
            >>> architectures = space.sample_batch(10)
        """
        return [self.sample(max_attempts) for _ in range(batch_size)]

    def _check_constraints(self, graph: ModelGraph) -> bool:
        """
        Check if graph satisfies all constraints.

        Args:
            graph: Graph to validate

        Returns:
            True if all constraints satisfied
        """
        for constraint_fn in self.constraints:
            try:
                if not constraint_fn(graph):
                    return False
            except Exception as e:
                logger.warning(f"Constraint check failed: {e}")
                return False

        return True

    def get_complexity(self) -> Dict[str, Any]:
        """
        Estimate search space complexity.

        Returns:
            Dictionary with complexity metrics
        """
        total_combinations = 1
        param_counts: Dict[str, int] = {}

        for layer_spec in self.layers:
            layer_combinations = 1
            for param_name, values in layer_spec.param_ranges.items():
                if isinstance(values, list):
                    layer_combinations *= len(values)
                    param_counts[param_name] = param_counts.get(param_name, 0) + len(values)

            total_combinations *= layer_combinations

        return {
            "total_combinations": total_combinations,
            "num_layers": len(self.layers),
            "param_counts": param_counts,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize search space to dictionary."""
        return {
            "name": self.name,
            "layers": [layer.to_dict() for layer in self.layers],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchSpace":
        """
        Deserialize search space from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            SearchSpace instance
        """
        space = cls(name=data["name"], metadata=data.get("metadata", {}))

        for layer_data in data["layers"]:
            layer_spec = LayerSpec.from_dict(layer_data)
            space.add_layer(layer_spec)

        return space

    def __repr__(self) -> str:
        """String representation."""
        complexity = self.get_complexity()
        return (
            f"SearchSpace(name={self.name}, "
            f"layers={len(self.layers)}, "
            f"combinations={complexity['total_combinations']})"
        )

    def __len__(self) -> int:
        """Return number of layers."""
        return len(self.layers)


# Convenience functions
def create_cnn_space(
    num_classes: int = 10,
    input_shape: tuple = (3, 32, 32),
    conv_filters: Optional[List[List[int]]] = None,
    dense_units: Optional[List[List[int]]] = None,
) -> SearchSpace:
    """
    Create a standard CNN search space.

    Args:
        num_classes: Number of output classes
        input_shape: Input shape (C, H, W)
        conv_filters: List of filter options for each conv layer
        dense_units: List of unit options for each dense layer

    Returns:
        Configured SearchSpace

    Example:
        >>> space = create_cnn_space(
        ...     num_classes=10,
        ...     conv_filters=[[32, 64], [64, 128]],
        ...     dense_units=[[128, 256]]
        ... )
    """
    from morphml.core.dsl.layers import Layer

    conv_filters = conv_filters or [[32, 64, 128], [64, 128, 256]]
    dense_units = dense_units or [[128, 256, 512]]

    space = SearchSpace(name="cnn_space")

    # Input
    space.add_layer(Layer.input(shape=input_shape))

    # Conv blocks
    for filters in conv_filters:
        space.add_layer(Layer.conv2d(filters=filters, kernel_size=[3, 5]))
        space.add_layer(Layer.relu())
        space.add_layer(Layer.maxpool(pool_size=2))

    # Dense layers
    for units in dense_units:
        space.add_layer(Layer.dense(units=units))
        space.add_layer(Layer.relu())
        space.add_layer(Layer.dropout(rate=[0.3, 0.5]))

    # Output
    space.add_layer(Layer.output(units=num_classes))

    return space


def create_mlp_space(
    num_classes: int = 10,
    input_shape: tuple = (784,),
    hidden_layers: int = 3,
    units_range: Optional[List[int]] = None,
) -> SearchSpace:
    """
    Create a multi-layer perceptron search space.

    Args:
        num_classes: Number of output classes
        input_shape: Input shape
        hidden_layers: Number of hidden layers
        units_range: Range of units per layer

    Returns:
        Configured SearchSpace
    """
    from morphml.core.dsl.layers import Layer

    if units_range is None:
        units_range = [128, 256, 512]

    space = SearchSpace(name="mlp_space")

    # Input
    space.add_layer(Layer.input(shape=input_shape))

    # Hidden layers
    for _ in range(hidden_layers):
        space.add_layer(Layer.dense(units=units_range))
        space.add_layer(Layer.relu())
        space.add_layer(Layer.dropout(rate=[0.3, 0.5]))

    # Output
    space.add_layer(Layer.output(units=num_classes))

    return space
