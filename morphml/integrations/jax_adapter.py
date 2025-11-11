"""JAX/Flax adapter for MorphML.

Converts ModelGraph to Flax Module for functional neural networks.

Example:
    >>> from morphml.integrations import JAXAdapter
    >>> adapter = JAXAdapter()
    >>> model = adapter.build_model(graph)
    >>> params = model.init(rng, x)
    >>> output = model.apply(params, x)
"""

from typing import Optional, Tuple

try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    nn = None

from morphml.core.graph import GraphNode, ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class JAXAdapter:
    """
    Convert ModelGraph to JAX/Flax Module.

    Provides functional neural network implementation using JAX and Flax.

    Example:
        >>> adapter = JAXAdapter()
        >>> model = adapter.build_model(graph)
        >>> rng = jax.random.PRNGKey(0)
        >>> params = model.init(rng, jnp.ones((1, 32, 32, 3)))
        >>> output = model.apply(params, x)
    """

    def __init__(self):
        """Initialize JAX adapter."""
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX and Flax are required for JAXAdapter. " "Install with: pip install jax flax"
            )
        logger.info("Initialized JAXAdapter")

    def build_model(
        self, graph: ModelGraph, input_shape: Optional[Tuple[int, ...]] = None
    ) -> nn.Module:
        """
        Build Flax module from graph.

        Args:
            graph: ModelGraph to convert
            input_shape: Input shape (H, W, C) for JAX

        Returns:
            Flax Module instance

        Example:
            >>> model = adapter.build_model(graph, input_shape=(32, 32, 3))
        """
        return GraphModule(graph, input_shape)


class GraphModule(nn.Module):
    """
    Flax module generated from ModelGraph.

    Implements functional neural network following graph topology.

    Attributes:
        graph: Source ModelGraph
        input_shape: Expected input shape
    """

    graph: ModelGraph
    input_shape: Optional[Tuple[int, ...]] = None

    def setup(self):
        """Setup layers."""
        self.layers = {}

        for node_id, node in self.graph.nodes.items():
            layer = self._create_layer(node)
            if layer is not None:
                self.layers[str(node_id)] = layer

    def _create_layer(self, node: GraphNode):
        """
        Create Flax layer from node.

        Args:
            node: GraphNode to convert

        Returns:
            Flax layer or None
        """
        op = node.operation
        params = node.params

        if op == "input":
            return None

        elif op == "conv2d":
            return nn.Conv(
                features=params.get("filters", 64),
                kernel_size=(params.get("kernel_size", 3),) * 2,
                strides=(params.get("stride", 1),) * 2,
                padding=params.get("padding", "SAME"),
            )

        elif op == "dense":
            return nn.Dense(features=params.get("units", 10))

        elif op == "batchnorm":
            return nn.BatchNorm()

        elif op == "dropout":
            return nn.Dropout(rate=params.get("rate", 0.5))

        else:
            return None

    @nn.compact
    def __call__(self, x, training: bool = False):
        """
        Forward pass.

        Args:
            x: Input array
            training: Whether in training mode

        Returns:
            Output array
        """
        # Track outputs
        outputs = {}

        for node in self.graph.topological_sort():
            layer = self.layers.get(str(node.id))

            # Get input
            if not node.predecessors:
                node_input = x
            else:
                pred_outputs = [outputs[pred.id] for pred in node.predecessors]

                if len(pred_outputs) == 1:
                    node_input = pred_outputs[0]
                else:
                    # Concatenate along channel dimension
                    node_input = jnp.concatenate(pred_outputs, axis=-1)

            # Apply layer
            if layer is not None:
                if isinstance(layer, nn.Dropout):
                    outputs[node.id] = layer(node_input, deterministic=not training)
                elif isinstance(layer, nn.BatchNorm):
                    outputs[node.id] = layer(node_input, use_running_average=not training)
                else:
                    outputs[node.id] = layer(node_input)
            else:
                # Apply activation or pass through
                op = node.operation
                if op == "relu":
                    outputs[node.id] = nn.relu(node_input)
                elif op == "sigmoid":
                    outputs[node.id] = nn.sigmoid(node_input)
                elif op == "tanh":
                    outputs[node.id] = nn.tanh(node_input)
                elif op == "softmax":
                    outputs[node.id] = nn.softmax(node_input)
                elif op == "maxpool":
                    pool_size = node.params.get("pool_size", 2)
                    outputs[node.id] = nn.max_pool(
                        node_input,
                        window_shape=(pool_size, pool_size),
                        strides=(pool_size, pool_size),
                    )
                elif op == "avgpool":
                    pool_size = node.params.get("pool_size", 2)
                    outputs[node.id] = nn.avg_pool(
                        node_input,
                        window_shape=(pool_size, pool_size),
                        strides=(pool_size, pool_size),
                    )
                elif op == "flatten":
                    outputs[node.id] = node_input.reshape((node_input.shape[0], -1))
                else:
                    outputs[node.id] = node_input

        # Return output
        output_nodes = [n for n in self.graph.nodes.values() if not n.successors]
        if output_nodes:
            return outputs[output_nodes[0].id]
        else:
            return outputs[list(outputs.keys())[-1]]
