"""TensorFlow/Keras adapter for MorphML.

Converts ModelGraph to Keras Model using Functional API.

Example:
    >>> from morphml.integrations import TensorFlowAdapter
    >>> adapter = TensorFlowAdapter()
    >>> model = adapter.build_model(graph)
    >>> model.compile(optimizer='adam', loss='categorical_crossentropy')
    >>> model.fit(x_train, y_train, validation_data=(x_val, y_val))
"""

from typing import Any, Dict, Optional, Tuple

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None
    layers = None

from morphml.core.graph import GraphNode, ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class TensorFlowAdapter:
    """
    Convert ModelGraph to TensorFlow/Keras Model.

    Uses Keras Functional API to build models from graph structure.

    Example:
        >>> adapter = TensorFlowAdapter()
        >>> model = adapter.build_model(graph, input_shape=(32, 32, 3))
        >>> model.summary()
    """

    def __init__(self):
        """Initialize TensorFlow adapter."""
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for TensorFlowAdapter. "
                "Install with: pip install tensorflow"
            )
        logger.info("Initialized TensorFlowAdapter")

    def build_model(
        self, graph: ModelGraph, input_shape: Optional[Tuple[int, ...]] = None
    ) -> "keras.Model":
        """
        Build Keras model from graph.

        Args:
            graph: ModelGraph to convert
            input_shape: Input shape (H, W, C) for Keras

        Returns:
            keras.Model instance

        Example:
            >>> model = adapter.build_model(graph, input_shape=(32, 32, 3))
        """
        if input_shape is None:
            input_shape = (32, 32, 3)

        # Create input
        inputs = keras.Input(shape=input_shape)

        # Build layers following graph topology
        layer_outputs = {}

        for node in graph.topological_sort():
            if node.operation == "input":
                layer_outputs[node.id] = inputs
                continue

            # Create layer
            layer = self._create_layer(node)

            # Get input
            if not node.predecessors:
                x = inputs
            else:
                # Get predecessor outputs
                pred_outputs = [layer_outputs[p.id] for p in node.predecessors]

                if len(pred_outputs) == 1:
                    x = pred_outputs[0]
                else:
                    # Concatenate along channel dimension
                    x = layers.Concatenate(axis=-1)(pred_outputs)

            # Apply layer
            if layer is not None:
                layer_outputs[node.id] = layer(x)
            else:
                layer_outputs[node.id] = x

        # Get output
        output_nodes = [n for n in graph.nodes.values() if not n.successors]
        if output_nodes:
            outputs = layer_outputs[output_nodes[0].id]
        else:
            # Use last node
            outputs = layer_outputs[list(layer_outputs.keys())[-1]]

        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)

        logger.info(f"Created Keras model with {len(model.layers)} layers")

        return model

    def _create_layer(self, node: GraphNode):
        """
        Create Keras layer from node.

        Args:
            node: GraphNode to convert

        Returns:
            Keras layer or None
        """
        op = node.operation
        params = node.params

        if op == "input":
            return None

        elif op == "conv2d":
            return layers.Conv2D(
                filters=params.get("filters", 64),
                kernel_size=params.get("kernel_size", 3),
                strides=params.get("stride", 1),
                padding=params.get("padding", "same"),
                activation=None,
            )

        elif op == "maxpool":
            return layers.MaxPooling2D(
                pool_size=params.get("pool_size", 2), strides=params.get("stride", None)
            )

        elif op == "avgpool":
            return layers.AveragePooling2D(
                pool_size=params.get("pool_size", 2), strides=params.get("stride", None)
            )

        elif op == "dense":
            return layers.Dense(units=params.get("units", 10), activation=None)

        elif op == "relu":
            return layers.ReLU()

        elif op == "sigmoid":
            return layers.Activation("sigmoid")

        elif op == "tanh":
            return layers.Activation("tanh")

        elif op == "softmax":
            return layers.Softmax()

        elif op == "batchnorm":
            return layers.BatchNormalization()

        elif op == "dropout":
            return layers.Dropout(rate=params.get("rate", 0.5))

        elif op == "flatten":
            return layers.Flatten()

        else:
            logger.warning(f"Unknown operation: {op}, using Lambda identity")
            return layers.Lambda(lambda x: x)

    def compile_model(
        self, model: "keras.Model", config: Optional[Dict[str, Any]] = None
    ) -> "keras.Model":
        """
        Compile Keras model with optimizer and loss.

        Args:
            model: Keras model
            config: Compilation configuration

        Returns:
            Compiled model

        Example:
            >>> model = adapter.compile_model(model, {
            ...     'optimizer': 'adam',
            ...     'learning_rate': 1e-3,
            ...     'loss': 'categorical_crossentropy'
            ... })
        """
        if config is None:
            config = {}

        # Optimizer
        optimizer_name = config.get("optimizer", "adam")
        lr = config.get("learning_rate", 1e-3)

        if optimizer_name == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=lr)
        elif optimizer_name == "sgd":
            optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=config.get("momentum", 0.9))
        elif optimizer_name == "rmsprop":
            optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        else:
            optimizer = optimizer_name

        # Loss
        loss = config.get("loss", "categorical_crossentropy")

        # Metrics
        metrics = config.get("metrics", ["accuracy"])

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        logger.info(f"Compiled model with {optimizer_name} optimizer")

        return model
