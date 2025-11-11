"""Export architectures to framework-specific code.

Example:
    >>> from morphml.utils import ArchitectureExporter
    >>>
    >>> exporter = ArchitectureExporter()
    >>> pytorch_code = exporter.to_pytorch(graph)
    >>> print(pytorch_code)
"""

import numpy as np

from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class ArchitectureExporter:
    """
    Export neural architectures to executable code.

    Generates framework-specific code (PyTorch, Keras) from
    ModelGraph representations. Supports custom layer handlers.

    Example:
        >>> exporter = ArchitectureExporter()
        >>>
        >>> # Add custom layer handler
        >>> def custom_handler(node, shapes):
        ...     return f"nn.CustomLayer({node.params})"
        >>> exporter.add_custom_layer_handler("custom_op", custom_handler)
        >>>
        >>> # PyTorch
        >>> pytorch_code = exporter.to_pytorch(graph)
        >>> with open('model.py', 'w') as f:
        ...     f.write(pytorch_code)
        >>>
        >>> # Keras
        >>> keras_code = exporter.to_keras(graph)
    """
    
    def __init__(self):
        """Initialize exporter with custom layer handlers."""
        self.custom_pytorch_handlers = {}
        self.custom_keras_handlers = {}
        logger.debug("Initialized ArchitectureExporter")
    
    def add_custom_layer_handler(
        self,
        operation_name: str,
        pytorch_handler=None,
        keras_handler=None,
    ):
        """
        Add custom handler for a layer type.
        
        Args:
            operation_name: Name of the operation (e.g., "custom_conv")
            pytorch_handler: Function(node, shapes) -> str for PyTorch code
            keras_handler: Function(node, shapes) -> str for Keras code
            
        Example:
            >>> def my_pytorch_handler(node, shapes):
            ...     return f"nn.MyCustomLayer({node.params['size']})"
            >>> exporter.add_custom_layer_handler("my_op", pytorch_handler=my_pytorch_handler)
        """
        if pytorch_handler:
            self.custom_pytorch_handlers[operation_name] = pytorch_handler
            logger.info(f"Added custom PyTorch handler for '{operation_name}'")
        
        if keras_handler:
            self.custom_keras_handlers[operation_name] = keras_handler
            logger.info(f"Added custom Keras handler for '{operation_name}'")
    
    def remove_custom_layer_handler(self, operation_name: str):
        """Remove custom handler for a layer type."""
        self.custom_pytorch_handlers.pop(operation_name, None)
        self.custom_keras_handlers.pop(operation_name, None)
        logger.info(f"Removed custom handlers for '{operation_name}'")

    def _infer_shape(self, node, shapes):
        """Infer shape of node based on predecessors."""
        if node.operation == "input":
            return node.get_param("shape", (32, 32, 3))

        pred_node = list(node.predecessors)[0]
        pred_shape = shapes.get(pred_node.id)

        if pred_shape is None:
            return None

        op = node.operation
        params = node.params

        if op == "conv2d":
            filters = params.get("filters", 64)
            kernel_size = params.get("kernel_size", 3)
            padding = params.get("padding", "same")
            padding_val = kernel_size // 2 if padding == "same" else 0

            if len(pred_shape) == 3:
                return (filters, pred_shape[1], pred_shape[2])
            elif len(pred_shape) == 1:
                return (filters, pred_shape[0])
            else:
                return None

        elif op == "dense":
            units = params.get("units", 128)
            return (units,)

        elif op == "maxpool":
            pool_size = params.get("pool_size", 2)
            if len(pred_shape) == 3:
                return (pred_shape[0], pred_shape[1] // pool_size, pred_shape[2] // pool_size)
            elif len(pred_shape) == 1:
                return (pred_shape[0],)
            else:
                return None

        elif op == "avgpool":
            pool_size = params.get("pool_size", 2)
            if len(pred_shape) == 3:
                return (pred_shape[0], pred_shape[1] // pool_size, pred_shape[2] // pool_size)
            elif len(pred_shape) == 1:
                return (pred_shape[0],)
            else:
                return None

        elif op == "flatten":
            return (int(np.prod(pred_shape)),)

        elif op in ["relu", "sigmoid", "tanh", "softmax"]:
            return pred_shape

        elif op == "dropout":
            return pred_shape

        elif op == "batchnorm":
            return pred_shape

        else:
            return None

    def to_pytorch(self, graph: ModelGraph, class_name: str = "GeneratedModel") -> str:
        """
        Generate PyTorch code.

        Args:
            graph: Architecture to export
            class_name: Name for generated class

        Returns:
            PyTorch code as string

        Example:
            >>> code = exporter.to_pytorch(graph, 'MyModel')
            >>> exec(code)  # Defines MyModel class
        """
        code = []
        code.append("import torch")
        code.append("import torch.nn as nn")
        code.append("import torch.nn.functional as F")
        code.append("")
        code.append("")
        code.append(f"class {class_name}(nn.Module):")
        code.append("    def __init__(self):")
        code.append("        super().__init__()")
        code.append("")

        # Get topological order
        try:
            sorted_nodes = graph.topological_sort()
        except Exception as e:
            logger.error(f"Failed to sort graph: {e}")
            sorted_nodes = list(graph.nodes.values())

        shapes = {}
        for node in sorted_nodes:
            shapes[node.id] = self._infer_shape(node, shapes)

        # Generate layers
        layer_names = {}
        for i, node in enumerate(sorted_nodes):
            layer_name = f"layer_{i}"
            layer_names[node.id] = layer_name

            op = node.operation
            params = node.params

            if op == "conv2d":
                filters = params.get("filters", 64)
                kernel_size = params.get("kernel_size", 3)
                padding = params.get("padding", "same")
                padding_val = kernel_size // 2 if padding == "same" else 0
                
                # Try to infer input channels
                in_channels = "?"
                if node.predecessors:
                    pred_node = list(node.predecessors)[0]
                    pred_shape = shapes.get(pred_node.id)
                    if pred_shape and len(pred_shape) >= 3:
                        in_channels = pred_shape[0]
                
                code.append(
                    f"        self.{layer_name} = nn.Conv2d("
                    f"in_channels={in_channels}, out_channels={filters}, "
                    f"kernel_size={kernel_size}, padding={padding_val})"
                )

            elif op == "dense":
                units = params.get("units", 128)
                
                # Try to infer input features
                in_features = "?"
                if node.predecessors:
                    pred_node = list(node.predecessors)[0]
                    pred_shape = shapes.get(pred_node.id)
                    if pred_shape:
                        if len(pred_shape) == 1:
                            in_features = pred_shape[0]
                        else:
                            # Need flattening
                            in_features = int(np.prod(pred_shape))
                
                code.append(
                    f"        self.{layer_name} = nn.Linear(in_features={in_features}, out_features={units})"
                )

            elif op == "maxpool":
                pool_size = params.get("pool_size", 2)
                code.append(f"        self.{layer_name} = nn.MaxPool2d(kernel_size={pool_size})")

            elif op == "avgpool":
                pool_size = params.get("pool_size", 2)
                code.append(f"        self.{layer_name} = nn.AvgPool2d(kernel_size={pool_size})")

            elif op == "dropout":
                rate = params.get("rate", 0.5)
                code.append(f"        self.{layer_name} = nn.Dropout(p={rate})")

            elif op == "batchnorm":
                # Try to infer number of features
                num_features = "?"
                if node.predecessors:
                    pred_node = list(node.predecessors)[0]
                    pred_shape = shapes.get(pred_node.id)
                    if pred_shape and len(pred_shape) >= 1:
                        num_features = pred_shape[0]  # First dimension is channels
                
                code.append(f"        self.{layer_name} = nn.BatchNorm2d(num_features={num_features})")

            elif op == "flatten":
                code.append(f"        self.{layer_name} = nn.Flatten()")

            elif op in ["relu", "sigmoid", "tanh", "softmax"]:
                # Functional, no layer definition needed
                pass

            elif op == "input":
                # No layer needed
                pass

            else:
                # Check for custom handler
                if op in self.custom_pytorch_handlers:
                    handler = self.custom_pytorch_handlers[op]
                    custom_code = handler(node, shapes)
                    code.append(f"        self.{layer_name} = {custom_code}")
                else:
                    code.append(f"        # {layer_name}: {op} (custom - no handler defined)")

        code.append("")
        code.append("    def forward(self, x):")

        # Generate forward pass
        for i, node in enumerate(sorted_nodes):
            layer_name = layer_names[node.id]
            op = node.operation

            if op == "input":
                code.append("        # x = input")
            elif op == "relu":
                code.append("        x = F.relu(x)")
            elif op == "sigmoid":
                code.append("        x = torch.sigmoid(x)")
            elif op == "tanh":
                code.append("        x = torch.tanh(x)")
            elif op == "softmax":
                code.append("        x = F.softmax(x, dim=1)")
            else:
                code.append(f"        x = self.{layer_name}(x)")

        code.append("        return x")
        code.append("")
        code.append("")
        code.append("# Usage:")
        code.append(f"# model = {class_name}()")
        code.append("# output = model(input_tensor)")

        return "\n".join(code)

    def to_keras(self, graph: ModelGraph, model_name: str = "generated_model") -> str:
        """
        Generate Keras/TensorFlow code.

        Args:
            graph: Architecture to export
            model_name: Name for generated model

        Returns:
            Keras code as string
        """
        code = []
        code.append("import tensorflow as tf")
        code.append("from tensorflow import keras")
        code.append("from tensorflow.keras import layers")
        code.append("")
        code.append("")
        code.append(f"def {model_name}():")
        code.append('    """Generated Keras model."""')

        # Get topological order
        try:
            sorted_nodes = graph.topological_sort()
        except Exception:
            sorted_nodes = list(graph.nodes.values())

        # Find input shape
        input_shape = None
        for node in sorted_nodes:
            if node.operation == "input":
                input_shape = node.get_param("shape", (32, 32, 3))
                break

        if input_shape is None:
            input_shape = (32, 32, 3)

        code.append(f"    inputs = keras.Input(shape={input_shape})")
        code.append("    x = inputs")
        code.append("")

        # Generate layers
        for node in sorted_nodes:
            op = node.operation
            params = node.params

            if op == "input":
                continue
            elif op == "conv2d":
                filters = params.get("filters", 64)
                kernel_size = params.get("kernel_size", 3)
                padding = params.get("padding", "same")
                code.append(
                    f"    x = layers.Conv2D({filters}, {kernel_size}, " f"padding='{padding}')(x)"
                )
            elif op == "dense":
                units = params.get("units", 128)
                code.append(f"    x = layers.Dense({units})(x)")
            elif op == "maxpool":
                pool_size = params.get("pool_size", 2)
                code.append(f"    x = layers.MaxPooling2D({pool_size})(x)")
            elif op == "avgpool":
                pool_size = params.get("pool_size", 2)
                code.append(f"    x = layers.AveragePooling2D({pool_size})(x)")
            elif op == "dropout":
                rate = params.get("rate", 0.5)
                code.append(f"    x = layers.Dropout({rate})(x)")
            elif op == "batchnorm":
                code.append("    x = layers.BatchNormalization()(x)")
            elif op == "flatten":
                code.append("    x = layers.Flatten()(x)")
            elif op == "relu":
                code.append("    x = layers.Activation('relu')(x)")
            elif op == "sigmoid":
                code.append("    x = layers.Activation('sigmoid')(x)")
            elif op == "tanh":
                code.append("    x = layers.Activation('tanh')(x)")
            elif op == "softmax":
                code.append("    x = layers.Activation('softmax')(x)")

        code.append("")
        code.append("    model = keras.Model(inputs=inputs, outputs=x)")
        code.append("    return model")
        code.append("")
        code.append("")
        code.append("# Usage:")
        code.append(f"# model = {model_name}()")
        code.append("# model.compile(optimizer='adam', loss='categorical_crossentropy')")
        code.append("# model.fit(x_train, y_train, epochs=10)")

        return "\n".join(code)

    def to_json(self, graph: ModelGraph) -> str:
        """
        Export architecture as JSON.

        Args:
            graph: Architecture to export

        Returns:
            JSON string
        """
        return graph.to_json()
