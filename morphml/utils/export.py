"""Export architectures to framework-specific code.

Example:
    >>> from morphml.utils import ArchitectureExporter
    >>>
    >>> exporter = ArchitectureExporter()
    >>> pytorch_code = exporter.to_pytorch(graph)
    >>> print(pytorch_code)
"""

from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class ArchitectureExporter:
    """
    Export neural architectures to executable code.

    Generates framework-specific code (PyTorch, Keras) from
    ModelGraph representations.

    Example:
        >>> exporter = ArchitectureExporter()
        >>>
        >>> # PyTorch
        >>> pytorch_code = exporter.to_pytorch(graph)
        >>> with open('model.py', 'w') as f:
        ...     f.write(pytorch_code)
        >>>
        >>> # Keras
        >>> keras_code = exporter.to_keras(graph)
    """

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
                code.append(
                    f"        self.{layer_name} = nn.Conv2d("
                    f"in_channels=?, out_channels={filters}, "
                    f"kernel_size={kernel_size}, padding={padding_val})"
                )

            elif op == "dense":
                units = params.get("units", 128)
                code.append(
                    f"        self.{layer_name} = nn.Linear(in_features=?, out_features={units})"
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
                code.append(f"        self.{layer_name} = nn.BatchNorm2d(num_features=?)")

            elif op in ["relu", "sigmoid", "tanh", "softmax"]:
                # Functional, no layer definition needed
                pass

            elif op == "input":
                # No layer needed
                pass

            else:
                code.append(f"        # {layer_name}: {op} (custom)")

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
