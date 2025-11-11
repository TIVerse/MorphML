"""Layer builders for the MorphML DSL.

This module provides a Pythonic interface for defining neural network layers
in search spaces using a builder pattern.

Example:
    >>> from morphml.core.dsl import Layer
    >>>
    >>> # Define a conv2d layer with multiple filter options
    >>> conv = Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5])
    >>>
    >>> # Define a dense layer
    >>> dense = Layer.dense(units=[128, 256, 512])
"""

from typing import Any, Dict, List, Optional, Union

from morphml.core.graph import GraphNode


class LayerSpec:
    """
    Specification for a layer in the search space.

    A LayerSpec defines a layer type and its parameter ranges.
    During search, specific parameter values are sampled from these ranges.

    Attributes:
        operation: Layer operation type (conv2d, dense, etc.)
        param_ranges: Dictionary of parameter names to possible values
        metadata: Additional metadata

    Example:
        >>> spec = LayerSpec("conv2d", {
        ...     "filters": [32, 64, 128],
        ...     "kernel_size": [3, 5, 7]
        ... })
    """

    def __init__(
        self,
        operation: str,
        param_ranges: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize layer specification.

        Args:
            operation: Layer operation type
            param_ranges: Parameter ranges (param_name -> list of values or single value)
            metadata: Additional metadata
        """
        self.operation = operation
        self.param_ranges = param_ranges or {}
        self.metadata = metadata or {}

    def sample(self) -> GraphNode:
        """
        Sample a concrete layer from this specification.

        Returns:
            GraphNode with sampled parameters
        """
        import random

        # Sample one value from each parameter range
        params = {}
        for param_name, values in self.param_ranges.items():
            if isinstance(values, list) and values:
                params[param_name] = random.choice(values)
            else:
                params[param_name] = values

        return GraphNode.create(self.operation, params=params, metadata=self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "operation": self.operation,
            "param_ranges": self.param_ranges,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayerSpec":
        """Deserialize from dictionary."""
        return cls(
            operation=data["operation"],
            param_ranges=data.get("param_ranges", {}),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"LayerSpec(operation={self.operation}, params={list(self.param_ranges.keys())})"


class Layer:
    """
    Builder for defining layers in the search space.

    Provides static methods for creating layer specifications with a
    Pythonic API.

    Example:
        >>> # Convolutional layers
        >>> Layer.conv2d(filters=[32, 64], kernel_size=3)
        >>> Layer.conv2d(filters=64, kernel_size=[3, 5, 7])
        >>>
        >>> # Pooling layers
        >>> Layer.maxpool(pool_size=2)
        >>> Layer.avgpool(pool_size=[2, 3])
        >>>
        >>> # Dense layers
        >>> Layer.dense(units=[128, 256, 512])
        >>>
        >>> # Activation layers
        >>> Layer.relu()
        >>> Layer.sigmoid()
    """

    @staticmethod
    def conv2d(
        filters: Union[int, List[int]],
        kernel_size: Union[int, List[int]] = 3,
        strides: Union[int, List[int]] = 1,
        padding: str = "same",
        activation: Optional[str] = None,
        **kwargs: Any,
    ) -> LayerSpec:
        """
        Define a 2D convolutional layer.

        Args:
            filters: Number of filters (can be list for search)
            kernel_size: Kernel size (can be list for search)
            strides: Stride size
            padding: Padding mode ('same' or 'valid')
            activation: Optional activation function
            **kwargs: Additional parameters

        Returns:
            LayerSpec for conv2d layer

        Example:
            >>> Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5])
        """
        param_ranges = {
            "filters": filters if isinstance(filters, list) else [filters],
            "kernel_size": kernel_size if isinstance(kernel_size, list) else [kernel_size],
            "strides": strides if isinstance(strides, list) else [strides],
            "padding": [padding],
        }

        if activation:
            param_ranges["activation"] = [activation]

        param_ranges.update(kwargs)

        return LayerSpec("conv2d", param_ranges)

    @staticmethod
    def dense(
        units: Union[int, List[int]],
        activation: Optional[str] = None,
        use_bias: bool = True,
        **kwargs: Any,
    ) -> LayerSpec:
        """
        Define a fully-connected (dense) layer.

        Args:
            units: Number of units (can be list for search)
            activation: Optional activation function
            use_bias: Whether to use bias
            **kwargs: Additional parameters

        Returns:
            LayerSpec for dense layer

        Example:
            >>> Layer.dense(units=[128, 256, 512])
        """
        param_ranges = {
            "units": units if isinstance(units, list) else [units],
            "use_bias": [use_bias],
        }

        if activation:
            param_ranges["activation"] = [activation]

        param_ranges.update(kwargs)

        return LayerSpec("dense", param_ranges)

    @staticmethod
    def maxpool(
        pool_size: Union[int, List[int]] = 2,
        strides: Optional[Union[int, List[int]]] = None,
        padding: str = "valid",
        **kwargs: Any,
    ) -> LayerSpec:
        """
        Define a max pooling layer.

        Args:
            pool_size: Pool size (can be list for search)
            strides: Stride size (defaults to pool_size)
            padding: Padding mode
            **kwargs: Additional parameters

        Returns:
            LayerSpec for maxpool layer
        """
        param_ranges = {
            "pool_size": pool_size if isinstance(pool_size, list) else [pool_size],
            "padding": [padding],
        }

        if strides is not None:
            param_ranges["strides"] = strides if isinstance(strides, list) else [strides]

        param_ranges.update(kwargs)

        return LayerSpec("maxpool", param_ranges)

    @staticmethod
    def avgpool(
        pool_size: Union[int, List[int]] = 2,
        strides: Optional[Union[int, List[int]]] = None,
        padding: str = "valid",
        **kwargs: Any,
    ) -> LayerSpec:
        """Define an average pooling layer."""
        param_ranges = {
            "pool_size": pool_size if isinstance(pool_size, list) else [pool_size],
            "padding": [padding],
        }

        if strides is not None:
            param_ranges["strides"] = strides if isinstance(strides, list) else [strides]

        param_ranges.update(kwargs)

        return LayerSpec("avgpool", param_ranges)

    @staticmethod
    def dropout(rate: Union[float, List[float]] = 0.5, **kwargs: Any) -> LayerSpec:
        """
        Define a dropout layer.

        Args:
            rate: Dropout rate (can be list for search)
            **kwargs: Additional parameters

        Returns:
            LayerSpec for dropout layer
        """
        param_ranges = {
            "rate": rate if isinstance(rate, list) else [rate],
        }
        param_ranges.update(kwargs)

        return LayerSpec("dropout", param_ranges)

    @staticmethod
    def batchnorm(**kwargs: Any) -> LayerSpec:
        """Define a batch normalization layer."""
        return LayerSpec("batchnorm", param_ranges=kwargs)

    @staticmethod
    def flatten(**kwargs: Any) -> LayerSpec:
        """
        Define a flatten layer.

        Flattens the input tensor to 1D (excluding batch dimension).
        Commonly used between convolutional and dense layers.

        Args:
            **kwargs: Additional parameters

        Returns:
            LayerSpec for flatten layer
        """
        return LayerSpec("flatten", param_ranges=kwargs)

    @staticmethod
    def relu(**kwargs: Any) -> LayerSpec:
        """Define a ReLU activation layer."""
        return LayerSpec("relu", param_ranges=kwargs)

    @staticmethod
    def sigmoid(**kwargs: Any) -> LayerSpec:
        """Define a sigmoid activation layer."""
        return LayerSpec("sigmoid", param_ranges=kwargs)

    @staticmethod
    def tanh(**kwargs: Any) -> LayerSpec:
        """Define a tanh activation layer."""
        return LayerSpec("tanh", param_ranges=kwargs)

    @staticmethod
    def softmax(**kwargs: Any) -> LayerSpec:
        """Define a softmax activation layer."""
        return LayerSpec("softmax", param_ranges=kwargs)

    @staticmethod
    def input(shape: tuple, **kwargs: Any) -> LayerSpec:
        """
        Define an input layer.

        Args:
            shape: Input shape (excluding batch dimension)
            **kwargs: Additional parameters

        Returns:
            LayerSpec for input layer
        """
        param_ranges = {"shape": [shape]}
        param_ranges.update(kwargs)

        return LayerSpec("input", param_ranges)

    @staticmethod
    def output(units: int, activation: str = "softmax", **kwargs: Any) -> LayerSpec:
        """
        Define an output layer.

        Args:
            units: Number of output units (classes)
            activation: Activation function
            **kwargs: Additional parameters

        Returns:
            LayerSpec for output layer
        """
        param_ranges = {
            "units": [units],
            "activation": [activation],
        }
        param_ranges.update(kwargs)

        return LayerSpec("dense", param_ranges, metadata={"is_output": True})

    @staticmethod
    def custom(
        operation: str,
        param_ranges: Optional[Dict[str, List[Any]]] = None,
        **kwargs: Any,
    ) -> LayerSpec:
        """
        Define a custom layer.

        Args:
            operation: Operation type
            param_ranges: Parameter ranges
            **kwargs: Additional parameter ranges

        Returns:
            LayerSpec for custom layer

        Example:
            >>> Layer.custom("my_op", {"param1": [1, 2, 3]})
        """
        ranges = param_ranges or {}
        ranges.update(kwargs)
        return LayerSpec(operation, ranges)
