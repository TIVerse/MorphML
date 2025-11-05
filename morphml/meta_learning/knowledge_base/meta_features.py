"""Meta-feature extraction for tasks and architectures.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from collections import Counter
from typing import Dict

import numpy as np

from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger
from morphml.meta_learning.experiment_database import TaskMetadata

logger = get_logger(__name__)


class MetaFeatureExtractor:
    """
    Extract meta-features from tasks and architectures.

    Task meta-features:
    - Dataset statistics
    - Class distribution
    - Input dimensionality

    Architecture meta-features:
    - Structural properties
    - Operation statistics
    - Connectivity patterns
    - Parameter counts

    Example:
        >>> extractor = MetaFeatureExtractor()
        >>> task_features = extractor.extract_task_features(task_metadata)
        >>> arch_features = extractor.extract_architecture_features(graph)
    """

    def __init__(self):
        """Initialize extractor."""
        logger.info("Initialized MetaFeatureExtractor")

    def extract_task_features(self, task: TaskMetadata, normalize: bool = True) -> Dict[str, float]:
        """
        Extract meta-features from task.

        Args:
            task: Task metadata
            normalize: Whether to normalize features

        Returns:
            Dictionary of meta-features
        """
        features = {}

        # Dataset size features
        features["num_samples"] = float(task.num_samples)
        features["num_classes"] = float(task.num_classes)

        # Input dimensionality
        if isinstance(task.input_size, (tuple, list)):
            features["input_channels"] = float(task.input_size[0])
            features["input_height"] = float(task.input_size[1])
            features["input_width"] = float(
                task.input_size[2] if len(task.input_size) > 2 else task.input_size[1]
            )
            features["input_dim"] = float(np.prod(task.input_size))
        else:
            features["input_dim"] = float(task.input_size)

        # Problem type encoding
        problem_types = ["classification", "detection", "segmentation", "regression"]
        for pt in problem_types:
            features[f"problem_{pt}"] = 1.0 if task.problem_type == pt else 0.0

        # Derived features
        if task.num_samples > 0:
            features["samples_per_class"] = task.num_samples / max(task.num_classes, 1)

        # Normalize if requested
        if normalize:
            features["num_samples"] /= 1000000.0  # Scale to millions
            features["num_classes"] /= 1000.0  # Scale to thousands
            features["input_dim"] /= 10000.0  # Scale

        return features

    def extract_architecture_features(
        self, graph: ModelGraph, normalize: bool = True
    ) -> Dict[str, float]:
        """
        Extract meta-features from architecture.

        Args:
            graph: Architecture graph
            normalize: Whether to normalize features

        Returns:
            Dictionary of meta-features
        """
        features = {}

        # Basic structure
        features["num_layers"] = float(len(graph.layers))
        features["num_parameters"] = float(graph.count_parameters())

        # Operation type counts
        op_counts = Counter(layer.layer_type for layer in graph.layers)

        op_types = [
            "conv2d",
            "maxpool2d",
            "avgpool2d",
            "dense",
            "relu",
            "tanh",
            "sigmoid",
            "batchnorm",
            "dropout",
            "flatten",
            "input",
            "output",
        ]

        for op_type in op_types:
            features[f"num_{op_type}"] = float(op_counts.get(op_type, 0))

        # Operation diversity
        features["num_unique_ops"] = float(len(op_counts))
        features["operation_diversity"] = features["num_unique_ops"] / max(
            features["num_layers"], 1
        )

        # Layer configuration statistics
        conv_filters = []
        conv_kernels = []
        dense_units = []
        dropout_rates = []

        for layer in graph.layers:
            if layer.layer_type == "conv2d":
                conv_filters.append(layer.config.get("filters", 64))
                conv_kernels.append(layer.config.get("kernel_size", 3))
            elif layer.layer_type == "dense":
                dense_units.append(layer.config.get("units", 128))
            elif layer.layer_type == "dropout":
                dropout_rates.append(layer.config.get("rate", 0.5))

        # Convolutional layer stats
        if conv_filters:
            features["conv_avg_filters"] = float(np.mean(conv_filters))
            features["conv_max_filters"] = float(np.max(conv_filters))
            features["conv_min_filters"] = float(np.min(conv_filters))
            features["conv_std_filters"] = float(np.std(conv_filters))
        else:
            features["conv_avg_filters"] = 0.0
            features["conv_max_filters"] = 0.0
            features["conv_min_filters"] = 0.0
            features["conv_std_filters"] = 0.0

        if conv_kernels:
            features["conv_avg_kernel"] = float(np.mean(conv_kernels))
        else:
            features["conv_avg_kernel"] = 0.0

        # Dense layer stats
        if dense_units:
            features["dense_avg_units"] = float(np.mean(dense_units))
            features["dense_max_units"] = float(np.max(dense_units))
        else:
            features["dense_avg_units"] = 0.0
            features["dense_max_units"] = 0.0

        # Dropout stats
        if dropout_rates:
            features["avg_dropout_rate"] = float(np.mean(dropout_rates))
        else:
            features["avg_dropout_rate"] = 0.0

        # Depth and width ratios
        if features["num_layers"] > 0:
            features["params_per_layer"] = features["num_parameters"] / features["num_layers"]
        else:
            features["params_per_layer"] = 0.0

        # Normalize if requested
        if normalize:
            features["num_layers"] /= 100.0
            features["num_parameters"] /= 10000000.0  # Scale to 10M
            features["conv_avg_filters"] /= 1024.0
            features["conv_max_filters"] /= 2048.0
            features["dense_avg_units"] /= 4096.0
            features["dense_max_units"] /= 8192.0

        return features

    def extract_combined_features(self, task: TaskMetadata, graph: ModelGraph) -> Dict[str, float]:
        """
        Extract combined task and architecture features.

        Args:
            task: Task metadata
            graph: Architecture graph

        Returns:
            Combined feature dictionary
        """
        task_features = self.extract_task_features(task)
        arch_features = self.extract_architecture_features(graph)

        # Combine
        combined = {**task_features, **arch_features}

        # Add interaction features
        if "num_classes" in task_features and "num_layers" in arch_features:
            combined["layers_per_class"] = arch_features["num_layers"] / max(
                task_features["num_classes"], 1
            )

        return combined

    def feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dict to numpy array.

        Args:
            features: Feature dictionary

        Returns:
            Feature vector
        """
        # Sort keys for consistency
        keys = sorted(features.keys())
        values = [features[k] for k in keys]

        return np.array(values, dtype=np.float32)

    def compute_feature_similarity(
        self, features1: Dict[str, float], features2: Dict[str, float]
    ) -> float:
        """
        Compute cosine similarity between feature vectors.

        Args:
            features1: First feature dict
            features2: Second feature dict

        Returns:
            Similarity score (0-1)
        """
        # Get common keys
        common_keys = sorted(set(features1.keys()) & set(features2.keys()))

        if not common_keys:
            return 0.0

        # Create vectors
        vec1 = np.array([features1[k] for k in common_keys])
        vec2 = np.array([features2[k] for k in common_keys])

        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

        return max(0.0, min(1.0, similarity))
