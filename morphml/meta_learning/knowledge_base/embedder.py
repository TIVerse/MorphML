"""Architecture embedding for vector search.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import List

import numpy as np

from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class ArchitectureEmbedder:
    """
    Embed neural architectures as fixed-size vectors.

    Methods:
    - Simple: Feature-based embedding (no external deps)
    - GNN: Graph neural network embedding (requires PyTorch)

    Args:
        method: Embedding method ('simple' or 'gnn')
        embedding_dim: Dimension of output vectors

    Example:
        >>> embedder = ArchitectureEmbedder(method='simple', embedding_dim=128)
        >>> embedding = embedder.embed(graph)
        >>> print(embedding.shape)  # (128,)
    """

    def __init__(self, method: str = "simple", embedding_dim: int = 128):
        """Initialize embedder."""
        self.method = method
        self.embedding_dim = embedding_dim

        logger.info(f"Initialized ArchitectureEmbedder (method={method}, dim={embedding_dim})")

    def embed(self, graph: ModelGraph) -> np.ndarray:
        """
        Embed architecture as vector.

        Args:
            graph: Architecture graph

        Returns:
            Embedding vector of shape [embedding_dim]
        """
        if self.method == "simple":
            return self._embed_simple(graph)
        elif self.method == "gnn":
            return self._embed_gnn(graph)
        else:
            raise ValueError(f"Unknown embedding method: {self.method}")

    def _embed_simple(self, graph: ModelGraph) -> np.ndarray:
        """
        Simple feature-based embedding.

        Uses architectural statistics and operation counts.
        """
        features = []

        # Operation type counts (one-hot style)
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

        ops = [layer.layer_type for layer in graph.layers]
        for op_type in op_types:
            count = ops.count(op_type) / max(len(ops), 1)  # Normalize
            features.append(count)

        # Graph structure features
        features.append(len(graph.layers) / 100.0)  # Normalize depth
        features.append(graph.count_parameters() / 1000000.0)  # Params in millions

        # Layer-specific features
        conv_filters = []
        dense_units = []

        for layer in graph.layers:
            if layer.layer_type == "conv2d":
                filters = layer.config.get("filters", 64)
                conv_filters.append(filters)
            elif layer.layer_type == "dense":
                units = layer.config.get("units", 128)
                dense_units.append(units)

        # Statistics of layer sizes
        if conv_filters:
            features.append(np.mean(conv_filters) / 512.0)
            features.append(np.max(conv_filters) / 1024.0)
        else:
            features.extend([0.0, 0.0])

        if dense_units:
            features.append(np.mean(dense_units) / 1024.0)
            features.append(np.max(dense_units) / 2048.0)
        else:
            features.extend([0.0, 0.0])

        # Convert to numpy array
        features = np.array(features, dtype=np.float32)

        # Pad or truncate to embedding_dim
        if len(features) < self.embedding_dim:
            # Pad with zeros
            padding = np.zeros(self.embedding_dim - len(features), dtype=np.float32)
            features = np.concatenate([features, padding])
        else:
            # Truncate
            features = features[: self.embedding_dim]

        # Normalize to unit length
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features

    def _embed_gnn(self, graph: ModelGraph) -> np.ndarray:
        """
        GNN-based embedding (requires PyTorch).

        Falls back to simple if PyTorch not available.
        """
        try:
            # TODO: Implement GNN encoder
            logger.warning("GNN embedding not implemented, using simple")
            return self._embed_simple(graph)
        except ImportError:
            logger.warning("PyTorch not available, using simple embedding")
            return self._embed_simple(graph)

    def batch_embed(self, graphs: List[ModelGraph]) -> np.ndarray:
        """
        Embed multiple graphs.

        Args:
            graphs: List of architectures

        Returns:
            Array of embeddings, shape [num_graphs, embedding_dim]
        """
        embeddings = [self.embed(g) for g in graphs]
        return np.array(embeddings)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0-1, 1=identical)
        """
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8
        )

        return max(0.0, min(1.0, similarity))
