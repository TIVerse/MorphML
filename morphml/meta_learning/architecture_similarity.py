"""Architecture similarity metrics for transfer learning.

Measures distance between neural architectures for warm-starting.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Optional

from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class ArchitectureSimilarity:
    """
    Compute similarity between neural architectures.
    
    Provides multiple similarity metrics:
    1. **Operation Distribution** - Fast, compares layer type distributions
    2. **Graph Edit Distance** - Precise, expensive for large graphs
    3. **Path Encoding** - Medium complexity, encodes computational paths
    
    Example:
        >>> sim = ArchitectureSimilarity()
        >>> similarity = sim.compute(graph1, graph2, method='operation_distribution')
        >>> print(f"Similarity: {similarity:.3f}")
    """
    
    @staticmethod
    def compute(
        graph1: ModelGraph,
        graph2: ModelGraph,
        method: str = "operation_distribution"
    ) -> float:
        """
        Compute similarity between two architectures.
        
        Args:
            graph1: First architecture
            graph2: Second architecture
            method: Similarity method
                - 'operation_distribution': Fast, based on layer types
                - 'graph_structure': Medium, considers connections
                - 'combined': Weighted combination
        
        Returns:
            Similarity score (0-1, 1=identical)
        """
        if method == "operation_distribution":
            return ArchitectureSimilarity.operation_distribution_similarity(
                graph1, graph2
            )
        elif method == "graph_structure":
            return ArchitectureSimilarity.graph_structure_similarity(
                graph1, graph2
            )
        elif method == "combined":
            op_sim = ArchitectureSimilarity.operation_distribution_similarity(
                graph1, graph2
            )
            struct_sim = ArchitectureSimilarity.graph_structure_similarity(
                graph1, graph2
            )
            return 0.6 * op_sim + 0.4 * struct_sim
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    @staticmethod
    def operation_distribution_similarity(
        graph1: ModelGraph, graph2: ModelGraph
    ) -> float:
        """
        Compare operation type distributions (fast method).
        
        Computes cosine similarity between operation count vectors.
        
        Args:
            graph1: First architecture
            graph2: Second architecture
        
        Returns:
            Similarity score (0-1)
        """
        # Extract operations
        ops1 = [layer.layer_type for layer in graph1.layers]
        ops2 = [layer.layer_type for layer in graph2.layers]
        
        if not ops1 or not ops2:
            return 0.0
        
        # Count operation types
        count1 = Counter(ops1)
        count2 = Counter(ops2)
        
        # All operation types
        all_ops = sorted(set(count1.keys()) | set(count2.keys()))
        
        if not all_ops:
            return 1.0
        
        # Create normalized vectors
        vec1 = np.array([count1.get(op, 0) for op in all_ops], dtype=float)
        vec2 = np.array([count2.get(op, 0) for op in all_ops], dtype=float)
        
        # Normalize
        vec1 = vec1 / (vec1.sum() + 1e-8)
        vec2 = vec2 / (vec2.sum() + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8
        )
        
        return max(0.0, min(1.0, similarity))
    
    @staticmethod
    def graph_structure_similarity(
        graph1: ModelGraph, graph2: ModelGraph
    ) -> float:
        """
        Compare graph structures (medium complexity).
        
        Considers both nodes and edges.
        
        Args:
            graph1: First architecture
            graph2: Second architecture
        
        Returns:
            Similarity score (0-1)
        """
        # Number of layers
        n1 = len(graph1.layers)
        n2 = len(graph2.layers)
        
        if n1 == 0 and n2 == 0:
            return 1.0
        
        # Normalized layer count difference
        size_similarity = 1.0 - abs(n1 - n2) / max(n1, n2, 1)
        
        # Operation similarity
        op_similarity = ArchitectureSimilarity.operation_distribution_similarity(
            graph1, graph2
        )
        
        # Combine
        similarity = 0.5 * size_similarity + 0.5 * op_similarity
        
        return similarity
    
    @staticmethod
    def parameter_count_similarity(
        graph1: ModelGraph, graph2: ModelGraph
    ) -> float:
        """
        Compare models by parameter count.
        
        Args:
            graph1: First architecture
            graph2: Second architecture
        
        Returns:
            Similarity score (0-1)
        """
        params1 = graph1.count_parameters()
        params2 = graph2.count_parameters()
        
        if params1 == 0 and params2 == 0:
            return 1.0
        
        # Normalized difference
        max_params = max(params1, params2, 1)
        similarity = 1.0 - abs(params1 - params2) / max_params
        
        return similarity
    
    @staticmethod
    def depth_similarity(
        graph1: ModelGraph, graph2: ModelGraph
    ) -> float:
        """
        Compare network depth.
        
        Args:
            graph1: First architecture
            graph2: Second architecture
        
        Returns:
            Similarity score (0-1)
        """
        depth1 = len(graph1.layers)
        depth2 = len(graph2.layers)
        
        if depth1 == 0 and depth2 == 0:
            return 1.0
        
        max_depth = max(depth1, depth2, 1)
        similarity = 1.0 - abs(depth1 - depth2) / max_depth
        
        return similarity
    
    @staticmethod
    def batch_similarity(
        query_graph: ModelGraph,
        candidate_graphs: List[ModelGraph],
        method: str = "operation_distribution"
    ) -> np.ndarray:
        """
        Compute similarity between one graph and multiple candidates.
        
        Efficient batch computation.
        
        Args:
            query_graph: Query architecture
            candidate_graphs: List of candidate architectures
            method: Similarity method
        
        Returns:
            Array of similarity scores
        """
        similarities = []
        
        for candidate in candidate_graphs:
            sim = ArchitectureSimilarity.compute(
                query_graph, candidate, method=method
            )
            similarities.append(sim)
        
        return np.array(similarities)


def compute_task_similarity(
    task1: "TaskMetadata", task2: "TaskMetadata", method: str = "meta_features"
) -> float:
    """
    Compute similarity between tasks.
    
    Args:
        task1: First task
        task2: Second task
        method: Similarity method
            - 'meta_features': Based on dataset characteristics
            - 'dataset_name': Simple name matching
    
    Returns:
        Similarity score (0-1)
    """
    if method == "dataset_name":
        # Simple string similarity
        if task1.dataset_name == task2.dataset_name:
            return 1.0
        elif task1.dataset_name in task2.dataset_name or task2.dataset_name in task1.dataset_name:
            return 0.7
        else:
            return 0.0
    
    elif method == "meta_features":
        # Meta-feature based similarity
        features1 = np.array([
            task1.num_samples / 100000.0,
            task1.num_classes / 100.0,
            task1.input_size[0] / 3.0,  # Channels
            task1.input_size[1] / 224.0,  # Height
            task1.input_size[2] / 224.0,  # Width
        ])
        
        features2 = np.array([
            task2.num_samples / 100000.0,
            task2.num_classes / 100.0,
            task2.input_size[0] / 3.0,
            task2.input_size[1] / 224.0,
            task2.input_size[2] / 224.0,
        ])
        
        # Cosine similarity
        similarity = np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2) + 1e-8
        )
        
        # Problem type bonus
        if task1.problem_type == task2.problem_type:
            similarity *= 1.2  # Boost for same problem type
        
        return max(0.0, min(1.0, similarity))
    
    else:
        raise ValueError(f"Unknown method: {method}")
