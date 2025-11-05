"""Architecture encoding/decoding utilities for continuous optimizers.

This module provides methods to convert between discrete graph architectures
and continuous vector representations, enabling the use of continuous optimizers
like CMA-ES and PSO for neural architecture search.

Encoding Strategies:
- Positional encoding of operations
- Hyperparameter normalization
- Fixed-length vector representation
- Padding/truncation for variable architectures

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from morphml.core.dsl import SearchSpace
from morphml.core.graph import GraphNode, ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


# Standard operation vocabulary for encoding
OPERATION_VOCABULARY = [
    'input',
    'output',
    'conv2d',
    'dense',
    'relu',
    'sigmoid',
    'tanh',
    'maxpool',
    'avgpool',
    'batchnorm',
    'dropout',
    'flatten',
    'add',
    'concat',
    'identity',
]


class ArchitectureEncoder:
    """
    Encoder/decoder for converting between graphs and continuous vectors.
    
    Provides bidirectional mapping between ModelGraph (discrete) and
    numpy arrays (continuous) for use with continuous optimizers.
    
    Encoding Scheme:
        For each node position (up to max_nodes):
        - operation_id: [0, 1] normalized index
        - param1: [0, 1] normalized hyperparameter
        - param2: [0, 1] normalized hyperparameter
        
    Total dimensions: max_nodes * 3
    
    Example:
        >>> encoder = ArchitectureEncoder(search_space, max_nodes=20)
        >>> vector = encoder.encode(graph)  # ModelGraph -> np.ndarray
        >>> decoded_graph = encoder.decode(vector)  # np.ndarray -> ModelGraph
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        max_nodes: int = 20,
        operation_vocab: Optional[List[str]] = None
    ):
        """
        Initialize encoder.
        
        Args:
            search_space: SearchSpace for sampling/validation
            max_nodes: Maximum number of nodes to encode
            operation_vocab: List of operation names (None = use default)
        """
        self.search_space = search_space
        self.max_nodes = max_nodes
        self.operation_vocab = operation_vocab or OPERATION_VOCABULARY
        self.vocab_size = len(self.operation_vocab)
        
        # Encoding dimension: 3 features per node
        self.dim = max_nodes * 3
        
        logger.debug(
            f"Initialized ArchitectureEncoder: "
            f"max_nodes={max_nodes}, dim={self.dim}, vocab_size={self.vocab_size}"
        )
    
    def encode(self, graph: ModelGraph) -> np.ndarray:
        """
        Encode ModelGraph as continuous vector.
        
        Args:
            graph: ModelGraph to encode
            
        Returns:
            Continuous vector of shape (dim,)
            
        Example:
            >>> vector = encoder.encode(graph)
            >>> print(vector.shape)  # (60,) for max_nodes=20
        """
        vector = np.zeros(self.dim)
        
        try:
            # Get topological order
            nodes = list(graph.topological_sort())
        except Exception:
            # Fallback: use nodes in any order
            nodes = list(graph.nodes.values())
        
        # Encode each node
        for i, node in enumerate(nodes[:self.max_nodes]):
            base_idx = i * 3
            
            # Feature 1: Operation ID (normalized)
            op_id = self._encode_operation(node.operation)
            vector[base_idx] = op_id / self.vocab_size
            
            # Features 2-3: Hyperparameters (normalized)
            param1, param2 = self._encode_parameters(node)
            vector[base_idx + 1] = param1
            vector[base_idx + 2] = param2
        
        # Padding is already zeros for nodes beyond graph length
        
        return vector
    
    def decode(self, vector: np.ndarray) -> ModelGraph:
        """
        Decode continuous vector to ModelGraph.
        
        Args:
            vector: Continuous vector of shape (dim,)
            
        Returns:
            Decoded ModelGraph
            
        Note:
            Decoding is approximate - the inverse mapping is not unique.
            We sample a base architecture and modify it based on the vector.
            
        Example:
            >>> graph = encoder.decode(vector)
            >>> assert graph.is_valid_dag()
        """
        # Start with a random architecture from search space
        graph = self.search_space.sample()
        
        try:
            nodes = list(graph.topological_sort())
        except Exception:
            nodes = list(graph.nodes.values())
        
        # Modify nodes based on vector
        for i in range(min(len(nodes), self.max_nodes)):
            base_idx = i * 3
            
            if base_idx + 2 >= len(vector):
                break
            
            # Decode operation
            op_id_norm = vector[base_idx]
            op_id = int(op_id_norm * self.vocab_size) % self.vocab_size
            operation = self.operation_vocab[op_id]
            
            # Decode parameters
            param1_norm = np.clip(vector[base_idx + 1], 0, 1)
            param2_norm = np.clip(vector[base_idx + 2], 0, 1)
            
            # Apply to node (if not input/output)
            if i < len(nodes) and nodes[i].operation not in ['input', 'output']:
                nodes[i].operation = operation
                self._decode_parameters(nodes[i], param1_norm, param2_norm)
        
        # Validate and return
        if graph.is_valid_dag():
            return graph
        else:
            # Fallback: return unmodified sample
            logger.warning("Decoded graph invalid, returning sample")
            return self.search_space.sample()
    
    def _encode_operation(self, operation: str) -> int:
        """
        Encode operation name to integer ID.
        
        Args:
            operation: Operation name
            
        Returns:
            Operation ID (0 to vocab_size-1)
        """
        try:
            return self.operation_vocab.index(operation)
        except ValueError:
            # Unknown operation -> map to first
            return 0
    
    def _encode_parameters(self, node: GraphNode) -> Tuple[float, float]:
        """
        Encode node hyperparameters to [0,1] range.
        
        Args:
            node: GraphNode
            
        Returns:
            (param1, param2) tuple of normalized values
        """
        param1, param2 = 0.0, 0.0
        
        if node.operation == 'conv2d':
            # Normalize filters and kernel_size
            filters = node.params.get('filters', 32)
            kernel_size = node.params.get('kernel_size', 3)
            param1 = np.clip(filters / 512.0, 0, 1)
            param2 = np.clip(kernel_size / 7.0, 0, 1)
        
        elif node.operation == 'dense':
            # Normalize units
            units = node.params.get('units', 128)
            param1 = np.clip(units / 1024.0, 0, 1)
            param2 = 0.0
        
        elif node.operation == 'dropout':
            # Normalize rate
            rate = node.params.get('rate', 0.5)
            param1 = np.clip(rate, 0, 1)
            param2 = 0.0
        
        elif node.operation in ['maxpool', 'avgpool']:
            # Normalize pool_size
            pool_size = node.params.get('pool_size', 2)
            param1 = np.clip(pool_size / 4.0, 0, 1)
            param2 = 0.0
        
        return param1, param2
    
    def _decode_parameters(
        self,
        node: GraphNode,
        param1: float,
        param2: float
    ) -> None:
        """
        Decode normalized parameters and update node.
        
        Args:
            node: GraphNode to update
            param1: First normalized parameter
            param2: Second normalized parameter
        """
        if node.operation == 'conv2d':
            # Denormalize filters and kernel_size
            filters = int(param1 * 512)
            filters = max(16, min(512, filters))
            kernel_size = int(param2 * 7)
            kernel_size = max(1, min(7, kernel_size))
            if kernel_size % 2 == 0:
                kernel_size += 1  # Make odd
            
            node.params['filters'] = filters
            node.params['kernel_size'] = kernel_size
        
        elif node.operation == 'dense':
            # Denormalize units
            units = int(param1 * 1024)
            units = max(32, min(1024, units))
            node.params['units'] = units
        
        elif node.operation == 'dropout':
            # Denormalize rate
            rate = np.clip(param1, 0.1, 0.9)
            node.params['rate'] = rate
        
        elif node.operation in ['maxpool', 'avgpool']:
            # Denormalize pool_size
            pool_size = int(param1 * 4)
            pool_size = max(2, min(4, pool_size))
            node.params['pool_size'] = pool_size
    
    def get_dimension(self) -> int:
        """Get encoding dimension."""
        return self.dim
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        Get bounds for each dimension.
        
        Returns:
            List of (min, max) tuples
        """
        # All dimensions in [0, 1]
        return [(0.0, 1.0) for _ in range(self.dim)]


class ContinuousArchitectureSpace:
    """
    Wrapper providing continuous interface to discrete architecture space.
    
    Combines encoder with evaluation, providing a continuous objective
    function for optimizers like CMA-ES and PSO.
    
    Example:
        >>> space = ContinuousArchitectureSpace(search_space, evaluator)
        >>> fitness = space.evaluate(vector)  # Evaluate continuous vector
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: callable,
        max_nodes: int = 20
    ):
        """
        Initialize continuous space.
        
        Args:
            search_space: Discrete architecture search space
            evaluator: Function to evaluate ModelGraph -> fitness
            max_nodes: Maximum nodes in encoding
        """
        self.search_space = search_space
        self.evaluator = evaluator
        self.encoder = ArchitectureEncoder(search_space, max_nodes)
        
        self.evaluation_count = 0
        self.cache: Dict[str, float] = {}
    
    def evaluate(self, vector: np.ndarray) -> float:
        """
        Evaluate fitness of continuous vector.
        
        Args:
            vector: Continuous architecture encoding
            
        Returns:
            Fitness value
        """
        # Check cache
        vector_key = vector.tobytes()
        if vector_key in self.cache:
            return self.cache[vector_key]
        
        # Decode to graph
        graph = self.encoder.decode(vector)
        
        # Evaluate
        fitness = self.evaluator(graph)
        
        # Cache
        self.cache[vector_key] = fitness
        self.evaluation_count += 1
        
        return fitness
    
    def get_dimension(self) -> int:
        """Get dimension of continuous space."""
        return self.encoder.get_dimension()
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for optimization."""
        return self.encoder.get_bounds()
    
    def random_vector(self) -> np.ndarray:
        """
        Sample random vector in continuous space.
        
        Returns:
            Random vector
        """
        return np.random.rand(self.encoder.get_dimension())


def test_encoding_invertibility(
    search_space: SearchSpace,
    n_samples: int = 100
) -> float:
    """
    Test encoding/decoding invertibility.
    
    Measures how well encoding->decoding preserves architecture properties.
    
    Args:
        search_space: SearchSpace to test
        n_samples: Number of test samples
        
    Returns:
        Average similarity score (0-1)
    """
    encoder = ArchitectureEncoder(search_space)
    
    similarities = []
    
    for _ in range(n_samples):
        # Sample architecture
        graph1 = search_space.sample()
        
        # Encode and decode
        vector = encoder.encode(graph1)
        graph2 = encoder.decode(vector)
        
        # Measure similarity (simple: count matching operations)
        ops1 = [n.operation for n in graph1.nodes.values()]
        ops2 = [n.operation for n in graph2.nodes.values()]
        
        matches = sum(1 for o1, o2 in zip(ops1, ops2) if o1 == o2)
        similarity = matches / max(len(ops1), len(ops2))
        
        similarities.append(similarity)
    
    avg_similarity = np.mean(similarities)
    logger.info(f"Encoding invertibility: {avg_similarity:.2%}")
    
    return avg_similarity


# Convenience functions
def encode_architecture(
    graph: ModelGraph,
    search_space: SearchSpace,
    max_nodes: int = 20
) -> np.ndarray:
    """Quick encoding of single architecture."""
    encoder = ArchitectureEncoder(search_space, max_nodes)
    return encoder.encode(graph)


def decode_architecture(
    vector: np.ndarray,
    search_space: SearchSpace,
    max_nodes: int = 20
) -> ModelGraph:
    """Quick decoding of single vector."""
    encoder = ArchitectureEncoder(search_space, max_nodes)
    return encoder.decode(vector)
