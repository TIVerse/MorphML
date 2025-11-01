"""Advanced crossover operators for genetic algorithms."""

import random
from typing import List, Optional, Tuple

from morphml.core.graph import GraphEdge, GraphNode, ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class CrossoverOperator:
    """Base class for crossover operators."""

    def __init__(self, name: str):
        """Initialize crossover operator."""
        self.name = name
    
    def crossover(
        self,
        parent1: ModelGraph,
        parent2: ModelGraph
    ) -> Tuple[ModelGraph, ModelGraph]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent graph
            parent2: Second parent graph
        
        Returns:
            Tuple of two offspring graphs
        """
        raise NotImplementedError


class SinglePointCrossover(CrossoverOperator):
    """Single-point crossover for graphs."""

    def __init__(self):
        """Initialize single-point crossover."""
        super().__init__("SinglePointCrossover")
    
    def crossover(
        self,
        parent1: ModelGraph,
        parent2: ModelGraph
    ) -> Tuple[ModelGraph, ModelGraph]:
        """Perform single-point crossover."""
        # Get sorted nodes
        try:
            nodes1 = parent1.topological_sort()
            nodes2 = parent2.topological_sort()
        except Exception:
            # Fallback to random ordering
            nodes1 = list(parent1.nodes.values())
            nodes2 = list(parent2.nodes.values())
        
        if len(nodes1) < 2 or len(nodes2) < 2:
            # Can't crossover, return clones
            return parent1.clone(), parent2.clone()
        
        # Choose crossover point
        point = random.randint(1, min(len(nodes1), len(nodes2)) - 1)
        
        # Create offspring
        child1 = ModelGraph()
        child2 = ModelGraph()
        
        # Child 1: nodes1[:point] + nodes2[point:]
        for node in nodes1[:point]:
            child1.add_node(node.clone())
        for node in nodes2[point:]:
            child1.add_node(node.clone())
        
        # Child 2: nodes2[:point] + nodes1[point:]
        for node in nodes2[:point]:
            child2.add_node(node.clone())
        for node in nodes1[point:]:
            child2.add_node(node.clone())
        
        # Reconnect edges
        self._reconnect_edges(child1)
        self._reconnect_edges(child2)
        
        return child1, child2
    
    def _reconnect_edges(self, graph: ModelGraph) -> None:
        """Reconnect edges in a sensible way."""
        nodes = list(graph.nodes.values())
        
        for i in range(len(nodes) - 1):
            try:
                graph.add_edge(GraphEdge(nodes[i], nodes[i + 1]))
            except Exception:
                pass


class TwoPointCrossover(CrossoverOperator):
    """Two-point crossover for graphs."""

    def __init__(self):
        """Initialize two-point crossover."""
        super().__init__("TwoPointCrossover")
    
    def crossover(
        self,
        parent1: ModelGraph,
        parent2: ModelGraph
    ) -> Tuple[ModelGraph, ModelGraph]:
        """Perform two-point crossover."""
        try:
            nodes1 = parent1.topological_sort()
            nodes2 = parent2.topological_sort()
        except Exception:
            nodes1 = list(parent1.nodes.values())
            nodes2 = list(parent2.nodes.values())
        
        if len(nodes1) < 3 or len(nodes2) < 3:
            return parent1.clone(), parent2.clone()
        
        # Choose two crossover points
        max_len = min(len(nodes1), len(nodes2))
        point1 = random.randint(1, max_len - 2)
        point2 = random.randint(point1 + 1, max_len - 1)
        
        # Create offspring
        child1 = ModelGraph()
        child2 = ModelGraph()
        
        # Child 1: nodes1[:point1] + nodes2[point1:point2] + nodes1[point2:]
        for node in nodes1[:point1]:
            child1.add_node(node.clone())
        for node in nodes2[point1:point2]:
            child1.add_node(node.clone())
        for node in nodes1[point2:]:
            child1.add_node(node.clone())
        
        # Child 2: nodes2[:point1] + nodes1[point1:point2] + nodes2[point2:]
        for node in nodes2[:point1]:
            child2.add_node(node.clone())
        for node in nodes1[point1:point2]:
            child2.add_node(node.clone())
        for node in nodes2[point2:]:
            child2.add_node(node.clone())
        
        # Reconnect edges
        self._reconnect_edges(child1)
        self._reconnect_edges(child2)
        
        return child1, child2
    
    def _reconnect_edges(self, graph: ModelGraph) -> None:
        """Reconnect edges."""
        nodes = list(graph.nodes.values())
        
        for i in range(len(nodes) - 1):
            try:
                graph.add_edge(GraphEdge(nodes[i], nodes[i + 1]))
            except Exception:
                pass


class UniformCrossover(CrossoverOperator):
    """Uniform crossover for graphs."""

    def __init__(self, swap_probability: float = 0.5):
        """
        Initialize uniform crossover.
        
        Args:
            swap_probability: Probability of swapping each node
        """
        super().__init__("UniformCrossover")
        self.swap_probability = swap_probability
    
    def crossover(
        self,
        parent1: ModelGraph,
        parent2: ModelGraph
    ) -> Tuple[ModelGraph, ModelGraph]:
        """Perform uniform crossover."""
        try:
            nodes1 = parent1.topological_sort()
            nodes2 = parent2.topological_sort()
        except Exception:
            nodes1 = list(parent1.nodes.values())
            nodes2 = list(parent2.nodes.values())
        
        # Make lists same length by padding
        max_len = max(len(nodes1), len(nodes2))
        
        child1 = ModelGraph()
        child2 = ModelGraph()
        
        for i in range(max_len):
            if random.random() < self.swap_probability:
                # Swap
                if i < len(nodes2):
                    child1.add_node(nodes2[i].clone())
                if i < len(nodes1):
                    child2.add_node(nodes1[i].clone())
            else:
                # Don't swap
                if i < len(nodes1):
                    child1.add_node(nodes1[i].clone())
                if i < len(nodes2):
                    child2.add_node(nodes2[i].clone())
        
        # Reconnect edges
        self._reconnect_edges(child1)
        self._reconnect_edges(child2)
        
        return child1, child2
    
    def _reconnect_edges(self, graph: ModelGraph) -> None:
        """Reconnect edges."""
        nodes = list(graph.nodes.values())
        
        for i in range(len(nodes) - 1):
            try:
                graph.add_edge(GraphEdge(nodes[i], nodes[i + 1]))
            except Exception:
                pass


class SubgraphCrossover(CrossoverOperator):
    """Crossover by exchanging subgraphs."""

    def __init__(self):
        """Initialize subgraph crossover."""
        super().__init__("SubgraphCrossover")
    
    def crossover(
        self,
        parent1: ModelGraph,
        parent2: ModelGraph
    ) -> Tuple[ModelGraph, ModelGraph]:
        """Perform subgraph crossover."""
        # Find compatible subgraphs
        nodes1 = list(parent1.nodes.values())
        nodes2 = list(parent2.nodes.values())
        
        if len(nodes1) < 2 or len(nodes2) < 2:
            return parent1.clone(), parent2.clone()
        
        # Select random subgraph from each parent
        subgraph_size = min(3, len(nodes1) // 2, len(nodes2) // 2)
        
        start1 = random.randint(0, len(nodes1) - subgraph_size)
        subgraph1 = nodes1[start1:start1 + subgraph_size]
        
        start2 = random.randint(0, len(nodes2) - subgraph_size)
        subgraph2 = nodes2[start2:start2 + subgraph_size]
        
        # Create offspring
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        # Replace subgraphs
        for i, node in enumerate(subgraph2):
            if start1 + i < len(nodes1):
                old_id = nodes1[start1 + i].id
                if old_id in child1.nodes:
                    child1.nodes[old_id] = node.clone()
        
        for i, node in enumerate(subgraph1):
            if start2 + i < len(nodes2):
                old_id = nodes2[start2 + i].id
                if old_id in child2.nodes:
                    child2.nodes[old_id] = node.clone()
        
        return child1, child2


class LayerWiseCrossover(CrossoverOperator):
    """Crossover by exchanging layers of same type."""

    def __init__(self):
        """Initialize layer-wise crossover."""
        super().__init__("LayerWiseCrossover")
    
    def crossover(
        self,
        parent1: ModelGraph,
        parent2: ModelGraph
    ) -> Tuple[ModelGraph, ModelGraph]:
        """Perform layer-wise crossover."""
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        # Group nodes by operation type
        ops1 = {}
        for node in child1.nodes.values():
            if node.operation not in ops1:
                ops1[node.operation] = []
            ops1[node.operation].append(node)
        
        ops2 = {}
        for node in child2.nodes.values():
            if node.operation not in ops2:
                ops2[node.operation] = []
            ops2[node.operation].append(node)
        
        # Exchange layers of same type
        common_ops = set(ops1.keys()) & set(ops2.keys())
        
        for op in common_ops:
            if random.random() < 0.5:
                # Swap parameters of first node of this type
                if ops1[op] and ops2[op]:
                    node1 = ops1[op][0]
                    node2 = ops2[op][0]
                    
                    # Swap parameters
                    node1.params, node2.params = node2.params.copy(), node1.params.copy()
        
        return child1, child2


class AdaptiveCrossover(CrossoverOperator):
    """Adaptive crossover that chooses operator based on parents."""

    def __init__(self):
        """Initialize adaptive crossover."""
        super().__init__("AdaptiveCrossover")
        self.operators = [
            SinglePointCrossover(),
            TwoPointCrossover(),
            UniformCrossover(),
            SubgraphCrossover(),
            LayerWiseCrossover(),
        ]
    
    def crossover(
        self,
        parent1: ModelGraph,
        parent2: ModelGraph
    ) -> Tuple[ModelGraph, ModelGraph]:
        """Perform adaptive crossover."""
        # Choose operator based on parent similarity
        similarity = self._calculate_similarity(parent1, parent2)
        
        if similarity > 0.7:
            # High similarity - use more explorative crossover
            operator = random.choice([
                self.operators[2],  # Uniform
                self.operators[3],  # Subgraph
            ])
        elif similarity < 0.3:
            # Low similarity - use more conservative crossover
            operator = random.choice([
                self.operators[0],  # Single-point
                self.operators[4],  # Layer-wise
            ])
        else:
            # Medium similarity - use balanced crossover
            operator = self.operators[1]  # Two-point
        
        return operator.crossover(parent1, parent2)
    
    def _calculate_similarity(self, graph1: ModelGraph, graph2: ModelGraph) -> float:
        """Calculate structural similarity between graphs."""
        # Simple similarity based on node count and types
        nodes1 = graph1.nodes.values()
        nodes2 = graph2.nodes.values()
        
        if len(nodes1) == 0 or len(nodes2) == 0:
            return 0.0
        
        # Size similarity
        size_sim = 1.0 - abs(len(nodes1) - len(nodes2)) / max(len(nodes1), len(nodes2))
        
        # Operation type similarity
        ops1 = set(n.operation for n in nodes1)
        ops2 = set(n.operation for n in nodes2)
        
        if not ops1 or not ops2:
            return size_sim
        
        intersection = len(ops1 & ops2)
        union = len(ops1 | ops2)
        op_sim = intersection / union if union > 0 else 0.0
        
        return 0.5 * size_sim + 0.5 * op_sim


class MultiParentCrossover(CrossoverOperator):
    """Crossover using multiple parents."""

    def __init__(self, num_parents: int = 3):
        """
        Initialize multi-parent crossover.
        
        Args:
            num_parents: Number of parents to use
        """
        super().__init__("MultiParentCrossover")
        self.num_parents = num_parents
    
    def crossover_multi(
        self,
        parents: List[ModelGraph]
    ) -> ModelGraph:
        """
        Perform multi-parent crossover.
        
        Args:
            parents: List of parent graphs
        
        Returns:
            Single offspring graph
        """
        if not parents:
            raise ValueError("No parents provided")
        
        if len(parents) < 2:
            return parents[0].clone()
        
        # Collect all nodes from parents
        all_nodes = []
        for parent in parents:
            try:
                nodes = parent.topological_sort()
            except Exception:
                nodes = list(parent.nodes.values())
            all_nodes.append(nodes)
        
        # Build offspring by selecting from parents
        child = ModelGraph()
        max_len = max(len(nodes) for nodes in all_nodes)
        
        for i in range(max_len):
            # Select node from random parent
            available_parents = [nodes for nodes in all_nodes if i < len(nodes)]
            if available_parents:
                selected_nodes = random.choice(available_parents)
                child.add_node(selected_nodes[i].clone())
        
        # Reconnect edges
        nodes = list(child.nodes.values())
        for i in range(len(nodes) - 1):
            try:
                child.add_edge(GraphEdge(nodes[i], nodes[i + 1]))
            except Exception:
                pass
        
        return child


def get_crossover_operator(name: str, **kwargs) -> CrossoverOperator:
    """
    Get crossover operator by name.
    
    Args:
        name: Operator name
        **kwargs: Operator parameters
    
    Returns:
        CrossoverOperator instance
    """
    operators = {
        "single_point": SinglePointCrossover,
        "two_point": TwoPointCrossover,
        "uniform": UniformCrossover,
        "subgraph": SubgraphCrossover,
        "layerwise": LayerWiseCrossover,
        "adaptive": AdaptiveCrossover,
    }
    
    operator_class = operators.get(name.lower())
    
    if operator_class is None:
        raise ValueError(f"Unknown crossover operator: {name}")
    
    return operator_class(**kwargs)
