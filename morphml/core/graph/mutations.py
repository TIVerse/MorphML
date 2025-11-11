"""Graph mutation operations for architecture search."""

import random
from typing import List, Optional

from morphml.core.graph.edge import GraphEdge
from morphml.core.graph.graph import ModelGraph
from morphml.core.graph.node import GraphNode
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class GraphMutator:
    """
    Applies mutations to ModelGraph for architecture search.

    Mutation operations:
    - add_node: Insert a new node
    - remove_node: Remove an existing node
    - modify_node: Change node parameters
    - add_edge: Add a new connection
    - remove_edge: Remove a connection

    All mutations preserve DAG property.

    Example:
        >>> mutator = GraphMutator()
        >>> mutated_graph = mutator.mutate(original_graph, mutation_rate=0.1)
    """

    def __init__(self, operation_types: Optional[List[str]] = None):
        """
        Initialize graph mutator.

        Args:
            operation_types: List of available operation types for mutations
        """
        self.operation_types = operation_types or [
            "conv2d",
            "maxpool",
            "avgpool",
            "dense",
            "relu",
            "batchnorm",
            "dropout",
        ]

    def mutate(
        self,
        graph: ModelGraph,
        mutation_rate: float = 0.1,
        max_mutations: Optional[int] = None,
    ) -> ModelGraph:
        """
        Apply random mutations to graph.

        Args:
            graph: Original graph
            mutation_rate: Probability of mutating each component
            max_mutations: Maximum number of mutations (None = no limit)

        Returns:
            Mutated graph (new instance)
        """
        mutated = graph.clone()
        mutations_applied = 0

        # Available mutation operations
        mutation_ops = [
            self.add_node_mutation,
            self.remove_node_mutation,
            self.modify_node_mutation,
            self.add_edge_mutation,
            self.remove_edge_mutation,
        ]

        # Apply mutations
        while random.random() < mutation_rate:
            if max_mutations and mutations_applied >= max_mutations:
                break

            # Select random mutation
            mutation_op = random.choice(mutation_ops)

            try:
                mutation_op(mutated)
                mutations_applied += 1
            except Exception as e:
                logger.debug(f"Mutation failed: {e}")
                continue

        logger.debug(f"Applied {mutations_applied} mutations")
        return mutated

    def add_node_mutation(self, graph: ModelGraph) -> None:
        """
        Add a new node to the graph.

        Strategy:
        1. Select random operation type
        2. Insert between two connected nodes
        3. Update edges accordingly

        Args:
            graph: Graph to mutate (modified in-place)
        """
        if len(graph.edges) == 0:
            logger.debug("No edges to insert node between")
            return

        # Select random edge to split
        edge = random.choice(list(graph.edges.values()))

        # Create new node
        operation = random.choice(self.operation_types)
        new_node = GraphNode.create(operation, params=self._random_params(operation))

        # Add node to graph
        graph.add_node(new_node)

        # Remove old edge
        graph.remove_edge(edge.id)

        # Add new edges: source -> new_node -> target
        edge1 = GraphEdge(edge.source, new_node)
        edge2 = GraphEdge(new_node, edge.target)

        graph.add_edge(edge1)
        graph.add_edge(edge2)

        logger.debug(
            f"Added node: {operation} between {edge.source.operation} and {edge.target.operation}"
        )

    def remove_node_mutation(self, graph: ModelGraph) -> None:
        """
        Remove a node from the graph.

        Strategy:
        1. Select random non-input/non-output node
        2. Connect its predecessors directly to its successors
        3. Remove the node and its edges

        Args:
            graph: Graph to mutate (modified in-place)
        """
        # Get candidates (exclude input/output nodes)
        input_ids = {n.id for n in graph.get_input_nodes()}
        output_ids = {n.id for n in graph.get_output_nodes()}

        candidates = [
            node
            for node in graph.nodes.values()
            if node.id not in input_ids and node.id not in output_ids
        ]

        if not candidates:
            logger.debug("No nodes available for removal")
            return

        # Select random node
        node_to_remove = random.choice(candidates)

        # Connect predecessors to successors
        for pred in node_to_remove.predecessors:
            for succ in node_to_remove.successors:
                # Check if edge doesn't already exist
                existing = any(
                    e.source.id == pred.id and e.target.id == succ.id for e in graph.edges.values()
                )

                if not existing:
                    new_edge = GraphEdge(pred, succ)
                    try:
                        graph.add_edge(new_edge)
                    except Exception as e:
                        logger.debug(f"Failed to add bypass edge: {e}")

        # Remove node
        graph.remove_node(node_to_remove.id)

        logger.debug(f"Removed node: {node_to_remove.operation}")

    def modify_node_mutation(self, graph: ModelGraph) -> None:
        """
        Modify parameters of an existing node.

        Args:
            graph: Graph to mutate (modified in-place)
        """
        if not graph.nodes:
            return

        # Select random node
        node = random.choice(list(graph.nodes.values()))

        # Modify a random parameter
        if node.params:
            param_key = random.choice(list(node.params.keys()))
            old_value = node.params[param_key]

            # Generate new value based on type
            if isinstance(old_value, int):
                # Multiply by random factor or add random offset
                if random.random() < 0.5:
                    node.params[param_key] = max(1, old_value * random.choice([2, 4, 8]))
                else:
                    node.params[param_key] = max(1, old_value // random.choice([2, 4]))

            elif isinstance(old_value, float):
                node.params[param_key] = old_value * random.uniform(0.5, 2.0)

            logger.debug(
                f"Modified {node.operation}.{param_key}: {old_value} -> {node.params[param_key]}"
            )
        else:
            # Add random parameters if node has none
            node.params = self._random_params(node.operation)
            logger.debug(f"Added params to {node.operation}: {node.params}")

    def add_edge_mutation(self, graph: ModelGraph) -> None:
        """
        Add a new edge (skip connection).

        Args:
            graph: Graph to mutate (modified in-place)
        """
        if len(graph.nodes) < 2:
            return

        nodes = list(graph.nodes.values())

        # Try multiple times to find valid edge
        for _ in range(10):
            source = random.choice(nodes)
            target = random.choice(nodes)

            # Skip if same node or edge exists or would create cycle
            if source.id == target.id:
                continue

            if any(
                e.source.id == source.id and e.target.id == target.id for e in graph.edges.values()
            ):
                continue

            try:
                new_edge = GraphEdge(source, target)
                graph.add_edge(new_edge)
                logger.debug(f"Added edge: {source.operation} -> {target.operation}")
                return
            except Exception:
                continue

        logger.debug("Failed to add edge after multiple attempts")

    def remove_edge_mutation(self, graph: ModelGraph) -> None:
        """
        Remove an edge.

        Ensures graph remains connected.

        Args:
            graph: Graph to mutate (modified in-place)
        """
        if len(graph.edges) <= len(graph.nodes) - 1:
            # Need at least n-1 edges to stay connected
            logger.debug("Too few edges to remove")
            return

        # Get non-critical edges (removing won't disconnect graph)
        candidates = []
        for edge in graph.edges.values():
            # Check if target has other predecessors
            if len(edge.target.predecessors) > 1:
                candidates.append(edge)

        if not candidates:
            logger.debug("No removable edges found")
            return

        edge_to_remove = random.choice(candidates)
        graph.remove_edge(edge_to_remove.id)

        logger.debug(
            f"Removed edge: {edge_to_remove.source.operation} -> "
            f"{edge_to_remove.target.operation}"
        )

    def _random_params(self, operation: str) -> dict:
        """
        Generate random parameters for an operation.

        Args:
            operation: Operation type

        Returns:
            Dictionary of parameters
        """
        if operation == "conv2d":
            return {
                "filters": random.choice([32, 64, 128, 256]),
                "kernel_size": random.choice([3, 5, 7]),
                "padding": "same",
            }

        elif operation == "dense":
            return {"units": random.choice([64, 128, 256, 512])}

        elif operation in ["maxpool", "avgpool"]:
            return {"pool_size": random.choice([2, 3]), "stride": 2}

        elif operation == "dropout":
            return {"rate": random.uniform(0.1, 0.5)}

        elif operation == "batchnorm":
            return {}

        elif operation == "relu":
            return {}

        else:
            return {}


def crossover(parent1: ModelGraph, parent2: ModelGraph) -> tuple[ModelGraph, ModelGraph]:
    """
    Perform crossover between two graphs.

    Creates two offspring by exchanging subgraphs.

    Args:
        parent1: First parent graph
        parent2: Second parent graph

    Returns:
        Tuple of two offspring graphs

    Note:
        Implements single-point crossover by splitting parents at a random point
        and combining their subgraphs.
    """
    import random

    # Get topologically sorted nodes from both parents
    try:
        nodes1 = parent1.topological_sort()
        nodes2 = parent2.topological_sort()
    except Exception as e:
        logger.warning(f"Crossover failed during topological sort: {e}, returning clones")
        return parent1.clone(), parent2.clone()

    # If either parent is too small, just return clones
    if len(nodes1) < 3 or len(nodes2) < 3:
        logger.debug("Parents too small for crossover, returning clones")
        return parent1.clone(), parent2.clone()

    # Choose crossover points (excluding input/output nodes)
    point1 = random.randint(1, len(nodes1) - 2)
    point2 = random.randint(1, len(nodes2) - 2)

    # Create offspring by combining subgraphs
    offspring1 = ModelGraph(metadata={"crossover": "parent1_start + parent2_end"})
    offspring2 = ModelGraph(metadata={"crossover": "parent2_start + parent1_end"})

    try:
        # Offspring 1: first part of parent1 + second part of parent2
        for i, node in enumerate(nodes1[:point1]):
            new_node = node.clone()
            offspring1.add_node(new_node)

        for i, node in enumerate(nodes2[point2:]):
            new_node = node.clone()
            offspring1.add_node(new_node)

        # Connect the nodes sequentially
        all_nodes1 = list(offspring1.nodes.values())
        for i in range(len(all_nodes1) - 1):
            edge = GraphEdge(all_nodes1[i], all_nodes1[i + 1])
            offspring1.add_edge(edge)

        # Offspring 2: first part of parent2 + second part of parent1
        for i, node in enumerate(nodes2[:point2]):
            new_node = node.clone()
            offspring2.add_node(new_node)

        for i, node in enumerate(nodes1[point1:]):
            new_node = node.clone()
            offspring2.add_node(new_node)

        # Connect the nodes sequentially
        all_nodes2 = list(offspring2.nodes.values())
        for i in range(len(all_nodes2) - 1):
            edge = GraphEdge(all_nodes2[i], all_nodes2[i + 1])
            offspring2.add_edge(edge)

        logger.debug(f"Performed single-point crossover at points {point1}/{point2}")

        # Validate offspring
        if not offspring1.is_valid() or not offspring2.is_valid():
            logger.warning("Crossover produced invalid offspring, returning clones")
            return parent1.clone(), parent2.clone()

        return offspring1, offspring2

    except Exception as e:
        logger.warning(f"Crossover failed: {e}, returning clones")
        return parent1.clone(), parent2.clone()
