"""Base class for Bayesian optimization algorithms.

This module provides the foundation for sample-efficient Bayesian optimization
methods that use surrogate models to guide the search process.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from morphml.core.dsl import SearchSpace
from morphml.core.graph import ModelGraph
from morphml.core.search import Individual
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class BaseBayesianOptimizer:
    """
    Base class for Bayesian optimization algorithms.

    Bayesian optimization uses a surrogate model to approximate the
    expensive-to-evaluate fitness function, enabling intelligent
    exploration-exploitation trade-offs.

    Key components:
    1. **Surrogate Model**: Approximates f(x) (e.g., GP, RF, TPE)
    2. **Acquisition Function**: Decides where to sample next
    3. **Architecture Encoding**: Maps graphs to continuous/discrete vectors

    Attributes:
        search_space: SearchSpace defining architecture options
        config: Algorithm configuration
        generation: Current generation/iteration
        history: List of all evaluated architectures
        best_individual: Best architecture found so far

    Example:
        >>> from morphml.optimizers.bayesian import GaussianProcessOptimizer
        >>> optimizer = GaussianProcessOptimizer(
        ...     search_space=space,
        ...     config={'acquisition': 'ei', 'n_initial_points': 10}
        ... )
        >>> best = optimizer.optimize(evaluator)
    """

    def __init__(self, search_space: SearchSpace, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Bayesian optimizer.

        Args:
            search_space: SearchSpace to sample architectures from
            config: Algorithm configuration dictionary
        """
        self.search_space = search_space
        self.config = config or {}

        self.generation = 0
        self.history: List[Dict[str, Any]] = []
        self.best_individual: Optional[Individual] = None

        # Configuration parameters
        self.n_initial_points = self.config.get("n_initial_points", 10)
        self.max_iterations = self.config.get("max_iterations", 100)
        self.random_state = self.config.get("random_state", None)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        logger.info(
            f"Initialized {self.__class__.__name__} with " f"{self.n_initial_points} initial points"
        )

    def optimize(
        self, evaluator: Any, max_evaluations: Optional[int] = None, callback: Optional[Any] = None
    ) -> Individual:
        """
        Run Bayesian optimization loop.

        Args:
            evaluator: Function that evaluates ModelGraph fitness
            max_evaluations: Maximum number of evaluations (overrides config)
            callback: Optional callback function called each iteration

        Returns:
            Best Individual found

        Example:
            >>> def my_evaluator(graph):
            ...     return train_and_evaluate(graph)
            >>> best = optimizer.optimize(my_evaluator, max_evaluations=50)
        """
        max_evals = max_evaluations or self.max_iterations

        logger.info(f"Starting Bayesian optimization for {max_evals} evaluations")

        for iteration in range(max_evals):
            # Ask: Get next candidate(s) to evaluate
            candidates = self.ask()

            # Evaluate candidates
            results = []
            for graph in candidates:
                fitness = evaluator(graph)
                results.append((graph, fitness))

                # Track best
                individual = Individual(graph)
                individual.fitness = fitness

                if self.best_individual is None or fitness > self.best_individual.fitness:
                    self.best_individual = individual
                    logger.info(f"Iteration {iteration}: New best fitness = {fitness:.4f}")

            # Tell: Update surrogate model with results
            self.tell(results)

            # Callback
            if callback is not None:
                callback(iteration, self.best_individual, self.history)

            self.generation += 1

        logger.info(f"Optimization complete. Best fitness: {self.best_individual.fitness:.4f}")

        return self.best_individual

    @abstractmethod
    def ask(self) -> List[ModelGraph]:
        """
        Generate next candidate architecture(s) to evaluate.

        Uses the surrogate model and acquisition function to select
        promising architectures.

        Returns:
            List of ModelGraph candidates
        """
        pass

    @abstractmethod
    def tell(self, results: List[Tuple[ModelGraph, float]]) -> None:
        """
        Update surrogate model with evaluation results.

        Args:
            results: List of (graph, fitness) tuples
        """
        pass

    def get_best(self) -> Optional[Individual]:
        """
        Get best individual found so far.

        Returns:
            Best Individual or None if no evaluations yet
        """
        return self.best_individual

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get optimization history.

        Returns:
            List of dictionaries with generation, graph, fitness
        """
        return self.history

    def reset(self) -> None:
        """Reset optimizer to initial state."""
        self.generation = 0
        self.history = []
        self.best_individual = None
        logger.info(f"{self.__class__.__name__} reset to initial state")

    def _encode_architecture(self, graph: ModelGraph) -> np.ndarray:
        """
        Encode ModelGraph as fixed-length numerical vector.

        This is a critical method that maps complex graph structures
        to continuous/discrete vectors suitable for surrogate models.

        Encoding strategies:
        1. **Positional Encoding**: Represent nodes by position in topological order
        2. **Operation One-Hot**: Encode operation types
        3. **Hyperparameters**: Include numerical parameters
        4. **Connectivity**: Encode edge structure

        Args:
            graph: ModelGraph to encode

        Returns:
            Fixed-length numpy array
        """
        # Get topological ordering
        try:
            topo_order = list(graph.topological_sort())
        except Exception:
            # Handle invalid graphs
            topo_order = list(graph.nodes.values())

        # Define operation vocabulary
        operation_types = [
            "input",
            "output",
            "conv2d",
            "dense",
            "relu",
            "sigmoid",
            "tanh",
            "maxpool",
            "avgpool",
            "batchnorm",
            "dropout",
            "flatten",
            "add",
            "concat",
        ]

        # Fixed encoding length (max depth)
        max_depth = 20
        encoding_per_node = 3  # operation_id, param1, param2

        encoding = []

        for i in range(max_depth):
            if i < len(topo_order):
                node = topo_order[i]

                # Encode operation type
                if node.operation in operation_types:
                    op_id = operation_types.index(node.operation)
                else:
                    op_id = 0  # Unknown operation
                encoding.append(float(op_id))

                # Encode key hyperparameters
                if node.operation == "conv2d":
                    filters = node.params.get("filters", 32)
                    kernel_size = node.params.get("kernel_size", 3)
                    encoding.extend([float(filters), float(kernel_size)])

                elif node.operation == "dense":
                    units = node.params.get("units", 128)
                    encoding.extend([float(units), 0.0])

                elif node.operation == "dropout":
                    rate = node.params.get("rate", 0.5)
                    encoding.extend([float(rate * 100), 0.0])

                elif node.operation in ["maxpool", "avgpool"]:
                    pool_size = node.params.get("pool_size", 2)
                    encoding.extend([float(pool_size), 0.0])

                else:
                    encoding.extend([0.0, 0.0])
            else:
                # Padding for shorter architectures
                encoding.extend([0.0] * encoding_per_node)

        return np.array(encoding, dtype=np.float64)

    def _decode_architecture(self, x: np.ndarray) -> ModelGraph:
        """
        Decode numerical vector back to ModelGraph.

        This is challenging because the mapping is many-to-one
        (many vectors may decode to similar graphs).

        Strategy:
        1. Sample random architecture from search space
        2. Use vector to guide mutations toward desired structure

        Args:
            x: Numerical encoding

        Returns:
            Decoded ModelGraph
        """
        # Simplified decoding: sample from search space
        # In practice, this would use more sophisticated reconstruction
        graph = self.search_space.sample()

        # TODO: Could add logic to mutate graph toward target encoding
        # For now, return sampled graph (acquisition still guides search)

        return graph

    def _get_encoding_bounds(self) -> List[Tuple[float, float]]:
        """
        Get bounds for architecture encoding dimensions.

        Returns:
            List of (min, max) tuples for each encoding dimension
        """
        operation_types_count = 14  # Number of supported operations
        max_filters = 512
        max_kernel_size = 7

        max_depth = 20

        bounds = []
        for _ in range(max_depth):
            bounds.append((0, operation_types_count))  # Operation ID
            bounds.append((0, max_filters))  # Param 1 (filters/units)
            bounds.append((0, max_kernel_size))  # Param 2 (kernel/pool size)

        return bounds


class BayesianOptimizationError(Exception):
    """Exception raised for errors in Bayesian optimization."""

    pass
