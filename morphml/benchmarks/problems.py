"""Benchmark problems for optimizer evaluation."""

import math
from typing import Callable

from morphml.core.dsl import Layer, SearchSpace
from morphml.core.graph import ModelGraph


class BenchmarkProblem:
    """Base class for benchmark problems."""

    def __init__(self, name: str, search_space: SearchSpace):
        """Initialize benchmark problem."""
        self.name = name
        self.search_space = search_space

    def evaluate(self, graph: ModelGraph) -> float:
        """Evaluate architecture on this problem."""
        raise NotImplementedError

    def get_optimal_fitness(self) -> float:
        """Get known optimal fitness if available."""
        return 1.0


class SimpleProblem(BenchmarkProblem):
    """Simple benchmark problem."""

    def __init__(self):
        """Initialize simple problem."""
        space = SearchSpace("simple_problem")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64], kernel_size=3),
            Layer.relu(),
            Layer.output(units=10),
        )
        super().__init__("SimpleProblem", space)

    def evaluate(self, graph: ModelGraph) -> float:
        """Simple evaluation based on node count."""
        num_nodes = len(graph.nodes)
        depth = graph.get_depth()

        # Prefer moderate complexity
        node_score = 1.0 / (1.0 + abs(num_nodes - 5))
        depth_score = 1.0 / (1.0 + abs(depth - 4))

        return 0.5 * node_score + 0.5 * depth_score


class ComplexProblem(BenchmarkProblem):
    """Complex benchmark problem with multiple criteria."""

    def __init__(self):
        """Initialize complex problem."""
        space = SearchSpace("complex_problem")
        space.add_layers(
            Layer.input(shape=(3, 224, 224)),
            Layer.conv2d(filters=[32, 64, 128, 256], kernel_size=[3, 5, 7]),
            Layer.relu(),
            Layer.batchnorm(),
            Layer.maxpool(pool_size=[2, 3]),
            Layer.conv2d(filters=[64, 128, 256, 512], kernel_size=[3, 5]),
            Layer.relu(),
            Layer.batchnorm(),
            Layer.maxpool(pool_size=2),
            Layer.dense(units=[256, 512, 1024, 2048]),
            Layer.dropout(rate=[0.3, 0.5, 0.7]),
            Layer.dense(units=[128, 256, 512]),
            Layer.output(units=1000),
        )
        super().__init__("ComplexProblem", space)

    def evaluate(self, graph: ModelGraph) -> float:
        """Complex evaluation considering multiple factors."""
        num_nodes = len(graph.nodes)
        depth = graph.get_depth()
        width = graph.get_max_width()
        params = graph.estimate_parameters()

        # Multiple objectives
        node_score = math.exp(-abs(num_nodes - 12) / 5.0)
        depth_score = math.exp(-abs(depth - 8) / 3.0)
        width_score = math.exp(-abs(width - 4) / 2.0)
        param_score = 1.0 / (1.0 + params / 5000000.0)

        # Weighted combination
        score = 0.25 * node_score + 0.25 * depth_score + 0.25 * width_score + 0.25 * param_score

        return min(1.0, max(0.0, score))


class MultiModalProblem(BenchmarkProblem):
    """Multi-modal problem with multiple local optima."""

    def __init__(self):
        """Initialize multi-modal problem."""
        space = SearchSpace("multimodal_problem")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[16, 32, 64, 128, 256], kernel_size=[3, 5, 7]),
            Layer.relu(),
            Layer.maxpool(pool_size=[2, 3]),
            Layer.conv2d(filters=[32, 64, 128, 256], kernel_size=[3, 5]),
            Layer.relu(),
            Layer.dense(units=[64, 128, 256, 512, 1024]),
            Layer.dropout(rate=[0.2, 0.3, 0.4, 0.5, 0.6]),
            Layer.output(units=10),
        )
        super().__init__("MultiModalProblem", space)

    def evaluate(self, graph: ModelGraph) -> float:
        """Evaluation with multiple peaks."""
        num_nodes = len(graph.nodes)
        depth = graph.get_depth()

        # Create multiple peaks
        peak1 = math.exp(-((num_nodes - 6) ** 2 + (depth - 3) ** 2) / 4.0)
        peak2 = 0.8 * math.exp(-((num_nodes - 9) ** 2 + (depth - 5) ** 2) / 3.0)
        peak3 = 0.6 * math.exp(-((num_nodes - 12) ** 2 + (depth - 7) ** 2) / 5.0)

        return max(peak1, peak2, peak3)


class ConstrainedProblem(BenchmarkProblem):
    """Problem with hard constraints."""

    def __init__(self):
        """Initialize constrained problem."""
        space = SearchSpace("constrained_problem")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64, 128], kernel_size=3),
            Layer.relu(),
            Layer.maxpool(pool_size=2),
            Layer.conv2d(filters=[64, 128, 256], kernel_size=3),
            Layer.relu(),
            Layer.dense(units=[128, 256, 512]),
            Layer.output(units=10),
        )

        # Add constraints
        def max_depth_constraint(g):
            return g.get_depth() <= 8

        def max_params_constraint(g):
            return g.estimate_parameters() <= 1000000

        space.add_constraint(max_depth_constraint)
        space.add_constraint(max_params_constraint)

        super().__init__("ConstrainedProblem", space)

    def evaluate(self, graph: ModelGraph) -> float:
        """Evaluation with penalty for constraint violation."""
        # Check constraints
        depth = graph.get_depth()
        params = graph.estimate_parameters()

        penalty = 0.0

        if depth > 8:
            penalty += 0.5 * (depth - 8) / 8.0

        if params > 1000000:
            penalty += 0.5 * (params - 1000000) / 1000000.0

        # Base fitness
        base_fitness = 0.5 + 0.05 * len(graph.nodes)

        # Apply penalty
        return max(0.0, base_fitness - penalty)


class NoisyProblem(BenchmarkProblem):
    """Problem with noisy evaluations."""

    def __init__(self, noise_level: float = 0.1):
        """Initialize noisy problem."""
        space = SearchSpace("noisy_problem")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64], kernel_size=3),
            Layer.relu(),
            Layer.maxpool(pool_size=2),
            Layer.dense(units=[128, 256]),
            Layer.output(units=10),
        )
        super().__init__("NoisyProblem", space)
        self.noise_level = noise_level

    def evaluate(self, graph: ModelGraph) -> float:
        """Evaluation with added noise."""
        import random

        # Base evaluation
        num_nodes = len(graph.nodes)
        base_fitness = 0.6 + 0.04 * num_nodes

        # Add noise
        noise = random.gauss(0, self.noise_level)
        noisy_fitness = base_fitness + noise

        return max(0.0, min(1.0, noisy_fitness))


class RastriginProblem(BenchmarkProblem):
    """Rastrigin-like function adapted for graphs."""

    def __init__(self):
        """Initialize Rastrigin problem."""
        space = SearchSpace("rastrigin_problem")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[16, 32, 64, 128], kernel_size=[3, 5, 7]),
            Layer.relu(),
            Layer.maxpool(pool_size=[2, 3]),
            Layer.dense(units=[64, 128, 256, 512]),
            Layer.output(units=10),
        )
        super().__init__("RastriginProblem", space)

    def evaluate(self, graph: ModelGraph) -> float:
        """Rastrigin-like evaluation."""
        num_nodes = len(graph.nodes)
        depth = graph.get_depth()

        # Rastrigin-like function
        A = 10
        x = (num_nodes - 8) / 4.0
        y = (depth - 5) / 3.0

        value = 2 * A
        value += x**2 - A * math.cos(2 * math.pi * x)
        value += y**2 - A * math.cos(2 * math.pi * y)

        # Convert to fitness (0-1 range, higher is better)
        fitness = 1.0 / (1.0 + value / 20.0)

        return fitness


class RosenbrockProblem(BenchmarkProblem):
    """Rosenbrock-like function adapted for graphs."""

    def __init__(self):
        """Initialize Rosenbrock problem."""
        space = SearchSpace("rosenbrock_problem")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5]),
            Layer.relu(),
            Layer.dense(units=[128, 256, 512]),
            Layer.output(units=10),
        )
        super().__init__("RosenbrockProblem", space)

    def evaluate(self, graph: ModelGraph) -> float:
        """Rosenbrock-like evaluation."""
        num_nodes = len(graph.nodes)
        depth = graph.get_depth()

        # Rosenbrock-like function
        a = 1
        b = 100

        x = (num_nodes - 7) / 3.0
        y = (depth - 4) / 2.0

        value = (a - x) ** 2 + b * (y - x**2) ** 2

        # Convert to fitness
        fitness = 1.0 / (1.0 + value / 50.0)

        return fitness


def get_all_problems() -> list:
    """Get all benchmark problems."""
    return [
        SimpleProblem(),
        ComplexProblem(),
        MultiModalProblem(),
        ConstrainedProblem(),
        NoisyProblem(noise_level=0.1),
        RastriginProblem(),
        RosenbrockProblem(),
    ]
