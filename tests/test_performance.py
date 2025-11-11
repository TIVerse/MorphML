"""Performance tests for MorphML optimizers.

Tests throughput, latency, and scaling characteristics.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import time
from typing import Dict

import pytest

from morphml.core.dsl import Layer, SearchSpace
from morphml.core.graph import ModelGraph
from morphml.optimizers import GeneticAlgorithm, HillClimbing, RandomSearch


@pytest.fixture
def search_space() -> SearchSpace:
    """Create test search space."""
    space = SearchSpace("perf_test")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=[32, 64], kernel_size=3),
        Layer.relu(),
        Layer.maxpool(pool_size=2),
        Layer.dense(units=[64, 128]),
        Layer.output(units=10),
    )
    return space


@pytest.fixture
def fast_evaluator():
    """Fast mock evaluator for performance testing."""

    def evaluate(graph: ModelGraph) -> Dict[str, float]:
        return {"accuracy": 0.85, "latency": 10.0}

    return evaluate


class TestOptimizerThroughput:
    """Test optimizer throughput."""

    def test_random_search_throughput(self, search_space: SearchSpace, fast_evaluator):
        """Test RandomSearch can generate samples quickly."""
        optimizer = RandomSearch(search_space=search_space, num_samples=100)

        start_time = time.time()

        for _ in range(100):
            candidates = optimizer.ask()
            results = [(c, fast_evaluator(c)["accuracy"]) for c in candidates]
            optimizer.tell(results)

        elapsed = time.time() - start_time
        throughput = 100 / elapsed

        # Should generate at least 50 samples per second
        assert throughput > 50, f"Throughput too low: {throughput:.2f} samples/sec"

    def test_genetic_algorithm_throughput(self, search_space: SearchSpace, fast_evaluator):
        """Test GeneticAlgorithm generation speed."""
        optimizer = GeneticAlgorithm(
            search_space=search_space, population_size=20, num_generations=10
        )

        start_time = time.time()

        # Initialize population
        population = optimizer.initialize()

        for _generation in range(10):
            # Evaluate
            for individual in population.individuals:
                result = fast_evaluator(individual.genome)
                individual.fitness = result["accuracy"]

            # Evolve
            population = optimizer.evolve(population)

        elapsed = time.time() - start_time

        # Should complete 10 generations in reasonable time
        assert elapsed < 5.0, f"Evolution too slow: {elapsed:.2f}s for 10 generations"


class TestGraphOperations:
    """Test graph operation performance."""

    def test_graph_creation_speed(self, search_space: SearchSpace):
        """Test graph creation is fast."""
        start_time = time.time()

        [search_space.sample() for _ in range(1000)]

        elapsed = time.time() - start_time
        rate = 1000 / elapsed

        # Should create at least 100 graphs per second
        assert rate > 100, f"Graph creation too slow: {rate:.2f} graphs/sec"

    def test_graph_clone_speed(self, search_space: SearchSpace):
        """Test graph cloning is fast."""
        graph = search_space.sample()

        start_time = time.time()

        [graph.clone() for _ in range(1000)]

        elapsed = time.time() - start_time
        rate = 1000 / elapsed

        # Should clone at least 200 graphs per second
        assert rate > 200, f"Graph cloning too slow: {rate:.2f} clones/sec"

    def test_graph_mutation_speed(self):
        """Test graph mutations are fast."""
        from morphml.core.graph import GraphMutator, ModelGraph

        # Create simple graph
        graph = ModelGraph()
        input_node = graph.add_node(graph.GraphNode.create("input", {"shape": (3, 32, 32)}))
        conv = graph.add_node(graph.GraphNode.create("conv2d", {"filters": 64}))
        output = graph.add_node(graph.GraphNode.create("output", {"units": 10}))
        graph.add_edge(graph.GraphEdge(input_node, conv))
        graph.add_edge(graph.GraphEdge(conv, output))

        mutator = GraphMutator()

        start_time = time.time()

        for _ in range(100):
            mutated = graph.clone()
            mutator.mutate(mutated, mutation_rate=0.3)

        elapsed = time.time() - start_time
        rate = 100 / elapsed

        # Should mutate at least 20 graphs per second
        assert rate > 20, f"Mutation too slow: {rate:.2f} mutations/sec"


class TestMemoryUsage:
    """Test memory efficiency."""

    def test_population_memory(self, search_space: SearchSpace):
        """Test population doesn't consume excessive memory."""
        import sys

        optimizer = GeneticAlgorithm(
            search_space=search_space, population_size=100, num_generations=1
        )

        population = optimizer.initialize()

        # Rough size estimate
        size_bytes = sys.getsizeof(population)

        # Population of 100 should be < 10MB
        assert (
            size_bytes < 10 * 1024 * 1024
        ), f"Population too large: {size_bytes / 1024 / 1024:.2f} MB"

    def test_history_memory(self, search_space: SearchSpace, fast_evaluator):
        """Test history tracking doesn't grow unbounded."""
        optimizer = GeneticAlgorithm(
            search_space=search_space, population_size=20, num_generations=100
        )

        # Run many generations
        population = optimizer.initialize()
        for _generation in range(100):
            for individual in population.individuals:
                individual.fitness = fast_evaluator(individual.genome)["accuracy"]
            population = optimizer.evolve(population)

        # History should not grow linearly with generations
        assert len(optimizer.history) < 500, "History growing too large"


class TestScaling:
    """Test scaling characteristics."""

    @pytest.mark.parametrize("population_size", [10, 20, 50, 100])
    def test_scaling_with_population_size(
        self, search_space: SearchSpace, fast_evaluator, population_size: int
    ):
        """Test performance scales reasonably with population size."""
        optimizer = GeneticAlgorithm(
            search_space=search_space, population_size=population_size, num_generations=5
        )

        start_time = time.time()

        population = optimizer.initialize()
        for _generation in range(5):
            for individual in population.individuals:
                individual.fitness = fast_evaluator(individual.genome)["accuracy"]
            population = optimizer.evolve(population)

        elapsed = time.time() - start_time
        time_per_individual = elapsed / (population_size * 5)

        # Should be roughly constant time per individual
        assert (
            time_per_individual < 0.1
        ), f"Scaling issue: {time_per_individual:.4f}s per individual"

    def test_search_space_complexity(self):
        """Test performance with varying search space complexity."""
        # Simple space
        simple_space = SearchSpace("simple")
        simple_space.add_layers(
            Layer.input(shape=(3, 32, 32)), Layer.dense(units=[64, 128]), Layer.output(units=10)
        )

        # Complex space
        complex_space = SearchSpace("complex")
        complex_space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64, 128, 256], kernel_size=[3, 5, 7]),
            Layer.relu(),
            Layer.batchnorm(),
            Layer.maxpool(pool_size=[2, 3]),
            Layer.conv2d(filters=[64, 128, 256, 512], kernel_size=[3, 5]),
            Layer.relu(),
            Layer.batchnorm(),
            Layer.maxpool(pool_size=2),
            Layer.flatten(),
            Layer.dense(units=[256, 512, 1024, 2048]),
            Layer.dropout(rate=[0.2, 0.3, 0.5]),
            Layer.dense(units=[128, 256, 512]),
            Layer.output(units=10),
        )

        # Sample from both
        start_simple = time.time()
        [simple_space.sample() for _ in range(100)]
        time_simple = time.time() - start_simple

        start_complex = time.time()
        [complex_space.sample() for _ in range(100)]
        time_complex = time.time() - start_complex

        # Complex space should not be more than 10x slower
        ratio = time_complex / time_simple
        assert ratio < 10, f"Complex space too slow: {ratio:.2f}x slower"


class TestConcurrency:
    """Test concurrent execution (if available)."""

    def test_parallel_evaluation(self, search_space: SearchSpace, fast_evaluator):
        """Test parallel evaluation of candidates."""
        from concurrent.futures import ThreadPoolExecutor

        candidates = [search_space.sample() for _ in range(20)]

        # Sequential
        start_seq = time.time()
        [fast_evaluator(c) for c in candidates]
        time_seq = time.time() - start_seq

        # Parallel
        start_par = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(fast_evaluator, candidates))
        time_par = time.time() - start_par

        # Parallel should be faster (or at least not much slower)
        assert time_par <= time_seq * 1.5, f"Parallel slower: {time_par:.2f}s vs {time_seq:.2f}s"


@pytest.mark.slow
class TestStressTest:
    """Stress tests for extreme conditions."""

    def test_large_population(self, search_space: SearchSpace, fast_evaluator):
        """Test with very large population."""
        optimizer = GeneticAlgorithm(
            search_space=search_space, population_size=500, num_generations=10
        )

        start_time = time.time()

        population = optimizer.initialize()
        assert len(population.individuals) == 500

        # Run a few generations
        for _generation in range(5):
            for individual in population.individuals:
                individual.fitness = fast_evaluator(individual.genome)["accuracy"]
            population = optimizer.evolve(population)

        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 60, f"Large population too slow: {elapsed:.2f}s"

    def test_many_iterations(self, search_space: SearchSpace, fast_evaluator):
        """Test long-running optimization."""
        optimizer = HillClimbing(search_space=search_space, max_iterations=1000)

        start_time = time.time()

        current = search_space.sample()
        best_fitness = fast_evaluator(current)["accuracy"]

        for _ in range(1000):
            neighbor = optimizer._generate_neighbor(current)
            fitness = fast_evaluator(neighbor)["accuracy"]

            if fitness > best_fitness:
                current = neighbor
                best_fitness = fitness

        elapsed = time.time() - start_time

        # Should complete 1000 iterations quickly
        assert elapsed < 30, f"1000 iterations too slow: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
