"""Comprehensive tests for all optimizers."""


from morphml.core.dsl import Layer, SearchSpace, create_cnn_space
from morphml.core.graph import ModelGraph
from morphml.evaluation import HeuristicEvaluator
from morphml.optimizers import (
    NSGA2,
    DifferentialEvolution,
    GeneticAlgorithm,
    HillClimbing,
    RandomSearch,
    SimulatedAnnealing,
)


class TestOptimizerBasics:
    """Test basic functionality of all optimizers."""

    def create_simple_space(self) -> SearchSpace:
        """Create simple search space for testing."""
        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64], kernel_size=3),
            Layer.relu(),
            Layer.output(units=10),
        )
        return space

    def simple_evaluator(self, graph: ModelGraph) -> float:
        """Simple evaluator for testing."""
        return 0.5 + 0.01 * len(graph.nodes)

    def test_random_search_basic(self) -> None:
        """Test RandomSearch creation and basic optimization."""
        space = self.create_simple_space()
        rs = RandomSearch(space, num_samples=10)

        best = rs.optimize(self.simple_evaluator)

        assert best is not None
        assert best.is_evaluated()
        assert len(rs.get_all_evaluated()) == 10

    def test_hill_climbing_basic(self) -> None:
        """Test HillClimbing creation and optimization."""
        space = self.create_simple_space()
        hc = HillClimbing(space, max_iterations=20, patience=5)

        best = hc.optimize(self.simple_evaluator)

        assert best is not None
        assert best.is_evaluated()
        assert len(hc.get_history()) > 0

    def test_simulated_annealing_basic(self) -> None:
        """Test SimulatedAnnealing creation and optimization."""
        space = self.create_simple_space()
        sa = SimulatedAnnealing(space, max_iterations=20, initial_temp=10.0)

        best = sa.optimize(self.simple_evaluator)

        assert best is not None
        assert best.is_evaluated()
        assert len(sa.get_history()) > 0

    def test_differential_evolution_basic(self) -> None:
        """Test DifferentialEvolution creation and optimization."""
        space = self.create_simple_space()
        de = DifferentialEvolution(space, population_size=10, num_generations=3)

        best = de.optimize(self.simple_evaluator)

        assert best is not None
        assert best.is_evaluated()

    def test_genetic_algorithm_basic(self) -> None:
        """Test GeneticAlgorithm with default settings."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(space, population_size=10, num_generations=3)

        best = ga.optimize(self.simple_evaluator)

        assert best is not None
        assert best.is_evaluated()

    def test_nsga2_basic(self) -> None:
        """Test NSGA2 multi-objective optimizer."""
        space = self.create_simple_space()
        nsga = NSGA2(space, objectives=["fitness", "params"], population_size=10, num_generations=3)

        def multi_eval(graph: ModelGraph) -> dict:
            return {"fitness": 0.5 + 0.01 * len(graph.nodes), "params": graph.estimate_parameters()}

        pareto = nsga.optimize(multi_eval)

        assert pareto is not None
        assert len(pareto) > 0


class TestOptimizerComparison:
    """Compare all optimizers on same task."""

    def create_test_space(self) -> SearchSpace:
        """Create test search space."""
        space = SearchSpace("comparison_test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5]),
            Layer.relu(),
            Layer.maxpool(pool_size=2),
            Layer.dense(units=[128, 256]),
            Layer.output(units=10),
        )
        return space

    def test_optimizer_comparison(self) -> None:
        """Compare all optimizers on same problem."""
        space = self.create_test_space()
        evaluator = HeuristicEvaluator()

        results = {}

        # Random Search
        rs = RandomSearch(space, num_samples=20)
        rs_best = rs.optimize(evaluator)
        results["RandomSearch"] = rs_best.fitness

        # Hill Climbing
        hc = HillClimbing(space, max_iterations=20)
        hc_best = hc.optimize(evaluator)
        results["HillClimbing"] = hc_best.fitness

        # Simulated Annealing
        sa = SimulatedAnnealing(space, max_iterations=20)
        sa_best = sa.optimize(evaluator)
        results["SimulatedAnnealing"] = sa_best.fitness

        # Genetic Algorithm
        ga = GeneticAlgorithm(space, population_size=10, num_generations=3)
        ga_best = ga.optimize(evaluator)
        results["GeneticAlgorithm"] = ga_best.fitness

        # All should find reasonable solutions
        for name, fitness in results.items():
            assert fitness > 0.5, f"{name} fitness too low: {fitness}"


class TestOptimizerReset:
    """Test reset functionality."""

    def create_space(self) -> SearchSpace:
        """Create space."""
        return SearchSpace("test").add_layers(
            Layer.input(shape=(3, 32, 32)), Layer.conv2d(filters=64), Layer.output(units=10)
        )

    def evaluator(self, graph: ModelGraph) -> float:
        """Simple evaluator."""
        return 0.7

    def test_random_search_reset(self) -> None:
        """Test RandomSearch reset."""
        rs = RandomSearch(self.create_space(), num_samples=5)
        rs.optimize(self.evaluator)

        assert len(rs.evaluated) > 0

        rs.reset()

        assert len(rs.evaluated) == 0
        assert rs.best_individual is None

    def test_hill_climbing_reset(self) -> None:
        """Test HillClimbing reset."""
        hc = HillClimbing(self.create_space(), max_iterations=10)
        hc.optimize(self.evaluator)

        assert hc.current is not None

        hc.reset()

        assert hc.current is None
        assert len(hc.history) == 0

    def test_simulated_annealing_reset(self) -> None:
        """Test SimulatedAnnealing reset."""
        sa = SimulatedAnnealing(self.create_space(), max_iterations=10)
        sa.optimize(self.evaluator)

        assert sa.best is not None

        sa.reset()

        assert sa.best is None
        assert len(sa.history) == 0

    def test_genetic_algorithm_reset(self) -> None:
        """Test GeneticAlgorithm reset."""
        ga = GeneticAlgorithm(self.create_space(), population_size=5, num_generations=2)
        ga.optimize(self.evaluator)

        assert ga.population.size() > 0

        ga.reset()

        assert ga.population.size() == 0


class TestOptimizerHistory:
    """Test history tracking."""

    def create_space(self) -> SearchSpace:
        """Create space."""
        return SearchSpace("test").add_layers(
            Layer.input(shape=(3, 32, 32)), Layer.conv2d(filters=64), Layer.output(units=10)
        )

    def evaluator(self, graph: ModelGraph) -> float:
        """Simple evaluator."""
        return 0.75 + 0.01 * len(graph.nodes)

    def test_hill_climbing_history(self) -> None:
        """Test HillClimbing history tracking."""
        hc = HillClimbing(self.create_space(), max_iterations=15)
        hc.optimize(self.evaluator)

        history = hc.get_history()

        assert len(history) == 15
        assert all(isinstance(f, float) for f in history)

    def test_simulated_annealing_history(self) -> None:
        """Test SimulatedAnnealing history."""
        sa = SimulatedAnnealing(self.create_space(), max_iterations=15)
        sa.optimize(self.evaluator)

        history = sa.get_history()

        assert len(history) == 16  # Including initial
        assert all("iteration" in h for h in history)
        assert all("temperature" in h for h in history)
        assert all("accepted" in h for h in history)

    def test_genetic_algorithm_history(self) -> None:
        """Test GeneticAlgorithm history."""
        ga = GeneticAlgorithm(self.create_space(), population_size=5, num_generations=3)
        ga.optimize(self.evaluator)

        history = ga.get_history()

        assert len(history) >= 3
        assert all("generation" in h for h in history)
        assert all("best_fitness" in h for h in history)


class TestOptimizerCallbacks:
    """Test callback functionality."""

    def create_space(self) -> SearchSpace:
        """Create space."""
        return SearchSpace("test").add_layers(
            Layer.input(shape=(3, 32, 32)), Layer.conv2d(filters=64), Layer.output(units=10)
        )

    def evaluator(self, graph: ModelGraph) -> float:
        """Simple evaluator."""
        return 0.8

    def test_genetic_algorithm_callback(self) -> None:
        """Test GA with callback."""
        ga = GeneticAlgorithm(self.create_space(), population_size=5, num_generations=3)

        callback_calls = []

        def callback(gen, pop):
            callback_calls.append((gen, pop.size()))

        ga.optimize(self.evaluator, callback=callback)

        assert len(callback_calls) >= 3
        assert all(isinstance(gen, int) for gen, _ in callback_calls)

    def test_differential_evolution_callback(self) -> None:
        """Test DE with callback."""
        de = DifferentialEvolution(self.create_space(), population_size=5, num_generations=3)

        callback_calls = []

        def callback(gen, pop):
            callback_calls.append(gen)

        de.optimize(self.evaluator, callback=callback)

        assert len(callback_calls) >= 3


class TestOptimizerStress:
    """Stress tests for optimizers."""

    def test_large_population_ga(self) -> None:
        """Test GA with large population."""
        space = SearchSpace("test").add_layers(
            Layer.input(shape=(3, 32, 32)), Layer.conv2d(filters=32), Layer.output(units=10)
        )

        ga = GeneticAlgorithm(space, population_size=50, num_generations=2)
        evaluator = HeuristicEvaluator()

        best = ga.optimize(evaluator)

        assert best is not None
        assert ga.population.size() <= 50

    def test_many_iterations_sa(self) -> None:
        """Test SA with many iterations."""
        space = SearchSpace("test").add_layers(
            Layer.input(shape=(3, 32, 32)), Layer.conv2d(filters=32), Layer.output(units=10)
        )

        sa = SimulatedAnnealing(space, max_iterations=100)
        evaluator = HeuristicEvaluator()

        best = sa.optimize(evaluator)

        assert best is not None
        assert len(sa.history) == 101  # +1 for initial

    def test_many_samples_random_search(self) -> None:
        """Test Random Search with many samples."""
        space = SearchSpace("test").add_layers(
            Layer.input(shape=(3, 32, 32)), Layer.conv2d(filters=32), Layer.output(units=10)
        )

        rs = RandomSearch(space, num_samples=100)
        evaluator = HeuristicEvaluator()

        best = rs.optimize(evaluator)

        assert best is not None
        assert len(rs.evaluated) == 100


class TestOptimizerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_search_space(self) -> None:
        """Test with minimal search space."""
        space = SearchSpace("minimal")
        space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))

        rs = RandomSearch(space, num_samples=5)
        evaluator = HeuristicEvaluator()

        best = rs.optimize(evaluator)

        assert best is not None

    def test_zero_fitness(self) -> None:
        """Test with zero fitness evaluator."""
        space = SearchSpace("test").add_layers(
            Layer.input(shape=(3, 32, 32)), Layer.conv2d(filters=32), Layer.output(units=10)
        )

        def zero_eval(graph):
            return 0.0

        hc = HillClimbing(space, max_iterations=10, patience=5)
        best = hc.optimize(zero_eval)

        assert best.fitness == 0.0

    def test_constant_fitness(self) -> None:
        """Test with constant fitness."""
        space = SearchSpace("test").add_layers(
            Layer.input(shape=(3, 32, 32)), Layer.conv2d(filters=32), Layer.output(units=10)
        )

        def constant_eval(graph):
            return 0.5

        ga = GeneticAlgorithm(space, population_size=5, num_generations=3)
        best = ga.optimize(constant_eval)

        assert best.fitness == 0.5


def test_optimizer_integration() -> None:
    """Integration test combining multiple optimizers."""
    # Create search space
    space = create_cnn_space(num_classes=10)
    evaluator = HeuristicEvaluator()

    # Stage 1: Random sampling
    rs = RandomSearch(space, num_samples=20)
    candidates = rs.optimize(evaluator)

    # Stage 2: Evolutionary refinement
    ga = GeneticAlgorithm(space, population_size=10, num_generations=5)
    ga_best = ga.optimize(evaluator)

    # Stage 3: Local search
    hc = HillClimbing(space, max_iterations=10)
    final_best = hc.optimize(evaluator)

    # All stages should produce valid results
    assert candidates.is_evaluated()
    assert ga_best.is_evaluated()
    assert final_best.is_evaluated()
