"""Tests for Genetic Algorithm optimizer."""

from morphml.core.dsl import Layer, SearchSpace
from morphml.core.graph import ModelGraph
from morphml.optimizers import GeneticAlgorithm


class TestGeneticAlgorithm:
    """Tests for GeneticAlgorithm class."""

    def create_simple_space(self) -> SearchSpace:
        """Helper to create a simple search space."""
        space = SearchSpace("test_space")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64], kernel_size=3),
            Layer.relu(),
            Layer.output(units=10),
        )
        return space

    def dummy_evaluator(self, graph: ModelGraph) -> float:
        """Simple evaluator for testing."""
        # Fitness based on number of nodes (just for testing)
        return 0.5 + 0.01 * len(graph.nodes)

    def test_create_ga(self) -> None:
        """Test GA creation."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(
            search_space=space, population_size=20, num_generations=10, mutation_rate=0.2
        )

        assert ga.config["population_size"] == 20
        assert ga.config["num_generations"] == 10
        assert ga.config["mutation_rate"] == 0.2
        assert ga.population.size() == 0  # Not initialized yet

    def test_initialize_population(self) -> None:
        """Test population initialization."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(search_space=space, population_size=10)

        ga.initialize_population()

        assert ga.population.size() == 10
        assert all(not ind.is_evaluated() for ind in ga.population)

    def test_evaluate_population(self) -> None:
        """Test population evaluation."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(search_space=space, population_size=5)

        ga.initialize_population()
        ga.evaluate_population(self.dummy_evaluator)

        # All should be evaluated
        assert all(ind.is_evaluated() for ind in ga.population)

        # Should have best individual
        assert ga.best_individual is not None
        assert ga.best_individual.fitness > 0

    def test_select_parents(self) -> None:
        """Test parent selection."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(search_space=space, population_size=10, selection_method="tournament")

        ga.initialize_population()
        ga.evaluate_population(self.dummy_evaluator)

        parents = ga.select_parents(n=4)

        assert len(parents) == 4
        assert all(p.is_evaluated() for p in parents)

    def test_crossover(self) -> None:
        """Test crossover operation."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(search_space=space, population_size=10)

        ga.initialize_population()
        ga.evaluate_population(self.dummy_evaluator)

        parents = ga.select_parents(n=2)
        offspring = ga.crossover(parents[0], parents[1])

        assert offspring is not None
        assert not offspring.is_evaluated()
        assert len(offspring.parent_ids) == 2

    def test_mutate(self) -> None:
        """Test mutation operation."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(search_space=space, mutation_rate=0.5)

        ga.initialize_population()
        individual = list(ga.population)[0]

        mutated = ga.mutate(individual)

        assert mutated is not None
        assert mutated.id != individual.id
        assert individual.id in mutated.parent_ids

    def test_generate_offspring(self) -> None:
        """Test offspring generation."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(
            search_space=space, population_size=10, mutation_rate=0.3, crossover_rate=0.7
        )

        ga.initialize_population()
        ga.evaluate_population(self.dummy_evaluator)

        offspring = ga.generate_offspring(num_offspring=5)

        assert len(offspring) == 5
        assert all(not o.is_evaluated() for o in offspring)

    def test_evolve_generation(self) -> None:
        """Test single generation evolution."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(search_space=space, population_size=10, elitism=2)

        ga.initialize_population()
        ga.evaluate_population(self.dummy_evaluator)

        initial_gen = ga.population.generation
        initial_best = ga.population.get_best(n=1)[0]

        ga.evolve_generation()

        assert ga.population.generation == initial_gen + 1
        assert ga.population.size() == 10

        # Elite should still be in population
        current_best = ga.population.get_best(n=1)[0]
        assert current_best.fitness >= initial_best.fitness

    def test_check_convergence_max_generations(self) -> None:
        """Test convergence check with max generations."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(search_space=space, num_generations=5)

        ga.initialize_population()

        # Not converged initially
        assert not ga.check_convergence()

        # Simulate reaching max generations
        ga.population.generation = 5

        assert ga.check_convergence()

    def test_check_convergence_early_stopping(self) -> None:
        """Test early stopping convergence."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(search_space=space, num_generations=100, early_stopping_patience=3)

        ga.initialize_population()
        ga._generations_without_improvement = 3

        assert ga.check_convergence()

    def test_optimize_basic(self) -> None:
        """Test basic optimization loop."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(
            search_space=space, population_size=10, num_generations=3, mutation_rate=0.2
        )

        best = ga.optimize(evaluator=self.dummy_evaluator)

        assert best is not None
        assert best.is_evaluated()
        assert best.fitness > 0
        assert ga.population.generation == 3

    def test_optimize_with_callback(self) -> None:
        """Test optimization with callback."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(search_space=space, population_size=5, num_generations=2)

        callback_calls = []

        def test_callback(generation, population):
            callback_calls.append((generation, population.size()))

        ga.optimize(evaluator=self.dummy_evaluator, callback=test_callback)

        # Callback should be called for each generation (after initial)
        assert len(callback_calls) >= 2

    def test_get_history(self) -> None:
        """Test history tracking."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(search_space=space, population_size=5, num_generations=3)

        ga.optimize(evaluator=self.dummy_evaluator)

        history = ga.get_history()

        # Should have entry for each generation
        assert len(history) >= 3

        # Each entry should have statistics
        for entry in history:
            assert "generation" in entry
            assert "best_fitness" in entry
            assert "mean_fitness" in entry
            assert "diversity" in entry

    def test_get_best_n(self) -> None:
        """Test getting top N individuals."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(search_space=space, population_size=20, num_generations=2)

        ga.optimize(evaluator=self.dummy_evaluator)

        best_5 = ga.get_best_n(n=5)

        assert len(best_5) == 5
        # Should be sorted by fitness
        for i in range(len(best_5) - 1):
            assert best_5[i].fitness >= best_5[i + 1].fitness

    def test_reset(self) -> None:
        """Test optimizer reset."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(search_space=space, population_size=10, num_generations=2)

        ga.optimize(evaluator=self.dummy_evaluator)

        assert ga.population.size() > 0
        assert len(ga.history) > 0
        assert ga.best_individual is not None

        ga.reset()

        assert ga.population.size() == 0
        assert len(ga.history) == 0
        assert ga.best_individual is None

    def test_different_selection_methods(self) -> None:
        """Test GA with different selection methods."""
        space = self.create_simple_space()

        for method in ["tournament", "roulette", "rank", "random"]:
            ga = GeneticAlgorithm(
                search_space=space,
                population_size=10,
                num_generations=2,
                selection_method=method,
            )

            best = ga.optimize(evaluator=self.dummy_evaluator)

            assert best is not None
            assert best.is_evaluated()

    def test_high_mutation_rate(self) -> None:
        """Test GA with high mutation rate."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(
            search_space=space, population_size=10, num_generations=3, mutation_rate=0.9
        )

        best = ga.optimize(evaluator=self.dummy_evaluator)

        assert best is not None
        # With high mutation, should explore well
        assert len(ga.history) >= 3

    def test_elitism_preservation(self) -> None:
        """Test that elitism preserves best individuals."""
        space = self.create_simple_space()
        ga = GeneticAlgorithm(search_space=space, population_size=10, num_generations=5, elitism=3)

        ga.initialize_population()
        ga.evaluate_population(self.dummy_evaluator)

        initial_best = ga.population.get_best(n=1)[0]
        initial_fitness = initial_best.fitness

        # Evolve several generations
        for _ in range(5):
            ga.evolve_generation()
            ga.evaluate_population(self.dummy_evaluator)

        # Best fitness should not decrease (due to elitism)
        final_best = ga.population.get_best(n=1)[0]
        assert final_best.fitness >= initial_fitness


def test_complete_optimization_workflow() -> None:
    """Integration test: Complete optimization workflow."""
    # Create search space with variety
    space = SearchSpace("integration_test")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5]),
        Layer.relu(),
        Layer.maxpool(pool_size=2),
        Layer.conv2d(filters=[64, 128], kernel_size=3),
        Layer.relu(),
        Layer.dense(units=[128, 256]),
        Layer.dropout(rate=[0.3, 0.5]),
        Layer.output(units=10),
    )

    # Create GA with reasonable parameters
    ga = GeneticAlgorithm(
        search_space=space,
        population_size=20,
        num_generations=5,
        mutation_rate=0.3,
        crossover_rate=0.7,
        elitism=3,
        selection_method="tournament",
        tournament_size=3,
    )

    # Custom evaluator with more variation
    def evaluator(graph):
        # Fitness based on graph structure
        base_fitness = 0.6
        node_bonus = 0.01 * len(graph.nodes)
        depth_bonus = 0.02 * graph.get_depth()
        return min(1.0, base_fitness + node_bonus + depth_bonus)

    # Track progress
    progress = []

    def callback(gen, pop):
        stats = pop.get_statistics()
        progress.append((gen, stats["best_fitness"], stats["mean_fitness"]))

    # Run optimization
    best = ga.optimize(evaluator=evaluator, callback=callback)

    # Verify results
    assert best is not None
    assert best.is_evaluated()
    assert best.fitness > 0.6

    # Check population evolved
    assert ga.population.generation == 5

    # Check history recorded
    history = ga.get_history()
    assert len(history) >= 5

    # Check callback was called
    assert len(progress) >= 5

    # Check best individuals
    top_5 = ga.get_best_n(n=5)
    assert len(top_5) == 5
    assert all(ind.fitness >= 0.6 for ind in top_5)

    # Verify fitness improvements or maintenance
    # (might not always improve, but shouldn't degrade with elitism)
    first_best = history[0]["best_fitness"]
    last_best = history[-1]["best_fitness"]
    assert last_best >= first_best  # Elitism guarantees this

    # Check diversity was tracked
    assert all("diversity" in h for h in history)

    # Verify architectures are valid
    for ind in top_5:
        assert ind.graph.is_valid()
        assert len(ind.graph.nodes) > 0
