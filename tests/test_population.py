"""Tests for population management system."""

from morphml.core.dsl import Layer, SearchSpace
from morphml.core.graph import GraphEdge, GraphNode, ModelGraph
from morphml.core.search import Individual, Population


class TestIndividual:
    """Tests for Individual class."""

    def create_sample_graph(self) -> ModelGraph:
        """Helper to create a sample graph."""
        graph = ModelGraph()
        input_node = GraphNode.create("input", {"shape": (3, 32, 32)})
        conv = GraphNode.create("conv2d", {"filters": 64})
        output = GraphNode.create("dense", {"units": 10})

        graph.add_node(input_node)
        graph.add_node(conv)
        graph.add_node(output)

        graph.add_edge(GraphEdge(input_node, conv))
        graph.add_edge(GraphEdge(conv, output))

        return graph

    def test_create_individual(self) -> None:
        """Test individual creation."""
        graph = self.create_sample_graph()
        individual = Individual(graph)

        assert individual.graph == graph
        assert individual.fitness is None
        assert not individual.is_evaluated()
        assert individual.age == 0

    def test_individual_with_fitness(self) -> None:
        """Test individual creation with fitness."""
        graph = self.create_sample_graph()
        individual = Individual(graph, fitness=0.95)

        assert individual.fitness == 0.95
        assert individual.is_evaluated()

    def test_set_fitness(self) -> None:
        """Test setting fitness."""
        graph = self.create_sample_graph()
        individual = Individual(graph)

        assert not individual.is_evaluated()

        individual.set_fitness(0.92, accuracy=0.92, loss=0.08)

        assert individual.is_evaluated()
        assert individual.fitness == 0.92
        assert individual.get_metric("accuracy") == 0.92
        assert individual.get_metric("loss") == 0.08

    def test_increment_age(self) -> None:
        """Test age increment."""
        graph = self.create_sample_graph()
        individual = Individual(graph)

        assert individual.age == 0

        individual.increment_age()
        assert individual.age == 1

        individual.increment_age()
        assert individual.age == 2

    def test_clone(self) -> None:
        """Test individual cloning."""
        graph = self.create_sample_graph()
        original = Individual(graph, fitness=0.90)
        original.metadata["test_key"] = "test_value"

        # Clone without fitness
        clone1 = original.clone(keep_fitness=False)
        assert clone1.id != original.id
        assert clone1.fitness is None
        assert clone1.age == 0
        assert clone1.metadata["test_key"] == "test_value"
        assert original.id in clone1.parent_ids

        # Clone with fitness
        clone2 = original.clone(keep_fitness=True)
        assert clone2.fitness == 0.90

    def test_individual_comparison(self) -> None:
        """Test individual comparison."""
        graph1 = self.create_sample_graph()
        graph2 = self.create_sample_graph()

        ind1 = Individual(graph1, fitness=0.95)
        ind2 = Individual(graph2, fitness=0.90)
        ind3 = Individual(graph1, fitness=0.98)

        # Less than comparison (for sorting)
        assert ind2 < ind1
        assert ind1 < ind3
        assert not (ind3 < ind1)

    def test_individual_serialization(self) -> None:
        """Test individual to_dict and from_dict."""
        graph = self.create_sample_graph()
        original = Individual(graph, fitness=0.88)
        original.metadata["accuracy"] = 0.88

        # Serialize
        data = original.to_dict()

        # Deserialize
        restored = Individual.from_dict(data)

        assert restored.id == original.id
        assert restored.fitness == original.fitness
        assert restored.get_metric("accuracy") == 0.88

    def test_dominates(self) -> None:
        """Test multi-objective dominance (higher is better)."""
        graph1 = self.create_sample_graph()
        graph2 = self.create_sample_graph()

        ind1 = Individual(graph1, fitness=0.95)
        ind1.metadata["accuracy"] = 0.95
        ind1.metadata["precision"] = 0.94

        ind2 = Individual(graph2, fitness=0.90)
        ind2.metadata["accuracy"] = 0.90
        ind2.metadata["precision"] = 0.89

        # ind1 dominates ind2 (better in both objectives, assuming higher is better)
        assert ind1.dominates(ind2, ["accuracy", "precision"])
        assert not ind2.dominates(ind1, ["accuracy", "precision"])


class TestPopulation:
    """Tests for Population class."""

    def create_sample_individual(self, fitness: float = 0.85) -> Individual:
        """Helper to create a sample individual."""
        space = SearchSpace()
        space.add_layers(
            Layer.input(shape=(3, 32, 32)), Layer.conv2d(filters=64), Layer.output(units=10)
        )
        graph = space.sample()
        return Individual(graph, fitness=fitness)

    def test_create_population(self) -> None:
        """Test population creation."""
        pop = Population(max_size=50, elitism=5)

        assert pop.max_size == 50
        assert pop.elitism == 5
        assert pop.size() == 0
        assert pop.generation == 0

    def test_add_individuals(self) -> None:
        """Test adding individuals."""
        pop = Population(max_size=10)

        ind1 = self.create_sample_individual(0.90)
        ind2 = self.create_sample_individual(0.85)

        pop.add(ind1)
        pop.add(ind2)

        assert pop.size() == 2

    def test_add_many(self) -> None:
        """Test adding multiple individuals."""
        pop = Population(max_size=10)

        individuals = [self.create_sample_individual(0.8 + i * 0.01) for i in range(5)]

        pop.add_many(individuals)

        assert pop.size() == 5

    def test_get_best(self) -> None:
        """Test getting best individuals."""
        pop = Population(max_size=10)

        # Add individuals with different fitnesses
        fitnesses = [0.85, 0.92, 0.78, 0.95, 0.88]
        for f in fitnesses:
            pop.add(self.create_sample_individual(f))

        # Get top 3
        best = pop.get_best(n=3)

        assert len(best) == 3
        assert best[0].fitness == 0.95
        assert best[1].fitness == 0.92
        assert best[2].fitness == 0.88

    def test_get_worst(self) -> None:
        """Test getting worst individuals."""
        pop = Population(max_size=10)

        fitnesses = [0.85, 0.92, 0.78, 0.95, 0.88]
        for f in fitnesses:
            pop.add(self.create_sample_individual(f))

        worst = pop.get_worst(n=2)

        assert len(worst) == 2
        assert worst[0].fitness == 0.78
        assert worst[1].fitness == 0.85

    def test_get_unevaluated(self) -> None:
        """Test getting unevaluated individuals."""
        pop = Population(max_size=10)

        # Add evaluated
        pop.add(self.create_sample_individual(0.90))

        # Add unevaluated
        space = SearchSpace()
        space.add_layer(Layer.conv2d(filters=64))
        graph = space.sample()
        unevaluated = Individual(graph)  # No fitness
        pop.add(unevaluated)

        unevaluated_list = pop.get_unevaluated()

        assert len(unevaluated_list) == 1
        assert unevaluated_list[0] == unevaluated

    def test_tournament_selection(self) -> None:
        """Test tournament selection."""
        pop = Population(max_size=20)

        # Add individuals
        for i in range(10):
            pop.add(self.create_sample_individual(0.70 + i * 0.02))

        # Select parents
        selected = pop.select(n=5, method="tournament", k=3)

        assert len(selected) == 5
        # Should prefer higher fitness
        mean_fitness = sum(ind.fitness for ind in selected) / len(selected)
        assert mean_fitness > 0.75

    def test_roulette_selection(self) -> None:
        """Test roulette wheel selection."""
        pop = Population(max_size=20)

        for i in range(10):
            pop.add(self.create_sample_individual(0.70 + i * 0.02))

        selected = pop.select(n=5, method="roulette")

        assert len(selected) == 5

    def test_rank_selection(self) -> None:
        """Test rank-based selection."""
        pop = Population(max_size=20)

        for i in range(10):
            pop.add(self.create_sample_individual(0.70 + i * 0.02))

        selected = pop.select(n=5, method="rank")

        assert len(selected) == 5

    def test_random_selection(self) -> None:
        """Test random selection."""
        pop = Population(max_size=20)

        for i in range(10):
            pop.add(self.create_sample_individual(0.70 + i * 0.02))

        selected = pop.select(n=5, method="random")

        assert len(selected) == 5

    def test_trim_population(self) -> None:
        """Test population trimming."""
        pop = Population(max_size=10, elitism=2)

        # Add 15 individuals
        for i in range(15):
            pop.add(self.create_sample_individual(0.70 + i * 0.01))

        assert pop.size() == 15

        # Trim to max_size
        pop.trim()

        assert pop.size() == 10

        # Best individuals should be kept
        best = pop.get_best(n=1)
        assert best[0].fitness >= 0.83  # Should be from the top

    def test_next_generation(self) -> None:
        """Test advancing to next generation."""
        pop = Population(max_size=10)

        for i in range(5):
            pop.add(self.create_sample_individual(0.80 + i * 0.02))

        assert pop.generation == 0

        # Check initial ages
        for ind in pop.individuals:
            assert ind.age == 0

        # Advance generation
        pop.next_generation()

        assert pop.generation == 1

        # Ages should be incremented
        for ind in pop.individuals:
            assert ind.age == 1

        # History should be recorded
        assert len(pop.history) == 1

    def test_get_statistics(self) -> None:
        """Test population statistics."""
        pop = Population(max_size=10)

        fitnesses = [0.85, 0.92, 0.78, 0.95, 0.88]
        for f in fitnesses:
            pop.add(self.create_sample_individual(f))

        stats = pop.get_statistics()

        assert stats["size"] == 5
        assert stats["evaluated"] == 5
        assert stats["best_fitness"] == 0.95
        assert stats["worst_fitness"] == 0.78
        assert stats["mean_fitness"] == sum(fitnesses) / len(fitnesses)

    def test_get_diversity(self) -> None:
        """Test diversity calculation."""
        pop = Population(max_size=10)

        # Add individuals
        space = SearchSpace()
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=[32, 64, 128]),  # Variation
            Layer.output(units=10),
        )

        # Sample different architectures
        for _ in range(5):
            graph = space.sample()
            ind = Individual(graph, fitness=0.85)
            pop.add(ind)

        diversity = pop.get_diversity(method="hash")

        # Should have some diversity
        assert 0 <= diversity <= 1
        assert diversity > 0  # Should have at least some variation

    def test_population_iteration(self) -> None:
        """Test iterating over population."""
        pop = Population(max_size=10)

        individuals = [self.create_sample_individual(0.80 + i * 0.01) for i in range(5)]
        pop.add_many(individuals)

        # Test __len__
        assert len(pop) == 5

        # Test __iter__
        count = 0
        for ind in pop:
            assert isinstance(ind, Individual)
            count += 1

        assert count == 5


def test_population_workflow() -> None:
    """Integration test: Complete population workflow."""
    # Create search space
    space = SearchSpace("test_space")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=[32, 64], kernel_size=[3, 5]),
        Layer.relu(),
        Layer.output(units=10),
    )

    # Initialize population
    pop = Population(max_size=20, elitism=3)

    # Sample initial population
    for _ in range(10):
        graph = space.sample()
        individual = Individual(graph)
        # Simulate evaluation
        individual.set_fitness(0.70 + 0.01 * len(graph.nodes), accuracy=0.85)
        pop.add(individual)

    assert pop.size() == 10

    # Get statistics
    stats = pop.get_statistics()
    assert stats["evaluated"] == 10
    assert stats["best_fitness"] > 0.70

    # Select parents
    parents = pop.select(n=6, method="tournament", k=3)
    assert len(parents) == 6

    # Create offspring (simulate)
    offspring = []
    for parent in parents[:4]:
        child = parent.clone(keep_fitness=False)
        child.set_fitness(0.75 + 0.02 * (len(parents) - parents.index(parent)))
        offspring.append(child)

    # Add offspring
    pop.add_many(offspring)
    assert pop.size() == 14

    # Trim to max size (shouldn't reduce since we're under max_size)
    pop.trim()
    assert pop.size() == 14  # Still 14 since below max_size

    # Advance generation
    pop.next_generation()
    assert pop.generation == 1

    # Check diversity
    diversity = pop.get_diversity()
    assert diversity > 0

    # Get best
    best = pop.get_best(n=3)
    assert len(best) == 3
    assert best[0].fitness >= best[1].fitness
    assert best[1].fitness >= best[2].fitness
