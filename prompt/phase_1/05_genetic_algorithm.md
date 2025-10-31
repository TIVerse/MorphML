# Component 5: Genetic Algorithm Optimizer

**Duration:** Weeks 6-7  
**LOC Target:** ~3,000  
**Dependencies:** Components 1-4

---

## 🎯 Objective

Implement a complete genetic algorithm for neural architecture search with selection, crossover, mutation, and evolution loop.

---

## 📋 Files to Create

### 1. `base.py` (~300 LOC)

**`BaseOptimizer` abstract class:**

```python
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """
    Base class for all optimization algorithms.
    
    All optimizers follow the same interface:
    1. Initialize with search space and configuration
    2. Generate candidate architectures
    3. Update based on evaluation results
    4. Track optimization progress
    """
    
    def __init__(self, search_space: SearchSpace, config: Dict[str, Any]):
        self.search_space = search_space
        self.config = config
        self.generation = 0
        self.history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def initialize(self) -> Population:
        """Create initial population."""
    
    @abstractmethod
    def ask(self) -> List[ModelGraph]:
        """Generate candidate architectures for evaluation."""
    
    @abstractmethod
    def tell(self, results: List[Tuple[ModelGraph, float]]) -> None:
        """Update optimizer with evaluation results."""
    
    @abstractmethod
    def should_stop(self) -> bool:
        """Check termination criteria."""
    
    def get_best(self) -> ModelGraph:
        """Return best architecture found so far."""
        if not self.history:
            raise OptimizerError("No evaluations performed yet")
        
        best_entry = max(self.history, key=lambda x: x['fitness'])
        return best_entry['genome']
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.history:
            return {}
        
        fitnesses = [entry['fitness'] for entry in self.history]
        return {
            'generation': self.generation,
            'num_evaluations': len(self.history),
            'best_fitness': max(fitnesses),
            'mean_fitness': sum(fitnesses) / len(fitnesses),
            'fitness_std': self._std(fitnesses)
        }
    
    def _std(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
```

---

### 2. `genetic.py` (~1,200 LOC)

**`GeneticAlgorithm` class:**

```python
class GeneticAlgorithm(BaseOptimizer):
    """
    Genetic algorithm for neural architecture search.
    
    Evolves population of architectures through:
    - Selection: Choose parents based on fitness
    - Crossover: Combine parent graphs
    - Mutation: Modify graph structure
    - Elitism: Preserve best individuals
    
    Config parameters:
        population_size: Number of individuals
        elite_size: Number of top individuals to preserve
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        selection_strategy: 'tournament', 'roulette', 'rank'
        tournament_size: Size for tournament selection
    """
    
    def __init__(self, search_space: SearchSpace, config: Dict[str, Any]):
        super().__init__(search_space, config)
        
        # Extract config with defaults
        self.population_size = config.get('population_size', 50)
        self.elite_size = config.get('elite_size', 5)
        self.mutation_rate = config.get('mutation_rate', 0.15)
        self.crossover_rate = config.get('crossover_rate', 0.7)
        self.max_generations = config.get('max_generations', 100)
        
        # Selection strategy
        selection_type = config.get('selection_strategy', 'tournament')
        if selection_type == 'tournament':
            self.selector = TournamentSelection(config.get('tournament_size', 3))
        elif selection_type == 'roulette':
            self.selector = RouletteWheelSelection()
        elif selection_type == 'rank':
            self.selector = RankSelection()
        else:
            raise ValueError(f"Unknown selection strategy: {selection_type}")
        
        # Mutation operator
        self.mutator = MutationSelector(
            mutations=[
                AddNodeMutation(['conv2d', 'dense', 'dropout']),
                RemoveNodeMutation(),
                ModifyNodeMutation()
            ],
            probabilities=[0.4, 0.3, 0.3]
        )
        
        self.population: Optional[Population] = None
    
    def initialize(self) -> Population:
        """Create random initial population."""
        individuals = []
        for _ in range(self.population_size):
            genome = self.search_space.sample()
            individual = Individual(genome=genome)
            individuals.append(individual)
        
        self.population = Population(individuals)
        logger.info(f"Initialized population with {self.population_size} individuals")
        return self.population
    
    def ask(self) -> List[ModelGraph]:
        """Return current population's genomes for evaluation."""
        if self.population is None:
            raise OptimizerError("Must call initialize() first")
        
        return [ind.genome for ind in self.population.individuals if ind.fitness is None]
    
    def tell(self, results: List[Tuple[ModelGraph, float]]) -> None:
        """
        Update population with evaluation results.
        
        Args:
            results: List of (genome, fitness) tuples
        """
        # Update fitnesses
        results_dict = {id(genome): fitness for genome, fitness in results}
        for individual in self.population.individuals:
            genome_id = id(individual.genome)
            if genome_id in results_dict:
                individual.fitness = results_dict[genome_id]
        
        # Record history
        for individual in self.population.individuals:
            if individual.fitness is not None:
                self.history.append({
                    'generation': self.generation,
                    'genome': individual.genome,
                    'fitness': individual.fitness
                })
    
    def should_stop(self) -> bool:
        """Check if evolution should terminate."""
        return self.generation >= self.max_generations
    
    def evolve(self) -> Population:
        """
        Perform one generation of evolution.
        
        Steps:
        1. Select elite individuals (carry over unchanged)
        2. Select parents for reproduction
        3. Apply crossover to create offspring
        4. Apply mutation to offspring
        5. Form new population
        """
        if self.population is None:
            raise OptimizerError("Must initialize population first")
        
        # Sort by fitness
        self.population.sort_by_fitness()
        
        # Elitism: preserve best individuals
        elite = self.population.get_best(self.elite_size)
        
        # Create offspring
        offspring = []
        while len(offspring) < self.population_size - self.elite_size:
            # Select parents
            parent1, parent2 = self.selector.select(self.population, 2)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1_genome, child2_genome = self._crossover(parent1.genome, parent2.genome)
            else:
                child1_genome = parent1.genome.clone()
                child2_genome = parent2.genome.clone()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1_genome = self.mutator.mutate(child1_genome)
            if random.random() < self.mutation_rate:
                child2_genome = self.mutator.mutate(child2_genome)
            
            offspring.append(Individual(child1_genome))
            if len(offspring) < self.population_size - self.elite_size:
                offspring.append(Individual(child2_genome))
        
        # Form new population
        new_population = Population(elite + offspring)
        new_population.generation = self.generation + 1
        
        self.population = new_population
        self.generation += 1
        
        return self.population
    
    def _crossover(self, graph1: ModelGraph, graph2: ModelGraph) -> Tuple[ModelGraph, ModelGraph]:
        """
        Perform crossover between two graphs.
        
        Strategy: Single-point crossover at random depth level.
        1. Find common depth level
        2. Split graphs at that level
        3. Recombine bottom of graph1 with top of graph2 and vice versa
        """
        # For Phase 1, use simple uniform crossover: swap random subgraphs
        child1 = graph1.clone()
        child2 = graph2.clone()
        
        # Get modifiable nodes (exclude input/output)
        nodes1 = [nid for nid in child1.nodes if nid not in (child1.input_node_id, child1.output_node_id)]
        nodes2 = [nid for nid in child2.nodes if nid not in (child2.input_node_id, child2.output_node_id)]
        
        if not nodes1 or not nodes2:
            return child1, child2
        
        # Swap random nodes
        num_swaps = min(len(nodes1), len(nodes2)) // 2
        for _ in range(num_swaps):
            if random.random() < 0.5:
                node1_id = random.choice(nodes1)
                node2_id = random.choice(nodes2)
                
                # Swap node operations and parameters
                node1 = child1.nodes[node1_id]
                node2 = child2.nodes[node2_id]
                
                node1.operation, node2.operation = node2.operation, node1.operation
                node1.params, node2.params = node2.params.copy(), node1.params.copy()
        
        return child1, child2
```

---

### 3. `operators/mutation.py` (~800 LOC)

**Graph-level mutation operators** (already in graph/mutations.py, can import):

```python
# Re-export from graph module
from morphml.core.graph.mutations import (
    GraphMutation,
    AddNodeMutation,
    RemoveNodeMutation,
    ModifyNodeMutation,
    MutationSelector
)

# Additional GA-specific mutations

class AdaptiveMutation(GraphMutation):
    """
    Adaptive mutation that adjusts rate based on diversity.
    
    If population diversity is low, increase mutation rate to explore.
    If diversity is high, decrease mutation rate to exploit.
    """
    
    def __init__(self, base_mutation: GraphMutation, min_rate: float = 0.05, max_rate: float = 0.5):
        self.base_mutation = base_mutation
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.current_rate = (min_rate + max_rate) / 2
    
    def update_rate(self, diversity: float) -> None:
        """
        Update mutation rate based on diversity metric (0-1).
        
        Low diversity → increase rate
        High diversity → decrease rate
        """
        self.current_rate = self.min_rate + (1 - diversity) * (self.max_rate - self.min_rate)
    
    def is_applicable(self, graph: ModelGraph) -> bool:
        return self.base_mutation.is_applicable(graph)
    
    def mutate(self, graph: ModelGraph) -> ModelGraph:
        if random.random() < self.current_rate:
            return self.base_mutation.mutate(graph)
        return graph.clone()
```

---

### 4. `operators/crossover.py` (~800 LOC)

**`Crossover` base class:**
```python
class Crossover(ABC):
    """Base class for crossover operators."""
    
    @abstractmethod
    def crossover(self, graph1: ModelGraph, graph2: ModelGraph) -> Tuple[ModelGraph, ModelGraph]:
        """Perform crossover and return two offspring."""
```

**`UniformCrossover`:**
```python
class UniformCrossover(Crossover):
    """
    Uniform crossover: each node has 50% chance to come from parent1 or parent2.
    """
    
    def crossover(self, graph1: ModelGraph, graph2: ModelGraph) -> Tuple[ModelGraph, ModelGraph]:
        child1 = graph1.clone()
        child2 = graph2.clone()
        
        # Get alignable nodes (same position in execution order)
        topo1 = graph1.topological_sort()
        topo2 = graph2.topological_sort()
        
        min_len = min(len(topo1), len(topo2))
        
        for i in range(min_len):
            if random.random() < 0.5:
                # Swap operations at position i
                node1 = child1.nodes[topo1[i].id]
                node2 = child2.nodes[topo2[i].id]
                node1.operation, node2.operation = node2.operation, node1.operation
                node1.params, node2.params = node2.params.copy(), node1.params.copy()
        
        return child1, child2
```

**`SinglePointCrossover`:**
```python
class SinglePointCrossover(Crossover):
    """
    Single-point crossover: split at random point, swap tails.
    """
    
    def crossover(self, graph1: ModelGraph, graph2: ModelGraph) -> Tuple[ModelGraph, ModelGraph]:
        # Get topological orders
        topo1 = graph1.topological_sort()
        topo2 = graph2.topological_sort()
        
        if len(topo1) < 3 or len(topo2) < 3:
            return graph1.clone(), graph2.clone()
        
        # Choose crossover point (exclude input/output)
        point = random.randint(1, min(len(topo1), len(topo2)) - 2)
        
        # Create offspring by swapping after crossover point
        child1 = self._recombine(graph1, graph2, topo1, topo2, point)
        child2 = self._recombine(graph2, graph1, topo2, topo1, point)
        
        return child1, child2
    
    def _recombine(self, g1: ModelGraph, g2: ModelGraph, topo1, topo2, point: int) -> ModelGraph:
        """Build child from g1[:point] + g2[point:]."""
        child = ModelGraph()
        
        # Copy nodes from g1 up to point
        for i in range(point):
            node = topo1[i]
            child.add_node(node.clone())
        
        # Copy nodes from g2 after point
        for i in range(point, len(topo2)):
            node = topo2[i]
            child.add_node(node.clone())
        
        # Reconnect edges (simplified)
        for i in range(len(child.nodes) - 1):
            nodes = list(child.nodes.values())
            child.add_edge(GraphEdge(nodes[i].id, nodes[i+1].id))
        
        child.input_node_id = list(child.nodes.values())[0].id
        child.output_node_id = list(child.nodes.values())[-1].id
        
        return child
```

---

### 5. `operators/selection.py` (~600 LOC)

**Additional selection methods:**

**`ElitistSelection`:**
```python
class ElitistSelection(SelectionStrategy):
    """Always select the best individuals."""
    
    def select(self, population: Population, n: int) -> List[Individual]:
        population.sort_by_fitness()
        return population.individuals[:n]
```

**`StochasticUniversalSampling`:**
```python
class StochasticUniversalSampling(SelectionStrategy):
    """
    SUS: Like roulette wheel but with evenly spaced pointers.
    Reduces variance compared to roulette wheel.
    """
    
    def select(self, population: Population, n: int) -> List[Individual]:
        fitnesses = [ind.fitness or 0 for ind in population.individuals]
        
        # Handle negative fitness
        min_fitness = min(fitnesses)
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 1 for f in fitnesses]
        
        total = sum(fitnesses)
        if total == 0:
            return random.sample(population.individuals, n)
        
        # Compute selection pointers
        pointer_distance = total / n
        start = random.uniform(0, pointer_distance)
        pointers = [start + i * pointer_distance for i in range(n)]
        
        # Select individuals
        selected = []
        cumulative = 0
        i = 0
        
        for pointer in pointers:
            while cumulative < pointer and i < len(population):
                cumulative += fitnesses[i]
                i += 1
            selected.append(population.individuals[max(0, i - 1)])
        
        return selected
```

---

## 🧪 Tests

**`test_genetic.py`:**

```python
def test_genetic_algorithm_initializes_population():
    space = SearchSpace(layers=[Layer.dense([128])])
    ga = GeneticAlgorithm(space, {'population_size': 10})
    pop = ga.initialize()
    assert len(pop) == 10

def test_genetic_algorithm_evolves_population():
    space = SearchSpace(layers=[Layer.dense([128])])
    ga = GeneticAlgorithm(space, {
        'population_size': 20,
        'elite_size': 2,
        'mutation_rate': 0.1
    })
    
    pop = ga.initialize()
    # Mock evaluate
    for ind in pop.individuals:
        ind.fitness = random.random()
    
    new_pop = ga.evolve()
    assert len(new_pop) == 20
    assert new_pop.generation == 1

def test_crossover_produces_valid_graphs():
    g1 = create_sample_graph()
    g2 = create_sample_graph()
    
    crossover = UniformCrossover()
    child1, child2 = crossover.crossover(g1, g2)
    
    assert child1.is_valid_dag()
    assert child2.is_valid_dag()

def test_mutation_preserves_dag_property():
    """Critical test: mutations must never break DAG."""
    graph = create_sample_graph()
    mutator = MutationSelector(
        [AddNodeMutation(['conv2d']), RemoveNodeMutation()],
        [0.5, 0.5]
    )
    
    for _ in range(100):
        mutated = mutator.mutate(graph)
        assert mutated.is_valid_dag(), "Mutation created cycle!"
```

**`test_operators.py`:**

```python
def test_tournament_selection_favors_high_fitness():
    pop = Population([
        Individual(MockGraph(), fitness=0.1),
        Individual(MockGraph(), fitness=0.9),
        Individual(MockGraph(), fitness=0.5)
    ])
    
    selector = TournamentSelection(tournament_size=2)
    selected = selector.select(pop, 100)
    
    # Higher fitness should be selected more often
    high_fitness_count = sum(1 for ind in selected if ind.fitness == 0.9)
    assert high_fitness_count > 50  # Should be majority
```

---

## 🎯 Usage Example

```python
from morphml import SearchSpace, Layer
from morphml.optimizers.evolutionary import GeneticAlgorithm

# Define search space
space = SearchSpace(
    layers=[
        Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5]),
        Layer.dense(units=[128, 256]),
        Layer.dropout(rate=[0.1, 0.3, 0.5])
    ]
)

# Configure GA
config = {
    'population_size': 50,
    'elite_size': 5,
    'mutation_rate': 0.15,
    'crossover_rate': 0.7,
    'max_generations': 100,
    'selection_strategy': 'tournament',
    'tournament_size': 3
}

# Initialize optimizer
ga = GeneticAlgorithm(space, config)
population = ga.initialize()

# Evolution loop
for generation in range(config['max_generations']):
    # Get candidates
    candidates = ga.ask()
    
    # Evaluate (user-defined)
    results = [(graph, evaluate(graph)) for graph in candidates]
    
    # Update
    ga.tell(results)
    
    # Evolve
    population = ga.evolve()
    
    print(f"Generation {generation}: best={ga.get_best_fitness()}")

# Get best architecture
best_graph = ga.get_best()
```

---

## ✅ Deliverables

- [ ] BaseOptimizer abstract class
- [ ] GeneticAlgorithm with full evolution loop
- [ ] Crossover operators (uniform, single-point)
- [ ] Selection operators (tournament, roulette, rank, SUS)
- [ ] Integration with mutation system
- [ ] Elitism support
- [ ] Statistics tracking
- [ ] Test coverage >80%
- [ ] All tests pass, especially DAG preservation

---

**Next:** `06_execution_cli.md`
