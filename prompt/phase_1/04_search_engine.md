# Component 4: Search Space & Engine

**Duration:** Week 5  
**LOC Target:** ~2,500  
**Dependencies:** Components 1-3

---

## 🎯 Objective

Implement search space definition system with parameter types, sampling strategies, and population management.

---

## 📋 Files to Create

### 1. `search_space.py` (~800 LOC)

**Parameter Classes:**

**`Parameter` (base ABC):**
```python
class Parameter(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def sample(self) -> Any:
        """Sample a value from parameter space."""
    
    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Check if value is valid."""
```

**`CategoricalParameter`:**
- Fields: `name`, `choices` (list)
- `sample()`: return random.choice(self.choices)
- Example: `CategoricalParameter('activation', ['relu', 'elu', 'gelu'])`

**`IntegerParameter`:**
- Fields: `name`, `low`, `high`, `log_scale` (bool)
- `sample()`: random.randint(low, high) or 2^uniform(log2(low), log2(high))
- Example: `IntegerParameter('filters', 32, 512, log_scale=True)`

**`FloatParameter`:**
- Fields: `name`, `low`, `high`, `log_scale` (bool)
- `sample()`: uniform(low, high) or 10^uniform(log10(low), log10(high))
- Example: `FloatParameter('learning_rate', 1e-4, 1e-2, log_scale=True)`

**`BooleanParameter`:**
- Fields: `name`, `probability` (float, default 0.5)
- `sample()`: random.random() < self.probability

---

**`Layer` class:**
```python
class Layer:
    """Layer specification with parameter ranges."""
    
    def __init__(self, layer_type: str, params: Dict[str, Parameter]):
        self.layer_type = layer_type
        self.params = params
    
    def sample(self) -> Dict[str, Any]:
        """Sample hyperparameters for this layer."""
        return {name: param.sample() for name, param in self.params.items()}
    
    @classmethod
    def conv2d(cls, filters=None, kernel_size=None, strides=1, padding='same'):
        """Factory method for conv2d layers."""
        params = {}
        if filters:
            if isinstance(filters, list):
                params['filters'] = CategoricalParameter('filters', filters)
            else:
                params['filters'] = IntegerParameter('filters', filters[0], filters[1])
        if kernel_size:
            params['kernel_size'] = CategoricalParameter('kernel_size', kernel_size if isinstance(kernel_size, list) else [kernel_size])
        params['strides'] = CategoricalParameter('strides', [strides] if isinstance(strides, int) else strides)
        params['padding'] = CategoricalParameter('padding', [padding])
        return cls('conv2d', params)
    
    @classmethod
    def dense(cls, units=None, activation=None):
        """Factory for dense layers."""
        params = {}
        if units:
            params['units'] = CategoricalParameter('units', units) if isinstance(units, list) else IntegerParameter('units', units[0], units[1])
        if activation:
            params['activation'] = CategoricalParameter('activation', activation if isinstance(activation, list) else [activation])
        return cls('dense', params)
    
    @classmethod
    def dropout(cls, rate=None):
        params = {}
        if rate:
            params['rate'] = FloatParameter('rate', rate[0], rate[1]) if isinstance(rate, tuple) else CategoricalParameter('rate', rate)
        return cls('dropout', params)
    
    @classmethod
    def batch_norm(cls):
        return cls('batch_norm', {})
```

**`SearchSpace` class:**
```python
class SearchSpace:
    """
    Defines the search space for architecture search.
    
    Contains layer specifications, global parameters (optimizer, LR),
    and constraints.
    """
    
    def __init__(
        self,
        layers: List[Layer],
        global_params: Dict[str, Parameter] = None,
        constraints: List[Constraint] = None
    ):
        self.layers = layers
        self.global_params = global_params or {}
        self.constraints = constraints or []
    
    def sample(self) -> ModelGraph:
        """
        Sample a random architecture from search space.
        
        Returns:
            ModelGraph representing sampled architecture
        """
        graph = ModelGraph()
        
        # Create input node
        input_node = GraphNode.create('input', {'shape': (224, 224, 3)})
        graph.add_node(input_node)
        graph.input_node_id = input_node.id
        
        prev_node_id = input_node.id
        
        # Sample layers
        num_layers = random.randint(3, 10)  # Random depth
        for _ in range(num_layers):
            layer = random.choice(self.layers)
            params = layer.sample()
            node = GraphNode.create(layer.layer_type, params)
            graph.add_node(node)
            graph.add_edge(GraphEdge(prev_node_id, node.id))
            prev_node_id = node.id
        
        # Create output node
        output_node = GraphNode.create('dense', {'units': 10, 'activation': 'softmax'})
        graph.add_node(output_node)
        graph.add_edge(GraphEdge(prev_node_id, output_node.id))
        graph.output_node_id = output_node.id
        
        # Validate constraints
        if not self.validate(graph):
            return self.sample()  # Resample if invalid
        
        return graph
    
    def validate(self, graph: ModelGraph) -> bool:
        """Check if graph satisfies all constraints."""
        for constraint in self.constraints:
            if not constraint.check(graph):
                return False
        return True
    
    def size(self) -> int:
        """Estimate search space size (combinatorial)."""
        size = 1
        for layer in self.layers:
            for param in layer.params.values():
                if isinstance(param, CategoricalParameter):
                    size *= len(param.choices)
                elif isinstance(param, IntegerParameter):
                    size *= (param.high - param.low + 1)
        return size
```

---

### 2. `population.py` (~400 LOC)

**`Individual` class:**
```python
@dataclass
class Individual:
    """Represents a candidate solution."""
    
    genome: ModelGraph
    fitness: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.metadata.setdefault('generation', 0)
        self.metadata.setdefault('parent_ids', [])
    
    def evaluate(self, evaluator: 'Evaluator') -> float:
        """Evaluate fitness using provided evaluator."""
        if self.fitness is None:
            self.fitness = evaluator.evaluate(self.genome)
        return self.fitness
    
    def clone(self) -> 'Individual':
        """Deep copy."""
        return Individual(
            genome=self.genome.clone(),
            fitness=self.fitness,
            metadata=self.metadata.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'genome': self.genome.to_dict(),
            'fitness': self.fitness,
            'metadata': self.metadata
        }
```

**`Population` class:**
```python
class Population:
    """Manages collection of individuals."""
    
    def __init__(self, individuals: List[Individual] = None):
        self.individuals = individuals or []
        self.generation = 0
    
    def add(self, individual: Individual) -> None:
        self.individuals.append(individual)
    
    def remove(self, individual: Individual) -> None:
        self.individuals.remove(individual)
    
    def sort_by_fitness(self, ascending: bool = False) -> None:
        """Sort population by fitness."""
        self.individuals.sort(
            key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'),
            reverse=not ascending
        )
    
    def get_best(self, n: int = 1) -> List[Individual]:
        """Get top n individuals."""
        self.sort_by_fitness()
        return self.individuals[:n]
    
    def get_worst(self, n: int = 1) -> List[Individual]:
        """Get bottom n individuals."""
        self.sort_by_fitness()
        return self.individuals[-n:]
    
    def average_fitness(self) -> float:
        """Compute mean fitness."""
        fitnesses = [ind.fitness for ind in self.individuals if ind.fitness is not None]
        return sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
    
    def best_fitness(self) -> float:
        """Get best fitness in population."""
        best = self.get_best(1)
        return best[0].fitness if best and best[0].fitness else 0.0
    
    def diversity_metric(self) -> float:
        """
        Measure population diversity (0=identical, 1=maximally diverse).
        
        Uses average graph edit distance between individuals.
        """
        if len(self.individuals) < 2:
            return 0.0
        
        total_distance = 0
        count = 0
        
        for i in range(len(self.individuals)):
            for j in range(i + 1, len(self.individuals)):
                dist = self._graph_distance(
                    self.individuals[i].genome,
                    self.individuals[j].genome
                )
                total_distance += dist
                count += 1
        
        avg_distance = total_distance / count if count > 0 else 0
        # Normalize by max possible distance
        max_dist = max(len(ind.genome) for ind in self.individuals)
        return min(1.0, avg_distance / max_dist) if max_dist > 0 else 0.0
    
    def _graph_distance(self, g1: ModelGraph, g2: ModelGraph) -> float:
        """Simple edit distance: |nodes1 - nodes2| + |edges1 - edges2|."""
        return abs(len(g1.nodes) - len(g2.nodes)) + abs(len(g1.edges) - len(g2.edges))
    
    def __len__(self) -> int:
        return len(self.individuals)
```

---

### 3. `selection.py` (~300 LOC)

**Base `SelectionStrategy` ABC:**
```python
class SelectionStrategy(ABC):
    @abstractmethod
    def select(self, population: Population, n: int) -> List[Individual]:
        """Select n individuals from population."""
```

**`TournamentSelection`:**
```python
class TournamentSelection(SelectionStrategy):
    """Select best from random tournament."""
    
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size
    
    def select(self, population: Population, n: int) -> List[Individual]:
        selected = []
        for _ in range(n):
            # Random tournament
            tournament = random.sample(population.individuals, self.tournament_size)
            winner = max(tournament, key=lambda ind: ind.fitness or 0)
            selected.append(winner)
        return selected
```

**`RouletteWheelSelection`:**
```python
class RouletteWheelSelection(SelectionStrategy):
    """Probability proportional to fitness."""
    
    def select(self, population: Population, n: int) -> List[Individual]:
        fitnesses = [ind.fitness or 0 for ind in population.individuals]
        # Handle negative fitness
        min_fitness = min(fitnesses)
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 1 for f in fitnesses]
        
        total = sum(fitnesses)
        if total == 0:
            return random.sample(population.individuals, n)
        
        probs = [f / total for f in fitnesses]
        return random.choices(population.individuals, weights=probs, k=n)
```

**`RankSelection`:**
```python
class RankSelection(SelectionStrategy):
    """Probability based on rank, not raw fitness."""
    
    def select(self, population: Population, n: int) -> List[Individual]:
        population.sort_by_fitness()
        ranks = list(range(1, len(population) + 1))
        total = sum(ranks)
        probs = [r / total for r in ranks]
        return random.choices(population.individuals, weights=probs, k=n)
```

---

### 4. `constraints.py` (~350 LOC)

**Base `Constraint` ABC:**
```python
class Constraint(ABC):
    @abstractmethod
    def check(self, graph: ModelGraph) -> bool:
        """Check if graph satisfies constraint."""
    
    @abstractmethod
    def message(self) -> str:
        """Human-readable constraint description."""
```

**Concrete Constraints:**

**`MaxNodesConstraint`:**
```python
class MaxNodesConstraint(Constraint):
    def __init__(self, max_nodes: int):
        self.max_nodes = max_nodes
    
    def check(self, graph: ModelGraph) -> bool:
        return len(graph.nodes) <= self.max_nodes
    
    def message(self) -> str:
        return f"Graph must have at most {self.max_nodes} nodes"
```

**`MaxDepthConstraint`:**
```python
class MaxDepthConstraint(Constraint):
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
    
    def check(self, graph: ModelGraph) -> bool:
        if not graph.input_node_id:
            return False
        
        # BFS to compute depth
        from collections import deque
        queue = deque([(graph.input_node_id, 0)])
        max_seen = 0
        visited = set()
        
        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            max_seen = max(max_seen, depth)
            
            node = graph.nodes[node_id]
            for succ_id in node.successors:
                queue.append((succ_id, depth + 1))
        
        return max_seen <= self.max_depth
```

**`RequiredLayerConstraint`:**
```python
class RequiredLayerConstraint(Constraint):
    """Ensure certain layer types are present."""
    
    def __init__(self, required_ops: List[str]):
        self.required_ops = required_ops
    
    def check(self, graph: ModelGraph) -> bool:
        present_ops = {node.operation for node in graph.nodes.values()}
        return all(op in present_ops for op in self.required_ops)
```

---

### 5. `search_engine.py` (~600 LOC)

**`SearchEngine` base class:**
```python
class SearchEngine(ABC):
    """Base class for search algorithms."""
    
    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space
        self.history: List[Individual] = []
    
    @abstractmethod
    def initialize_population(self, size: int) -> Population:
        """Create initial population."""
    
    @abstractmethod
    def step(self, population: Population) -> Population:
        """Execute one search iteration."""
    
    def search(
        self,
        evaluator: 'Evaluator',
        max_generations: int,
        population_size: int,
        early_stopping_patience: int = 10
    ) -> Individual:
        """
        Main search loop.
        
        Returns:
            Best individual found
        """
        population = self.initialize_population(population_size)
        
        # Evaluate initial population
        for individual in population.individuals:
            individual.evaluate(evaluator)
        
        best_fitness = population.best_fitness()
        patience_counter = 0
        
        for generation in range(max_generations):
            # Evolution step
            population = self.step(population)
            
            # Evaluate new individuals
            for individual in population.individuals:
                if individual.fitness is None:
                    individual.evaluate(evaluator)
            
            # Track history
            self.history.extend(population.individuals)
            
            # Check improvement
            current_best = population.best_fitness()
            if current_best > best_fitness:
                best_fitness = current_best
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at generation {generation}")
                break
            
            logger.info(
                f"Generation {generation}: "
                f"best={current_best:.4f}, avg={population.average_fitness():.4f}"
            )
        
        return population.get_best(1)[0]
```

---

## 🧪 Tests

**`test_search_space.py`:**
```python
def test_categorical_parameter_samples_from_choices():
    param = CategoricalParameter('act', ['relu', 'elu'])
    samples = [param.sample() for _ in range(100)]
    assert all(s in ['relu', 'elu'] for s in samples)

def test_search_space_samples_valid_graphs():
    space = SearchSpace(
        layers=[Layer.conv2d([32, 64]), Layer.dense([128])]
    )
    graph = space.sample()
    assert graph.is_valid_dag()
    assert len(graph.nodes) >= 3  # input + layers + output
```

**`test_population.py`:**
```python
def test_population_sorts_by_fitness():
    pop = Population([
        Individual(MockGraph(), fitness=0.5),
        Individual(MockGraph(), fitness=0.9),
        Individual(MockGraph(), fitness=0.3)
    ])
    pop.sort_by_fitness()
    assert pop.individuals[0].fitness == 0.9
```

---

## ✅ Deliverables

- [ ] Parameter types with sampling
- [ ] Layer factory methods
- [ ] SearchSpace with constraint validation
- [ ] Population management
- [ ] Selection strategies
- [ ] Constraint system
- [ ] Test coverage >80%

---

**Next:** `05_genetic_algorithm.md`
