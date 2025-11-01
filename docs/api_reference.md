# MorphML API Reference

**Complete API documentation for MorphML components**

---

## Optimizers

### GeneticAlgorithm

```python
from morphml.optimizers import GeneticAlgorithm

ga = GeneticAlgorithm(
    search_space: SearchSpace,
    population_size: int = 50,
    num_generations: int = 100,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.8,
    elitism: int = 5,
    selection_method: str = 'tournament',
    tournament_size: int = 3,
    max_mutations: int = 3,
    early_stopping_patience: Optional[int] = None
)
```

**Methods:**
- `optimize(evaluator, callback=None)` - Run optimization
- `get_history()` - Get optimization history
- `get_best_n(n)` - Get top N individuals
- `reset()` - Reset optimizer state

---

### RandomSearch

```python
from morphml.optimizers import RandomSearch

rs = RandomSearch(
    search_space: SearchSpace,
    num_samples: int = 100,
    allow_duplicates: bool = False
)
```

**Methods:**
- `optimize(evaluator)` - Run random search
- `get_all_evaluated()` - Get all evaluated individuals
- `get_best_n(n)` - Get top N individuals
- `reset()` - Reset optimizer

---

### HillClimbing

```python
from morphml.optimizers import HillClimbing

hc = HillClimbing(
    search_space: SearchSpace,
    max_iterations: int = 100,
    patience: int = 10,
    num_mutations: int = 3,
    mutation_rate: float = 0.3
)
```

**Methods:**
- `optimize(evaluator)` - Run hill climbing
- `get_history()` - Get fitness history
- `reset()` - Reset optimizer

---

## Evaluation

### HeuristicEvaluator

```python
from morphml.evaluation import HeuristicEvaluator

evaluator = HeuristicEvaluator(
    param_weight: float = 0.3,
    depth_weight: float = 0.3,
    width_weight: float = 0.2,
    connectivity_weight: float = 0.2,
    target_params: int = 1000000,
    target_depth: int = 20
)
```

**Methods:**
- `__call__(graph)` - Evaluate using combined score
- `combined_score(graph)` - Combined heuristic score
- `parameter_score(graph)` - Score based on parameter count
- `depth_score(graph)` - Score based on network depth
- `width_score(graph)` - Score based on network width
- `connectivity_score(graph)` - Score based on connectivity
- `get_all_scores(graph)` - Get all individual scores

---

## Utilities

### Checkpoint

```python
from morphml.utils import Checkpoint

# Save
Checkpoint.save(optimizer, filepath)

# Load
optimizer = Checkpoint.load(filepath, search_space, optimizer_class=None)

# List checkpoints
checkpoints = Checkpoint.list_checkpoints(directory='.')
```

---

### ArchitectureExporter

```python
from morphml.utils import ArchitectureExporter

exporter = ArchitectureExporter()

# PyTorch
pytorch_code = exporter.to_pytorch(graph, class_name='GeneratedModel')

# Keras
keras_code = exporter.to_keras(graph, model_name='generated_model')

# JSON
json_str = exporter.to_json(graph)
```

---

## DSL

### SearchSpace

```python
from morphml.core.dsl import SearchSpace, Layer

space = SearchSpace(name='my_space', metadata={})
```

**Methods:**
- `add_layer(layer_spec)` - Add single layer
- `add_layers(*layer_specs)` - Add multiple layers
- `add_constraint(constraint_fn)` - Add constraint
- `sample(max_attempts=100)` - Sample architecture
- `sample_batch(batch_size, max_attempts=100)` - Sample multiple
- `get_complexity()` - Get search space complexity
- `to_dict()` / `from_dict(data)` - Serialization

---

### Layer Builders

```python
from morphml.core.dsl import Layer

# Convolution
Layer.conv2d(filters, kernel_size=3, strides=1, padding='same', activation=None)

# Dense
Layer.dense(units, activation=None, use_bias=True)

# Pooling
Layer.maxpool(pool_size=2, strides=None, padding='valid')
Layer.avgpool(pool_size=2, strides=None, padding='valid')

# Regularization
Layer.dropout(rate=0.5)
Layer.batchnorm()

# Activations
Layer.relu()
Layer.sigmoid()
Layer.tanh()
Layer.softmax()

# Special
Layer.input(shape)
Layer.output(units, activation='softmax')
Layer.custom(operation, param_ranges)
```

---

## Graph System

### ModelGraph

```python
from morphml.core.graph import ModelGraph

graph = ModelGraph(metadata={})
```

**Methods:**
- `add_node(node)` - Add node
- `add_edge(edge)` - Add edge
- `remove_node(node_id)` - Remove node
- `remove_edge(edge_id)` - Remove edge
- `get_input_nodes()` - Get input nodes
- `get_output_nodes()` - Get output nodes
- `topological_sort()` - Get topological order
- `is_valid()` - Check validity
- `clone()` - Deep copy
- `to_networkx()` - Convert to NetworkX
- `to_dict()` / `from_dict(data)` - Serialization
- `to_json()` / `from_json(json_str)` - JSON serialization
- `hash()` - Get graph hash
- `get_depth()` - Get maximum depth
- `get_max_width()` - Get maximum width
- `estimate_parameters()` - Estimate parameter count

---

### GraphNode

```python
from morphml.core.graph import GraphNode

node = GraphNode.create(operation, params={}, metadata={})
```

**Methods:**
- `get_param(key, default=None)` - Get parameter
- `set_param(key, value)` - Set parameter
- `clone()` - Clone node
- `to_dict()` / `from_dict(data)` - Serialization

---

### GraphMutator

```python
from morphml.core.graph import GraphMutator

mutator = GraphMutator(operation_types=[...])
```

**Methods:**
- `mutate(graph, mutation_rate=0.1, max_mutations=None)` - Mutate graph
- `add_node_mutation(graph)` - Add node
- `remove_node_mutation(graph)` - Remove node
- `modify_node_mutation(graph)` - Modify parameters
- `add_edge_mutation(graph)` - Add edge
- `remove_edge_mutation(graph)` - Remove edge

---

## Population Management

### Individual

```python
from morphml.core.search import Individual

individual = Individual(graph, fitness=None, metadata={}, parent_ids=[])
```

**Methods:**
- `is_evaluated()` - Check if evaluated
- `set_fitness(fitness, **metrics)` - Set fitness
- `increment_age()` - Increment age
- `clone(keep_fitness=False)` - Clone individual
- `get_metric(key, default=None)` - Get metric
- `dominates(other, objectives)` - Check dominance
- `to_dict()` / `from_dict(data)` - Serialization

---

### Population

```python
from morphml.core.search import Population

pop = Population(max_size=100, elitism=5)
```

**Methods:**
- `add(individual)` - Add individual
- `add_many(individuals)` - Add multiple
- `remove(individual)` - Remove individual
- `clear()` - Remove all
- `size()` - Get size
- `is_full()` - Check if full
- `get_best(n=1)` - Get best individuals
- `get_worst(n=1)` - Get worst individuals
- `get_unevaluated()` - Get unevaluated
- `select(n, method='tournament', **kwargs)` - Select individuals
- `trim(target_size=None)` - Trim to size
- `increment_ages()` - Increment all ages
- `next_generation()` - Advance generation
- `get_statistics()` - Get statistics
- `get_diversity(method='hash')` - Calculate diversity

---

## Configuration

### ConfigManager

```python
from morphml.config import ConfigManager

# Load from YAML
config = ConfigManager.from_yaml('config.yaml')

# Load from dict
config = ConfigManager.from_dict({'key': 'value'})

# Get values
value = config.get('nested.key', default='default')

# Validate
exp_config = config.validate()

# Save
config.save('output.yaml')
```

---

## Logging

```python
from morphml.logging_config import setup_logging, get_logger

# Setup logging
logger = setup_logging(
    level='INFO',
    log_file='morphml.log',
    console=True
)

# Get logger for module
logger = get_logger(__name__)

# Use logger
logger.info("Information message")
logger.warning("Warning message")
logger.error("Error message")
logger.debug("Debug message")
```

---

## Exceptions

```python
from morphml.exceptions import (
    MorphMLError,          # Base exception
    DSLError,              # DSL errors
    GraphError,            # Graph errors
    SearchSpaceError,      # Search space errors
    OptimizerError,        # Optimizer errors
    EvaluationError,       # Evaluation errors
    ConfigurationError,    # Configuration errors
    DistributedError,      # Distributed errors
    ValidationError        # Validation errors
)
```

---

## Convenience Functions

### Pre-built Search Spaces

```python
from morphml.core.dsl import create_cnn_space, create_mlp_space

# CNN for image classification
cnn_space = create_cnn_space(
    num_classes=10,
    input_shape=(3, 32, 32),
    conv_filters=[[32, 64], [64, 128]],
    dense_units=[[256, 512]]
)

# MLP for structured data
mlp_space = create_mlp_space(
    num_classes=10,
    input_shape=(784,),
    hidden_layers=3,
    units_range=[128, 256, 512]
)
```

---

## Examples

### Complete Workflow

```python
# Imports
from morphml.optimizers import GeneticAlgorithm
from morphml.core.dsl import create_cnn_space
from morphml.evaluation import HeuristicEvaluator
from morphml.utils import Checkpoint, ArchitectureExporter

# 1. Define search space
space = create_cnn_space(num_classes=10)

# 2. Create optimizer
ga = GeneticAlgorithm(
    search_space=space,
    population_size=50,
    num_generations=100
)

# 3. Define evaluator
evaluator = HeuristicEvaluator()

# 4. Run optimization with callback
def callback(gen, pop):
    if gen % 10 == 0:
        stats = pop.get_statistics()
        print(f"Gen {gen}: Best={stats['best_fitness']:.4f}")
        Checkpoint.save(ga, f'checkpoint_{gen}.json')

best = ga.optimize(evaluator, callback=callback)

# 5. Export best architecture
exporter = ArchitectureExporter()
pytorch_code = exporter.to_pytorch(best.graph)

with open('best_model.py', 'w') as f:
    f.write(pytorch_code)

print(f"Best fitness: {best.fitness:.4f}")
print(f"Architecture: {best.graph}")
```

---

For more examples, see the [User Guide](user_guide.md) and [examples/](../examples/) directory.
