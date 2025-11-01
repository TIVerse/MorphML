# MorphML User Guide

**Complete guide to using MorphML for Neural Architecture Search**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Optimizers](#optimizers)
3. [Evaluation](#evaluation)
4. [Utilities](#utilities)
5. [Advanced Usage](#advanced-usage)
6. [Best Practices](#best-practices)

---

## Quick Start

### Installation

```bash
git clone https://github.com/TIVerse/MorphML.git
cd MorphML
poetry install
```

### Your First NAS

```python
from morphml.optimizers import GeneticAlgorithm
from morphml.core.dsl import create_cnn_space

# 1. Define search space
space = create_cnn_space(num_classes=10)

# 2. Create optimizer
ga = GeneticAlgorithm(
    search_space=space,
    population_size=50,
    num_generations=100
)

# 3. Define evaluator
def evaluate_architecture(graph):
    # Your training/evaluation logic
    return accuracy

# 4. Run optimization
best = ga.optimize(evaluator=evaluate_architecture)
print(f"Best fitness: {best.fitness:.4f}")
```

---

## Optimizers

### 1. Genetic Algorithm

**Best for:** General-purpose NAS, large search spaces

```python
from morphml.optimizers import GeneticAlgorithm

ga = GeneticAlgorithm(
    search_space=space,
    population_size=50,          # Population size
    num_generations=100,         # Max generations
    mutation_rate=0.2,           # Mutation probability
    crossover_rate=0.8,          # Crossover probability
    elitism=5,                   # Keep top 5
    selection_method='tournament', # Selection strategy
    tournament_size=3,           # Tournament size
    early_stopping_patience=10   # Early stopping
)

best = ga.optimize(evaluator=evaluate)
```

**Features:**
- Population-based search
- Selection strategies (tournament, roulette, rank, random)
- Elitism preservation
- Early stopping
- Progress callbacks
- History tracking

### 2. Random Search

**Best for:** Baseline comparison, quick experiments

```python
from morphml.optimizers import RandomSearch

rs = RandomSearch(
    search_space=space,
    num_samples=100,            # Number of architectures to try
    allow_duplicates=False      # Skip duplicate architectures
)

best = rs.optimize(evaluator=evaluate)
```

**Features:**
- Simple and fast
- Good baseline
- No hyperparameters to tune
- Surprisingly effective

### 3. Hill Climbing

**Best for:** Local refinement, small search spaces

```python
from morphml.optimizers import HillClimbing

hc = HillClimbing(
    search_space=space,
    max_iterations=100,         # Max iterations
    patience=10,                # Stop if no improvement
    num_mutations=3,            # Mutations per neighbor
    mutation_rate=0.3           # Mutation rate
)

best = hc.optimize(evaluator=evaluate)
```

**Features:**
- Local search
- Fast convergence
- Good for refinement
- Low memory usage

---

## Evaluation

### Custom Evaluators

```python
def my_evaluator(graph):
    """Your custom evaluation logic."""
    # Build model from graph
    model = build_model(graph)
    
    # Train and evaluate
    model.fit(x_train, y_train, epochs=10)
    accuracy = model.evaluate(x_test, y_test)
    
    return accuracy
```

### Heuristic Evaluators

**Fast evaluation without training:**

```python
from morphml.evaluation import HeuristicEvaluator

# Create evaluator
evaluator = HeuristicEvaluator(
    param_weight=0.3,           # Weight for parameter penalty
    depth_weight=0.3,           # Weight for depth score
    width_weight=0.2,           # Weight for width score
    connectivity_weight=0.2,    # Weight for connectivity
    target_params=1000000,      # Target parameter count
    target_depth=20             # Target network depth
)

# Evaluate
score = evaluator(graph)

# Get all scores
all_scores = evaluator.get_all_scores(graph)
print(all_scores)
# {'parameter': 0.85, 'depth': 0.92, 'width': 0.78, ...}
```

**Use cases:**
- Quick prototyping
- Development and debugging
- Initial screening
- Constraint validation

---

## Utilities

### Checkpointing

**Save and resume optimization:**

```python
from morphml.utils import Checkpoint

# During optimization
if generation % 10 == 0:
    Checkpoint.save(ga, f'checkpoint_gen_{generation}.json')

# Resume later
ga = Checkpoint.load('checkpoint_gen_50.json', search_space)
best = ga.optimize(evaluator)  # Continue from generation 50
```

### Architecture Export

**Generate framework-specific code:**

```python
from morphml.utils import ArchitectureExporter

exporter = ArchitectureExporter()

# PyTorch
pytorch_code = exporter.to_pytorch(best.graph, class_name='MyModel')
with open('model_pytorch.py', 'w') as f:
    f.write(pytorch_code)

# Keras
keras_code = exporter.to_keras(best.graph, model_name='my_model')
with open('model_keras.py', 'w') as f:
    f.write(keras_code)

# JSON
json_str = exporter.to_json(best.graph)
with open('architecture.json', 'w') as f:
    f.write(json_str)
```

---

## Advanced Usage

### Progress Tracking

```python
def progress_callback(generation, population):
    stats = population.get_statistics()
    diversity = population.get_diversity()
    
    print(f"Generation {generation}")
    print(f"  Best:      {stats['best_fitness']:.4f}")
    print(f"  Mean:      {stats['mean_fitness']:.4f}")
    print(f"  Diversity: {diversity:.2f}")

best = ga.optimize(evaluator, callback=progress_callback)
```

### Multi-Run Experiments

```python
results = []

for run in range(5):
    print(f"\nRun {run + 1}/5")
    
    # Reset optimizer
    ga.reset()
    
    # Run optimization
    best = ga.optimize(evaluator)
    results.append({
        'run': run + 1,
        'fitness': best.fitness,
        'nodes': len(best.graph.nodes),
        'hash': best.graph.hash()
    })

# Analyze results
import pandas as pd
df = pd.DataFrame(results)
print(df.describe())
```

### Getting Top-N Architectures

```python
# Run optimization
best = ga.optimize(evaluator)

# Get top 10 architectures
top_10 = ga.get_best_n(n=10)

for i, individual in enumerate(top_10, 1):
    print(f"{i}. Fitness: {individual.fitness:.4f}")
    print(f"   Nodes: {len(individual.graph.nodes)}")
    print(f"   Hash: {individual.graph.hash()[:16]}")
    print()
```

### Custom Search Spaces

```python
from morphml.core.dsl import SearchSpace, Layer

# Define custom space
space = SearchSpace("custom_cnn")

space.add_layers(
    Layer.input(shape=(3, 224, 224)),
    
    # Block 1
    Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5, 7]),
    Layer.relu(),
    Layer.batchnorm(),
    Layer.maxpool(pool_size=2),
    
    # Block 2
    Layer.conv2d(filters=[64, 128, 256], kernel_size=[3, 5]),
    Layer.relu(),
    Layer.batchnorm(),
    Layer.maxpool(pool_size=2),
    
    # Block 3
    Layer.conv2d(filters=[128, 256, 512], kernel_size=3),
    Layer.relu(),
    Layer.maxpool(pool_size=2),
    
    # Classifier
    Layer.dense(units=[256, 512, 1024]),
    Layer.relu(),
    Layer.dropout(rate=[0.3, 0.5, 0.7]),
    Layer.dense(units=[128, 256, 512]),
    Layer.relu(),
    Layer.dropout(rate=[0.3, 0.5]),
    
    Layer.output(units=1000)  # ImageNet
)

# Check complexity
complexity = space.get_complexity()
print(f"Search space size: {complexity['total_combinations']:,}")
```

### Adding Constraints

```python
# Define constraint
def max_parameters(graph):
    return graph.estimate_parameters() <= 10_000_000

def max_depth(graph):
    return graph.get_depth() <= 30

# Add to search space
space.add_constraint(max_parameters)
space.add_constraint(max_depth)

# Sampling will respect constraints
arch = space.sample()  # Guaranteed to satisfy constraints
```

---

## Best Practices

### 1. Start Small

```python
# Begin with small population and few generations
ga = GeneticAlgorithm(
    search_space=space,
    population_size=10,
    num_generations=10
)

# Verify everything works
best = ga.optimize(heuristic_evaluator)
```

### 2. Use Heuristic Evaluators for Development

```python
from morphml.evaluation import HeuristicEvaluator

# Fast iteration during development
heuristic = HeuristicEvaluator()

# Test your search space
for _ in range(10):
    arch = space.sample()
    score = heuristic(arch)
    print(f"Score: {score:.3f}, Nodes: {len(arch.nodes)}")
```

### 3. Checkpoint Regularly

```python
for gen in range(100):
    ga.evolve_generation()
    ga.evaluate_population(evaluator)
    
    # Save every 10 generations
    if gen % 10 == 0:
        Checkpoint.save(ga, f'checkpoint_{gen}.json')
```

### 4. Compare Multiple Optimizers

```python
optimizers = {
    'Random': RandomSearch(space, num_samples=50),
    'HillClimb': HillClimbing(space, max_iterations=50),
    'GA': GeneticAlgorithm(space, population_size=20, num_generations=10)
}

results = {}
for name, opt in optimizers.items():
    print(f"\nRunning {name}...")
    best = opt.optimize(evaluator)
    results[name] = best.fitness
    print(f"{name}: {best.fitness:.4f}")

# Compare
best_optimizer = max(results, key=results.get)
print(f"\nBest: {best_optimizer} ({results[best_optimizer]:.4f})")
```

### 5. Analyze History

```python
import matplotlib.pyplot as plt

# Run optimization
best = ga.optimize(evaluator)

# Get history
history = ga.get_history()

# Plot
generations = [h['generation'] for h in history]
best_fitness = [h['best_fitness'] for h in history]
mean_fitness = [h['mean_fitness'] for h in history]

plt.figure(figsize=(10, 5))
plt.plot(generations, best_fitness, label='Best', linewidth=2)
plt.plot(generations, mean_fitness, label='Mean', linewidth=2)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.title('Evolution Progress')
plt.grid(True)
plt.show()
```

### 6. Export and Use Best Architectures

```python
from morphml.utils import ArchitectureExporter

# Run NAS
best = ga.optimize(evaluator)

# Export to PyTorch
exporter = ArchitectureExporter()
code = exporter.to_pytorch(best.graph)

# Save
with open('best_model.py', 'w') as f:
    f.write(code)

# Now you can import and use it
# from best_model import GeneratedModel
# model = GeneratedModel()
```

---

## Tips and Tricks

### Faster Evaluation

```python
# Use smaller epochs during search
def fast_evaluator(graph):
    model = build_model(graph)
    model.fit(x_train, y_train, epochs=5)  # Quick training
    return model.evaluate(x_val, y_val)

# Full training for final model
best = ga.optimize(fast_evaluator)
final_model = build_model(best.graph)
final_model.fit(x_train, y_train, epochs=200)  # Full training
```

### Warm Starting

```python
# First run
ga1 = GeneticAlgorithm(space, num_generations=50)
best1 = ga1.optimize(evaluator)
Checkpoint.save(ga1, 'warmstart.json')

# Continue with more generations
ga2 = Checkpoint.load('warmstart.json', space)
ga2.config['num_generations'] = 100
best2 = ga2.optimize(evaluator)
```

### Parallel Evaluation (Conceptual)

```python
from multiprocessing import Pool

def parallel_evaluator(graphs):
    with Pool(processes=4) as pool:
        fitnesses = pool.map(evaluate_single, graphs)
    return fitnesses

# Use in custom optimization loop
```

---

## Common Issues

### Out of Memory

**Solution:** Reduce population size or use checkpointing

```python
ga = GeneticAlgorithm(
    search_space=space,
    population_size=20,  # Smaller population
    num_generations=200  # More generations
)
```

### Slow Convergence

**Solution:** Increase mutation rate or try different optimizer

```python
ga = GeneticAlgorithm(
    search_space=space,
    mutation_rate=0.4,  # Higher mutation
    early_stopping_patience=20
)
```

### No Improvement

**Solution:** Check search space and evaluation

```python
# Verify search space
complexity = space.get_complexity()
print(f"Combinations: {complexity['total_combinations']}")

# Test with heuristic
from morphml.evaluation import HeuristicEvaluator
heuristic = HeuristicEvaluator()
for _ in range(10):
    arch = space.sample()
    print(f"Score: {heuristic(arch):.3f}")
```

---

## Next Steps

- Read [API Reference](api_reference.md) for detailed documentation
- See [Examples](../examples/) for complete examples
- Check [Advanced Topics](advanced.md) for expert usage
- Join [Discussions](https://github.com/TIVerse/MorphML/discussions) for help

---

**Happy Architecture Searching!** 🚀
