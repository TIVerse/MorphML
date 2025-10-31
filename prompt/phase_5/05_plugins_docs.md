# Component 5: Plugin System & Documentation

**Duration:** Weeks 7-8  
**LOC Target:** ~3,000  
**Dependencies:** All components complete

---

## 🎯 Objective

Complete ecosystem with extensibility:
1. **Plugin System** - Easy extensions
2. **Documentation Site** - MkDocs Material
3. **Tutorials** - Jupyter notebooks
4. **Examples** - Complete workflows
5. **Contributing Guide** - Community docs

---

## 📋 Files to Create

### 1. `plugins/plugin_system.py` (~1,000 LOC)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import importlib

class Plugin(ABC):
    """Base class for MorphML plugins."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]):
        """Initialize plugin with config."""
        pass
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Any:
        """Execute plugin logic."""
        pass


class OptimizerPlugin(Plugin):
    """Plugin for custom optimizers."""
    
    def get_optimizer(self) -> BaseOptimizer:
        """Return optimizer instance."""
        pass


class EvaluatorPlugin(Plugin):
    """Plugin for custom evaluation metrics."""
    
    def evaluate(self, architecture: ModelGraph) -> float:
        """Evaluate architecture."""
        pass


class PluginManager:
    """
    Manage and load plugins.
    
    Plugins can be:
    - Python modules in morphml/plugins/
    - External packages with entry points
    """
    
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
    
    def load_plugin(self, plugin_name: str, config: Dict = None):
        """Load plugin by name."""
        try:
            # Try importing from morphml.plugins
            module = importlib.import_module(f'morphml.plugins.{plugin_name}')
            plugin_class = getattr(module, 'Plugin')
            
            plugin = plugin_class()
            plugin.initialize(config or {})
            
            self.plugins[plugin_name] = plugin
            
            logger.info(f"Loaded plugin: {plugin_name}")
        
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
    
    def get_plugin(self, plugin_name: str) -> Plugin:
        """Get loaded plugin."""
        return self.plugins.get(plugin_name)


# Example plugin
class CustomOptimizerPlugin(OptimizerPlugin):
    """Example custom optimizer plugin."""
    
    def initialize(self, config):
        self.config = config
    
    def get_optimizer(self):
        # Return custom optimizer
        return MyCustomOptimizer(self.config)
```

---

### 2. Documentation Structure

```
docs/
├── index.md                          # Landing page
├── getting-started/
│   ├── installation.md               # Installation guide
│   ├── quickstart.md                 # 5-minute tutorial
│   └── concepts.md                   # Core concepts
├── user-guide/
│   ├── search-spaces.md              # Define search spaces
│   ├── optimizers.md                 # Using optimizers
│   ├── distributed.md                # Distributed setup
│   └── visualization.md              # Plotting results
├── api-reference/
│   ├── core.md                       # Core APIs
│   ├── optimizers.md                 # Optimizer APIs
│   └── distributed.md                # Distributed APIs
├── tutorials/
│   ├── cifar10-nas.ipynb             # Basic tutorial
│   ├── multi-objective.ipynb         # Multi-objective
│   └── custom-optimizer.ipynb        # Extend MorphML
├── deployment/
│   ├── docker.md                     # Docker setup
│   ├── kubernetes.md                 # K8s deployment
│   └── cloud.md                      # Cloud providers
└── contributing/
    ├── development.md                # Dev setup
    ├── code-style.md                 # Style guide
    └── pull-requests.md              # PR process
```

---

### 3. `docs/mkdocs.yml` (~100 LOC)

```yaml
site_name: MorphML Documentation
site_description: Automated Neural Architecture Search
site_author: Eshan Roy
repo_url: https://github.com/TIVerse/MorphML
repo_name: TIVerse/MorphML

theme:
  name: material
  palette:
    primary: indigo
    accent: indigo
  features:
    - navigation.tabs
    - navigation.sections
    - toc.integrate
    - search.suggest
    - content.code.copy

nav:
  - Home: index.md
  - Getting Started:
      - Installation: getting-started/installation.md
      - Quickstart: getting-started/quickstart.md
      - Core Concepts: getting-started/concepts.md
  - User Guide:
      - Search Spaces: user-guide/search-spaces.md
      - Optimizers: user-guide/optimizers.md
      - Distributed: user-guide/distributed.md
  - Tutorials:
      - CIFAR-10 NAS: tutorials/cifar10-nas.md
      - Multi-Objective: tutorials/multi-objective.md
  - API Reference:
      - Core: api-reference/core.md
      - Optimizers: api-reference/optimizers.md
  - Deployment:
      - Docker: deployment/docker.md
      - Kubernetes: deployment/kubernetes.md
  - Contributing: contributing/development.md

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true

markdown_extensions:
  - pymdownx.highlight
  - pymdownx.superfences
  - admonition
  - codehilite
```

---

### 4. `docs/getting-started/quickstart.md` (~300 LOC)

```markdown
# Quickstart

Get started with MorphML in 5 minutes.

## Installation

```bash
pip install morphml
```

## Define Search Space

```python
from morphml import SearchSpace, Layer

space = SearchSpace()
space.add_layer(Layer.conv2d(filters=[32, 64, 128]))
space.add_layer(Layer.maxpool())
space.add_layer(Layer.conv2d(filters=[64, 128, 256]))
space.add_layer(Layer.dense(units=[128, 256]))
```

## Run Search

```python
from morphml.optimizers import GeneticAlgorithm

optimizer = GeneticAlgorithm(space, {
    'population_size': 50,
    'num_generations': 100
})

best_architecture = optimizer.optimize()
```

## Evaluate

```python
from morphml.evaluation import evaluate_architecture

results = evaluate_architecture(
    best_architecture,
    dataset='cifar10',
    num_epochs=50
)

print(f"Accuracy: {results['accuracy']:.4f}")
```

## Next Steps

- [Full User Guide](../user-guide/search-spaces.md)
- [Distributed Search](../user-guide/distributed.md)
- [Advanced Tutorials](../tutorials/cifar10-nas.md)
```

---

### 5. Example Plugins

**`morphml/plugins/custom_optimizer_example.py`:**

```python
from morphml.plugins import OptimizerPlugin

class Plugin(OptimizerPlugin):
    """Example: Simulated Annealing optimizer."""
    
    def initialize(self, config):
        self.temperature = config.get('temperature', 100.0)
        self.cooling_rate = config.get('cooling_rate', 0.95)
    
    def get_optimizer(self):
        from morphml.optimizers.simulated_annealing import SimulatedAnnealing
        
        return SimulatedAnnealing(
            temperature=self.temperature,
            cooling_rate=self.cooling_rate
        )
```

---

## 📚 Complete Documentation

### Key Documentation Files to Create:

1. **Installation Guide** (500 lines)
   - pip install
   - conda install
   - from source
   - GPU setup

2. **Tutorials** (10 notebooks, ~200 lines each)
   - CIFAR-10 basic search
   - Multi-objective optimization
   - Distributed search on K8s
   - Custom optimizer
   - Transfer learning
   - Benchmarking

3. **API Reference** (Auto-generated, ~5000 lines)
   - All public classes
   - All public functions
   - Type signatures
   - Examples

4. **Deployment Guides** (1000 lines)
   - Docker
   - Kubernetes
   - AWS/GCP/Azure
   - On-premise clusters

---

## 🧪 Example Notebooks

**`examples/cifar10_basic.ipynb`:**

```python
# Cell 1: Setup
import morphml
from morphml import SearchSpace, Layer
from morphml.optimizers import GeneticAlgorithm

# Cell 2: Define search space
space = SearchSpace()
# ... define layers

# Cell 3: Run search
optimizer = GeneticAlgorithm(space, {})
best = optimizer.optimize()

# Cell 4: Visualize
from morphml.visualization import plot_architecture
plot_architecture(best)

# Cell 5: Train and evaluate
results = train_and_evaluate(best, dataset='cifar10')
print(f"Test accuracy: {results['test_acc']:.4f}")
```

---

## ✅ Deliverables

- [ ] Plugin system with examples
- [ ] Complete documentation site (100+ pages)
- [ ] 10+ tutorial notebooks
- [ ] Example workflows
- [ ] Contributing guide
- [ ] API reference (auto-generated)
- [ ] Deployment guides for all major platforms

---

## 🎉 MorphML Complete!

All 5 phases finished with:
- ✅ ~95,000 LOC production code
- ✅ ~12,000 LOC tests
- ✅ Complete documentation
- ✅ Full ecosystem (dashboard, CLI, API, integrations)
- ✅ Production-ready deployment (Kubernetes)
- ✅ Extensible plugin system

**Ready for open-source release!** 🚀
