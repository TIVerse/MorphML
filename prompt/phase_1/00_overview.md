# Phase 1: Foundation - Overview

**Authors:** Vedanth ([@vedanthq](https://github.com/vedanthq)) & Eshan Roy ([@eshanized](https://github.com/eshanized))  
**Organization:** TONMOY INFRASTRUCTURE & VISION  
**Repository:** https://github.com/TIVerse/MorphML  
**Phase Duration:** Months 1-6 (8-12 weeks)  
**Target LOC:** ~20,000 production + 3,000 tests

---

## 🎯 Phase 1 Mission

Build the foundational infrastructure of MorphML, enabling users to:
1. Define search spaces using a Pythonic DSL
2. Execute genetic algorithm-based neural architecture search
3. Visualize and track experiments through a CLI
4. Run end-to-end workflows on a single machine

By the end of Phase 1, a researcher should be able to install MorphML, write a 20-line Python script defining a search space, run `morphml run experiment.py`, and get back optimized neural architectures with performance metrics.

---

## 📋 Success Criteria

### Functional Requirements
- ✅ DSL can parse valid search space definitions
- ✅ DSL provides clear error messages for invalid syntax
- ✅ Genetic algorithm evolves populations over generations
- ✅ Model graphs maintain valid DAG structure after mutations
- ✅ Local executor evaluates models and tracks results
- ✅ CLI provides interactive progress display
- ✅ Results are serializable and resumable

### Quality Requirements
- ✅ Test coverage >75% across all modules
- ✅ Type hints on all public APIs
- ✅ Documentation for every public function
- ✅ Code passes black, ruff, mypy without warnings
- ✅ No function exceeds 50 lines
- ✅ All code looks human-written with thoughtful comments

### Performance Requirements
- ✅ DSL compilation completes in <1 second for typical specs
- ✅ Genetic algorithm handles populations of 100+ individuals
- ✅ Graph mutations execute in <10ms
- ✅ Memory usage <2GB for standard experiments

---

## 🏗️ Architecture Overview

### Layer Stack

```
┌─────────────────────────────────────────┐
│           User Interface                 │
│              CLI + Rich                  │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│          Experiment Executor             │
│         LocalExecutor + Evaluator        │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Search Engine                    │
│      GeneticAlgorithm + Population       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│          Core Abstractions               │
│    SearchSpace + ModelGraph + DSL        │
└─────────────────────────────────────────┘
```

### Data Flow

```
User Script (DSL)
    ↓
Lexer → Parser → AST
    ↓
Compiler → SearchSpace + Evolution Config
    ↓
Experiment → LocalExecutor
    ↓
GeneticAlgorithm:
    - Initialize Population (sample from SearchSpace)
    - For each generation:
        - Select parents (tournament, roulette)
        - Apply crossover (graph mixing)
        - Apply mutation (add/remove/modify nodes)
        - Evaluate fitness (train + validate)
        - Update population
    ↓
Results → Logger + Serialization
    ↓
Best Models + Metrics
```

---

## 📦 Phase 1 Components

### Component 1: Project Infrastructure (Week 1)
**File:** `01_project_setup.md`

- Repository structure and configuration
- Poetry setup with dependencies
- CI/CD pipeline (GitHub Actions)
- Code quality tools (black, mypy, ruff)
- Logging and configuration system
- Custom exceptions

### Component 2: DSL Implementation (Weeks 2-3)
**File:** `02_dsl_implementation.md`

- Lexical analysis (tokenizer)
- Parser (token stream → AST)
- AST node definitions
- Compiler (AST → internal representation)
- Semantic validator
- Type system

### Component 3: Model Graph System (Week 4)
**File:** `03_graph_system.md`

- Graph nodes (layers/operations)
- Graph edges (data flow)
- ModelGraph class (DAG management)
- Mutation operators (add, remove, modify, rewire)
- Serialization and visualization

### Component 4: Search Space & Engine (Week 5)
**File:** `04_search_engine.md`

- Parameter types (categorical, integer, float)
- SearchSpace class
- Layer specifications
- Population management
- Selection strategies
- Constraint validation

### Component 5: Genetic Algorithm (Week 6-7)
**File:** `05_genetic_algorithm.md`

- BaseOptimizer abstract class
- GeneticAlgorithm implementation
- Selection operators (tournament, roulette, rank)
- Crossover operators (uniform, single-point)
- Mutation operators (graph-level)
- Evolution loop

### Component 6: Execution & CLI (Week 8)
**File:** `06_execution_cli.md`

- LocalExecutor for single-machine runs
- ModelEvaluator for fitness evaluation
- Result logging and checkpointing
- CLI interface with Rich formatting
- Command structure (run, status, config)

---

## 🔧 Technology Stack

### Core Dependencies
```toml
python = "^3.10"           # Modern Python with match/case
numpy = "^1.24.0"          # Numerical operations
scipy = "^1.10.0"          # Scientific computing
networkx = "^3.1"          # Graph algorithms (DAG validation)
pydantic = "^2.0.0"        # Data validation
click = "^8.1.0"           # CLI framework
rich = "^13.0.0"           # Terminal formatting
pyyaml = "^6.0"            # Configuration files
typing-extensions = "^4.5" # Enhanced typing
```

### Development Dependencies
```toml
pytest = "^7.4.0"          # Testing framework
pytest-cov = "^4.1.0"      # Coverage reporting
black = "^23.7.0"          # Code formatting
mypy = "^1.4.0"            # Static type checking
ruff = "^0.0.280"          # Fast linting
pre-commit = "^3.3.0"      # Git hooks
```

### Why These Choices?

**NetworkX**: Battle-tested graph library, handles DAG validation, topological sorting, and shortest paths efficiently.

**Pydantic**: Runtime validation with excellent error messages, perfect for configuration and DSL validation.

**Click + Rich**: Click provides robust CLI framework, Rich adds beautiful progress bars, tables, and colored output.

**Poetry**: Superior dependency management, lock files, and virtual environment handling compared to pip.

---

## 📝 Coding Standards

### Python Style
- **PEP 8** compliant (enforced by black)
- **Line length**: 100 characters
- **Naming**:
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private members: `_leading_underscore`

### Type Hints
```python
# Always use type hints on function signatures
def sample_architecture(
    space: SearchSpace,
    constraints: Optional[List[Constraint]] = None
) -> ModelGraph:
    """Sample a random architecture from search space."""
    pass

# Use modern union syntax (Python 3.10+)
def get_param(name: str) -> int | float | str | None:
    pass

# Use TypeVar for generics
T = TypeVar('T')
def first_element(items: List[T]) -> T:
    return items[0]
```

### Documentation
```python
def evaluate_model(
    graph: ModelGraph,
    dataset: str,
    metrics: List[str]
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.

    This function trains the model defined by the graph on the specified
    dataset and computes the requested metrics. Training uses early stopping
    with patience=10 and monitors validation loss.

    Args:
        graph: Model architecture as a DAG
        dataset: Dataset name (e.g., 'cifar10', 'mnist')
        metrics: List of metrics to compute (e.g., ['accuracy', 'f1'])

    Returns:
        Dictionary mapping metric names to values

    Raises:
        ValidationError: If graph is invalid or dataset not found
        ExecutionError: If training fails

    Example:
        >>> graph = ModelGraph(...)
        >>> results = evaluate_model(graph, 'cifar10', ['accuracy'])
        >>> print(f"Accuracy: {results['accuracy']:.2%}")
    """
    pass
```

### Comments
```python
# Good: Explain WHY, not WHAT
# Use tournament selection to maintain diversity while still favoring
# high-fitness individuals. Tournament size of 3 provides good balance.
parents = tournament_selection(population, tournament_size=3)

# Bad: Repeating what code says
# Select parents from population with tournament selection
parents = tournament_selection(population, tournament_size=3)
```

### Error Handling
```python
# Prefer specific exceptions with helpful messages
if not self._is_valid_dag():
    raise GraphError(
        f"Graph contains cycle: {self._find_cycle()}. "
        "Neural networks require acyclic graphs."
    )

# Not generic exceptions
if not self._is_valid_dag():
    raise Exception("Invalid graph")
```

### Function Size
```python
# Good: Small, focused functions
def mutate_graph(graph: ModelGraph, rate: float) -> ModelGraph:
    """Apply mutations to graph."""
    mutated = graph.clone()
    
    if random.random() < rate:
        mutated = _add_node_mutation(mutated)
    if random.random() < rate:
        mutated = _modify_node_mutation(mutated)
    
    return mutated

# Bad: Large, multi-responsibility functions
def run_evolution():
    # 200 lines of mixed concerns
    pass
```

---

## 🧪 Testing Strategy

### Test Organization
```
tests/
├── unit/                    # Test individual components
│   ├── test_dsl/
│   ├── test_graph/
│   ├── test_search/
│   └── test_optimizers/
├── integration/             # Test component interactions
│   └── test_end_to_end.py
└── conftest.py             # Shared fixtures
```

### Test Coverage Targets
- **Overall**: 75% minimum
- **Core modules** (dsl, graph, search): 85%
- **Optimizers**: 80%
- **CLI**: 60% (harder to test)

### Test Patterns
```python
# Use descriptive test names
def test_lexer_recognizes_floating_point_numbers():
    """Lexer should tokenize floats with scientific notation."""
    lexer = Lexer("3.14e-5")
    tokens = lexer.tokenize()
    assert tokens[0].type == TokenType.NUMBER
    assert abs(tokens[0].value - 3.14e-5) < 1e-10

# Use fixtures for common setup
@pytest.fixture
def sample_graph():
    """Create a simple 3-layer CNN graph."""
    graph = ModelGraph()
    graph.add_node(GraphNode("input", "input", {}))
    graph.add_node(GraphNode("conv1", "conv2d", {"filters": 32}))
    graph.add_node(GraphNode("output", "dense", {"units": 10}))
    graph.add_edge(GraphEdge("input", "conv1"))
    graph.add_edge(GraphEdge("conv1", "output"))
    return graph

# Test edge cases
def test_mutation_preserves_dag_property():
    """Mutations should never create cycles."""
    graph = sample_graph()
    for _ in range(100):  # Try many times
        mutated = AddNodeMutation().mutate(graph)
        assert mutated.is_valid_dag()
```

---

## 📊 Deliverables Checklist

### Code
- [ ] All components implemented per specifications
- [ ] Type hints on all public APIs
- [ ] Docstrings on all public functions/classes
- [ ] Code passes black, ruff, mypy
- [ ] No security vulnerabilities (bandit scan)

### Tests
- [ ] Unit tests for all components (>75% coverage)
- [ ] Integration test for end-to-end workflow
- [ ] Tests pass on Python 3.10 and 3.11
- [ ] All tests documented with clear assertions

### Documentation
- [ ] README with quickstart example
- [ ] Architecture documentation updated
- [ ] Example scripts (quickstart.py, cifar10_example.py)
- [ ] Inline code documentation

### Infrastructure
- [ ] CI/CD pipeline running on GitHub Actions
- [ ] Code coverage reporting enabled
- [ ] Pre-commit hooks configured
- [ ] PyPI package structure ready

---

## 🚀 Getting Started with This Phase

### For LLM-Assisted Development:

1. **Start with Component 1** (`01_project_setup.md`)
   - Set up project structure
   - Configure tools
   - Test CI/CD pipeline

2. **Proceed to Component 2** (`02_dsl_implementation.md`)
   - Implement lexer with tests
   - Implement parser with tests
   - Implement compiler with tests

3. **Continue sequentially** through components 3-6
   - Each component builds on previous ones
   - Run tests after each component
   - Ensure integration works

4. **Final integration**
   - Run end-to-end test
   - Test example scripts
   - Validate all success criteria

### Development Workflow:

```bash
# Setup
poetry install
poetry run pre-commit install

# Development cycle
poetry run black morphml tests
poetry run ruff morphml tests
poetry run mypy morphml
poetry run pytest

# Run example
poetry run morphml run examples/quickstart.py
```

---

## 📚 References

- **Architecture**: `docs/architecture.md` - Detailed system design
- **Flows**: `docs/flows.md` - Algorithm diagrams
- **Research**: `docs/research.md` - Theoretical background
- **Info**: `docs/info.md` - Project overview

---

## 💡 Key Design Decisions

### Why Pythonic DSL vs Text DSL?

**Decision**: Use Python builder pattern (Pythonic DSL)

**Rationale**:
- Better IDE support (autocomplete, type checking)
- Easier to debug (Python stack traces)
- No context switching for users
- Can leverage Python's full power (loops, conditionals)

**Trade-off**: Less portable than text format, but Phase 2 can add text serialization.

### Why NetworkX vs Custom Graph?

**Decision**: Use NetworkX internally

**Rationale**:
- Mature, well-tested library
- Efficient DAG operations
- Rich algorithm support
- Can wrap it with our own API

**Trade-off**: External dependency, but the benefits far outweigh the cost.

### Why Local Execution Only in Phase 1?

**Decision**: Defer distributed execution to Phase 3

**Rationale**:
- Simplifies initial implementation
- Local execution sufficient for testing and small experiments
- Distributed adds significant complexity
- Better to have solid foundation first

---

## 🎓 Learning Resources

### Python Best Practices
- PEP 8: https://peps.python.org/pep-0008/
- Type Hints: https://docs.python.org/3/library/typing.html
- Testing: https://docs.pytest.org/

### Genetic Algorithms
- Introduction to Evolutionary Algorithms (Eiben & Smith)
- DEAP library: https://deap.readthedocs.io/

### Neural Architecture Search
- Neural Architecture Search Survey (Elsken et al., 2019)
- DARTS: https://arxiv.org/abs/1806.09055

---

**Next Steps**: Proceed to `01_project_setup.md` to begin implementation.
