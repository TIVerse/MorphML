# MorphML Testing & Metrics Guide

**Author:** Eshan Roy <eshanized@proton.me>  
**Organization:** TONMOY INFRASTRUCTURE & VISION  
**Last Updated:** November 6, 2025

---

## Table of Contents

1. [Testing Overview](#testing-overview)
2. [Test Categories](#test-categories)
3. [Running Tests](#running-tests)
4. [Benchmarking](#benchmarking)
5. [Metrics & KPIs](#metrics--kpis)
6. [Performance Standards](#performance-standards)
7. [CI/CD Integration](#cicd-integration)

---

## Testing Overview

MorphML has a comprehensive testing strategy covering:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end distributed system tests
- **Performance Tests**: Throughput, latency, and scaling tests
- **Validation Tests**: Helm chart and Kubernetes manifest validation
- **Benchmark Tests**: Optimizer comparison on standard problems

### Test Coverage

- **Current Coverage**: 76%
- **Target Coverage**: >75%
- **Total Test Files**: 41
- **Total Tests**: 90+

---

## Test Categories

### 1. Unit Tests

**Location**: `tests/`

**Coverage**:
- DSL components (`test_dsl*.py`)
- Graph operations (`test_graph.py`)
- Optimizers (`test_genetic_algorithm.py`, etc.)
- Search space (`test_search_space.py`)
- Mutations (`test_mutations.py`)

**Run**:
```bash
pytest tests/ -v -m "not slow and not integration"
```

### 2. Integration Tests

**Location**: `tests/test_distributed/test_integration_e2e.py`

**Coverage**:
- Master-worker communication
- Task distribution
- Failure recovery
- Heartbeat monitoring
- Load balancing
- Checkpointing

**Run**:
```bash
pytest tests/test_distributed/test_integration_e2e.py -v -m integration
```

### 3. Performance Tests

**Location**: `tests/test_performance.py`

**Coverage**:
- Optimizer throughput
- Graph operation speed
- Memory usage
- Scaling characteristics
- Parallel execution

**Run**:
```bash
pytest tests/test_performance.py -v
```

**Performance Targets**:
| Metric | Target | Current |
|--------|--------|---------|
| Graph creation | >100/sec | ✅ |
| Graph cloning | >200/sec | ✅ |
| Random sampling | >50/sec | ✅ |
| GA evolution | <5s/10gen | ✅ |

### 4. Helm Validation Tests

**Location**: `tests/test_helm_validation.py`

**Coverage**:
- Chart structure validation
- Template syntax checking
- Resource configuration validation
- Security configuration checks
- Documentation validation

**Run**:
```bash
pytest tests/test_helm_validation.py -v
```

### 5. Stress Tests

**Location**: `tests/test_performance.py` (marked as `slow`)

**Coverage**:
- Large populations (500+)
- Long-running experiments (1000+ iterations)
- High concurrency

**Run**:
```bash
pytest tests/ -v -m slow
```

---

## Running Tests

### Quick Test

Run fast tests only:
```bash
pytest tests/ -v --maxfail=3
```

### Full Test Suite

Run all tests including slow ones:
```bash
pytest tests/ -v
```

### With Coverage Report

```bash
pytest tests/ --cov=morphml --cov-report=html --cov-report=term
```

### Specific Test Categories

```bash
# Unit tests only
pytest tests/ -v -m "not slow and not integration"

# Integration tests only
pytest tests/ -v -m integration

# Performance tests only
pytest tests/test_performance.py -v

# Slow/stress tests only
pytest tests/ -v -m slow
```

### Using Test Runner Script

Run comprehensive test suite:
```bash
python scripts/run_all_tests.py
```

This will:
1. Run unit tests
2. Run integration tests
3. Run performance tests
4. Run Helm validation
5. Check code quality
6. Generate JSON report

---

## Benchmarking

### Benchmark Suite

**Location**: `benchmarks/run_benchmarks.py`

**Purpose**: Compare all optimizers on standard search spaces

**Run**:
```bash
python benchmarks/run_benchmarks.py
```

### What It Tests

1. **Random Search** - Baseline
2. **Genetic Algorithm** - Evolutionary search
3. **Hill Climbing** - Local search
4. **Bayesian Optimization** - GP-based (if available)
5. **TPE** - Tree-structured Parzen Estimator (if available)
6. **NSGA-II** - Multi-objective (if available)

### Benchmark Configuration

```python
config = BenchmarkConfig(
    max_evaluations=100,     # Evaluations per run
    num_runs=3,              # Runs per optimizer (for statistics)
    output_dir="benchmark_results",
    save_results=True
)
```

### Benchmark Metrics

For each optimizer:
- **Avg Best Fitness**: Mean best fitness across runs
- **Std Dev**: Standard deviation of best fitness
- **Avg Convergence**: Average iterations to convergence
- **Avg Time**: Average wall-clock time
- **Evals/sec**: Throughput (evaluations per second)

### Example Output

```
┌──────────────────────┬───────────────────┬──────────┬─────────────────┬──────────────┬───────────┐
│ Optimizer            │ Avg Best Fitness  │ Std Dev  │ Avg Convergence │ Avg Time (s) │ Evals/sec │
├──────────────────────┼───────────────────┼──────────┼─────────────────┼──────────────┼───────────┤
│ RandomSearch         │ 0.8234            │ ±0.0312  │ 87              │ 10.23        │ 9.78      │
│ GeneticAlgorithm     │ 0.8956            │ ±0.0189  │ 64              │ 12.45        │ 8.03      │
│ HillClimbing         │ 0.8567            │ ±0.0234  │ 72              │ 11.12        │ 8.99      │
│ GaussianProcess      │ 0.9123            │ ±0.0156  │ 45              │ 15.67        │ 6.38      │
│ TPEOptimizer         │ 0.9087            │ ±0.0167  │ 48              │ 14.89        │ 6.72      │
└──────────────────────┴───────────────────┴──────────┴─────────────────┴──────────────┴───────────┘

🏆 Best Optimizer: GaussianProcess (fitness: 0.9123)
```

---

## Metrics & KPIs

### Experiment Metrics

**Tracked by**: `benchmarks/metrics_tracker.py`

#### Fitness Metrics
- **best_fitness**: Highest fitness achieved
- **mean_fitness**: Average fitness across evaluations
- **fitness_std**: Standard deviation
- **fitness_history**: Complete history of fitness values

#### Performance Metrics
- **total_evaluations**: Total architectures evaluated
- **successful_evaluations**: Successful evaluations
- **failed_evaluations**: Failed evaluations
- **evaluation_times**: Time per evaluation

#### Convergence Metrics
- **convergence_iteration**: When improvement stopped
- **convergence_threshold**: Threshold for convergence (default: 0.001)

#### Architecture Metrics
- **architecture_depths**: Depth of evaluated architectures
- **architecture_params**: Parameter counts
- **best_architecture**: Best architecture found

### Usage

```python
from benchmarks.metrics_tracker import MetricsTracker

tracker = MetricsTracker("my_experiment")
metrics = tracker.create_experiment("exp_001", "GeneticAlgorithm")

for iteration in range(100):
    # ... run optimization ...
    metrics.update(iteration, fitness, architecture_dict)
    metrics.add_evaluation_time(eval_time)
    metrics.add_architecture_stats(depth, params)

tracker.save_experiment("exp_001")
report = tracker.generate_report()
```

### Standard KPIs

1. **Sample Efficiency**
   - Definition: Evaluations to reach target fitness
   - Target: <50 for fitness=0.9
   - Calculation: `PerformanceMetrics.calculate_sample_efficiency()`

2. **Convergence Rate**
   - Definition: Average improvement per evaluation
   - Target: >0.01
   - Calculation: `PerformanceMetrics.calculate_convergence_rate()`

3. **Exploration Ratio**
   - Definition: Unique architectures / total evaluations
   - Target: >0.8
   - Calculation: `PerformanceMetrics.calculate_exploration_ratio()`

4. **Cumulative Regret**
   - Definition: Sum of (optimal - actual) fitness
   - Target: Minimize
   - Calculation: `PerformanceMetrics.calculate_regret()`

5. **Hypervolume** (for multi-objective)
   - Definition: Volume of dominated space
   - Target: Maximize
   - Calculation: `PerformanceMetrics.calculate_hypervolume()`

---

## Performance Standards

### Optimizer Performance

| Optimizer | Sample Efficiency | Convergence Rate | Time Complexity |
|-----------|-------------------|------------------|-----------------|
| Random Search | Low | N/A | O(n) |
| Genetic Algorithm | Medium | Medium | O(n·p·g) |
| Hill Climbing | Medium | High | O(n·k) |
| Bayesian (GP) | High | High | O(n³) |
| DARTS | High | Very High | O(n·e) |

**Legend**:
- n = evaluations
- p = population size
- g = generations
- k = neighbors explored
- e = epochs

### Throughput Targets

| Operation | Target | Achieved |
|-----------|--------|----------|
| Graph sampling | >100/sec | ✅ 200+/sec |
| Graph cloning | >200/sec | ✅ 500+/sec |
| Graph mutation | >20/sec | ✅ 50+/sec |
| Evaluation (mock) | >10/sec | ✅ 10/sec |
| Evaluation (real) | >0.1/sec | Depends on model |

### Memory Constraints

| Component | Max Memory | Current |
|-----------|------------|---------|
| Population (100) | <10 MB | ✅ ~5 MB |
| Graph | <100 KB | ✅ ~50 KB |
| History (1000) | <50 MB | ✅ ~20 MB |

### Scaling Targets

| Workers | Throughput | Efficiency | Status |
|---------|------------|------------|--------|
| 1 | 1x | 100% | ✅ |
| 5 | 4.5x | 90% | ⚠️ Untested |
| 10 | 8.5x | 85% | ⚠️ Untested |
| 50 | 40x | 80% | ⚠️ Untested |

---

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install poetry
        poetry install
    
    - name: Run tests
      run: |
        poetry run pytest tests/ -v --cov=morphml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pre-commit install
```

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.0.270
    hooks:
      - id: ruff
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### Test on Save

Add to VSCode settings:
```json
{
  "python.testing.pytestEnabled": true,
  "python.testing.autoTestDiscoverOnSaveEnabled": true
}
```

---

## Continuous Benchmarking

### Nightly Benchmarks

Schedule benchmarks to run nightly:

```bash
# crontab -e
0 2 * * * cd /path/to/MorphML && python benchmarks/run_benchmarks.py
```

### Regression Detection

Compare new results with baseline:

```python
from benchmarks.metrics_tracker import MetricsTracker

tracker = MetricsTracker()
current_report = tracker.generate_report()

# Load baseline
with open("baseline_report.json") as f:
    baseline = json.load(f)

# Compare
for optimizer in current_report["comparison"]:
    current_fitness = current_report["comparison"][optimizer]["avg_best_fitness"]
    baseline_fitness = baseline["comparison"][optimizer]["avg_best_fitness"]
    
    regression = (baseline_fitness - current_fitness) / baseline_fitness
    if regression > 0.05:  # 5% regression
        print(f"⚠️  Regression detected in {optimizer}: {regression:.2%}")
```

---

## Best Practices

### Writing Tests

1. **Use fixtures** for common setup
2. **Parameterize** tests for multiple inputs
3. **Mock expensive operations** (training, GPU)
4. **Test edge cases** (empty graphs, invalid inputs)
5. **Keep tests fast** (<1s per test)

### Running Tests Locally

Before committing:
```bash
# Format code
black morphml/
ruff check morphml/ --fix

# Type check
mypy morphml/ --ignore-missing-imports

# Run tests
pytest tests/ -v --cov=morphml

# Run benchmarks (optional)
python benchmarks/run_benchmarks.py
```

### Debugging Failed Tests

```bash
# Run with more detail
pytest tests/test_file.py::test_name -vv -s

# Drop into debugger on failure
pytest tests/ --pdb

# Show full traceback
pytest tests/ --tb=long
```

---

## Monitoring Production

### Prometheus Metrics

Expose metrics from master and workers:

```python
from prometheus_client import Counter, Histogram, Gauge

tasks_completed = Counter('morphml_tasks_completed_total', 'Total completed tasks')
task_duration = Histogram('morphml_task_duration_seconds', 'Task duration')
best_fitness = Gauge('morphml_best_fitness', 'Current best fitness')
```

### Grafana Dashboards

Import dashboard from `deployment/monitoring/grafana-dashboard.json`

Key panels:
- Active workers
- Task throughput
- GPU utilization
- Best fitness over time
- Error rate

---

## Troubleshooting

### Tests Taking Too Long

```bash
# Run only fast tests
pytest tests/ -v -m "not slow"

# Run with parallelization
pytest tests/ -n auto  # requires pytest-xdist
```

### Import Errors

```bash
# Install in editable mode
pip install -e .

# Or use poetry
poetry install
```

### GPU Tests Failing

```bash
# Skip GPU tests if no GPU
pytest tests/ -v -m "not gpu"
```

---

## References

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)

---

**Happy Testing! ✅**
