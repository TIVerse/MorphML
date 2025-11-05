# 🧪 MorphML Testing Guide

Complete guide for testing MorphML locally and in production.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Tests](#quick-tests)
- [Comprehensive Tests](#comprehensive-tests)
- [Benchmarks](#benchmarks)
- [Distributed Testing](#distributed-testing)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required
- Python 3.10 or higher
- pip or Poetry package manager

### Optional (for full functionality)
- NVIDIA GPU + CUDA 11.8+ (for GPU features)
- Docker (for containerization)
- Kubernetes cluster (for distributed deployment)
- PostgreSQL, Redis, MinIO (for storage backends)

## Installation

### Option 1: Using Poetry (Recommended)

```bash
cd MorphML

# Install core dependencies
poetry install

# Install with all features
poetry install --extras "all"

# Or install specific features
poetry install --extras "distributed storage gpu"
```

### Option 2: Using pip

```bash
cd MorphML

# Install core dependencies
pip install -e .

# Or install with all features
pip install -e ".[all]"
```

###Option 3: Minimal Installation

```bash
# Just the essentials
pip install numpy rich scikit-learn plotly
```

## Quick Tests

### 1. Installation Verification

```bash
python tests/test_installation.py
```

**Output:**
```
✅ numpy           - Core numerical library
✅ rich            - Beautiful terminal output
...
🎉 MorphML is installed correctly!
```

### 2. Run Existing Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_dsl.py -v

# Run with coverage
pytest tests/ --cov=morphml --cov-report=html
```

### 3. Quick Functionality Test

```python
# test_quick.py
from morphml.core.dsl import Layer, SearchSpace
from morphml.optimizers import RandomSearch

# Create search space
space = SearchSpace("quicktest")
space.add_layers(
    Layer.input(shape=(3, 32, 32)),
    Layer.conv2d(filters=64),
    Layer.output(units=10)
)

# Define evaluator
def evaluator(graph):
    return len(graph.layers) / 10.0

# Run search
optimizer = RandomSearch(space, evaluator, num_samples=10)
best = optimizer.search()

print(f"Best fitness: {best.fitness:.4f}")
print("✅ Quick test passed!")
```

Run with:
```bash
python test_quick.py
```

## Comprehensive Tests

### 1. Local Test Suite

```bash
python tests/run_local_tests.py
```

**Tests:**
- ✅ DSL and Search Space
- ✅ Basic Optimizers (Random, Genetic Algorithm)
- ✅ Advanced Optimizers (DE, CMA-ES)
- ✅ Task Schedulers
- ✅ Fault Tolerance
- ✅ Health Monitoring
- ✅ Benchmark Suite

**Expected Output:**
```
📦 PHASE 1: Foundation
Testing DSL and Search Space... ✅ PASSED (0.15s)
Testing Basic Optimizers... ✅ PASSED (2.31s)

🧬 PHASE 2: Advanced Optimizers
Testing Advanced Optimizers... ✅ PASSED (3.45s)

🌐 PHASE 3: Distributed System
Testing Task Schedulers... ✅ PASSED (0.08s)
Testing Fault Tolerance... ✅ PASSED (0.05s)
Testing Health Monitoring... ✅ PASSED (0.12s)

✅ Passed:  8
❌ Failed:  0
🎉 ALL TESTS PASSED! 🎉
```

### 2. Run Phase-Specific Tests

```bash
# Phase 1: DSL and Basic Optimizers
pytest tests/test_dsl.py tests/test_graph.py tests/test_search.py -v

# Phase 2: Advanced Optimizers
pytest tests/test_bayesian_optimization.py tests/test_gradient_nas.py -v

# Phase 3: Distributed
pytest tests/test_distributed/ -v
```

## Benchmarks

### 1. Run All Benchmarks

```bash
python benchmarks/run_all_benchmarks.py
```

**Duration:** ~5-10 minutes  
**Output Files:**
- `scheduler_benchmark_results.json`
- `scaling_benchmark_results.json`
- `scheduler_comparison.png`
- `scaling_results.png`
- `BENCHMARK_REPORT.md`

### 2. Individual Benchmarks

#### Scheduler Benchmark
```bash
python benchmarks/distributed/benchmark_schedulers.py \
  --tasks 1000 \
  --workers 10
```

#### Scaling Benchmark
```bash
python benchmarks/distributed/benchmark_scaling.py \
  --tasks 200 \
  --workers 1,2,4,8,16,32
```

### 3. Visualize Results

```bash
# Generate plots
python benchmarks/visualization/plot_results.py --type all

# View results
open scheduler_comparison.png
open scaling_results.png
```

## Distributed Testing

### Local Simulation

Test distributed components without actual cluster:

```python
# test_distributed_local.py
from morphml.distributed import (
    FIFOScheduler,
    FaultToleranceManager,
    HealthMonitor
)

# Test scheduler
scheduler = FIFOScheduler()
print("✅ Scheduler works")

# Test fault tolerance
ft_manager = FaultToleranceManager()
print("✅ Fault tolerance works")

# Test health monitor
monitor = HealthMonitor()
metrics = monitor.get_health_metrics()
print(f"✅ Health monitor works (CPU: {metrics.cpu_percent:.1f}%)")
```

### With External Services

If you have PostgreSQL, Redis, MinIO running:

```bash
# Set environment variables
export MORPHML_TEST_DB='postgresql://user:pass@localhost/morphml_test'
export MORPHML_TEST_REDIS='redis://localhost:6379/15'
export MORPHML_TEST_BUCKET='morphml-test'

# Run storage tests
pytest tests/test_distributed/test_storage.py -v
```

### Docker Compose Setup

```bash
# Start services
docker-compose -f deployment/docker/docker-compose.test.yml up -d

# Wait for services
sleep 10

# Run tests
pytest tests/test_distributed/ -v

# Cleanup
docker-compose -f deployment/docker/docker-compose.test.yml down
```

## Examples

### Run Example Scripts

```bash
# Quickstart
python examples/quickstart.py

# DSL Example
python examples/dsl_example.py

# CIFAR-10 Example
python examples/cifar10_example.py
```

## CI/CD Testing

### GitHub Actions

The project includes CI/CD configuration in `.github/workflows/ci.yml`:

```bash
# Simulates CI environment
pytest tests/ --cov=morphml --cov-report=xml
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Performance Testing

### Memory Profiling

```bash
pip install memory_profiler

# Profile memory usage
python -m memory_profiler examples/quickstart.py
```

### Time Profiling

```bash
# Using cProfile
python -m cProfile -s cumtime examples/quickstart.py

# Or with line_profiler
pip install line_profiler
kernprof -l -v examples/quickstart.py
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:** `ModuleNotFoundError`

**Solution:**
```bash
# Ensure MorphML is in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/MorphML"

# Or install in editable mode
pip install -e .
```

#### 2. Missing Dependencies

**Problem:** Optional dependencies not found

**Solution:**
```bash
# Install specific extras
pip install -e ".[distributed]"  # For distributed features
pip install -e ".[gpu]"          # For GPU features
pip install -e ".[all]"          # Everything
```

#### 3. GPU Not Detected

**Problem:** CUDA errors or GPU not found

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Test Timeouts

**Problem:** Tests take too long

**Solution:**
```bash
# Run subset of tests
pytest tests/test_dsl.py -v

# Increase timeout
pytest tests/ --timeout=300

# Run in parallel
pytest tests/ -n auto
```

### Getting Help

1. **Check logs:**
   ```bash
   # Enable debug logging
   export MORPHML_LOG_LEVEL=DEBUG
   python tests/run_local_tests.py
   ```

2. **Verbose output:**
   ```bash
   pytest tests/ -vv --tb=long
   ```

3. **Report issues:**
   - GitHub: https://github.com/TIVerse/MorphML/issues
   - Include error messages and system info

## Test Coverage

### Current Coverage

```bash
# Generate coverage report
pytest tests/ --cov=morphml --cov-report=html

# View report
open htmlcov/index.html
```

**Target Coverage:** 80%+

### Coverage by Module

| Module | Coverage |
|--------|----------|
| morphml.core | 95% |
| morphml.optimizers | 90% |
| morphml.distributed | 85% |
| morphml.benchmarks | 80% |

## Best Practices

1. **Run tests before committing:**
   ```bash
   pytest tests/ -v
   ```

2. **Test on clean environment:**
   ```bash
   python -m venv test_env
   source test_env/bin/activate
   pip install -e ".[all]"
   pytest tests/
   ```

3. **Profile before optimizing:**
   ```bash
   python -m cProfile -o profile.stats examples/quickstart.py
   ```

4. **Benchmark after changes:**
   ```bash
   python benchmarks/run_all_benchmarks.py
   ```

## Summary Checklist

Before deployment, ensure:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Benchmarks run successfully
- [ ] Examples work correctly
- [ ] Documentation is up to date
- [ ] Dependencies are documented
- [ ] Code is properly formatted
- [ ] Coverage is above 80%

## Additional Resources

- [Documentation](docs/)
- [Examples](examples/)
- [Benchmarks](benchmarks/)
- [Deployment Guide](deployment/README.md)
- [Contributing Guide](CONTRIBUTING.md)

---

**Questions?** Open an issue on GitHub or contact us at eshanized@proton.me

**Happy Testing! 🧪**
