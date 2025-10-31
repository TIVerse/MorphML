# Phase 2: Advanced Search - Overview

**Author:** Eshan Roy <eshanized@proton.me>  
**Organization:** TONMOY INFRASTRUCTURE & VISION  
**Repository:** https://github.com/TIVerse/MorphML  
**Phase Duration:** Months 7-12 (8-12 weeks)  
**Target LOC:** ~25,000 production + 4,000 tests  
**Prerequisites:** Phase 1 complete

---

## 🎯 Phase 2 Mission

Expand MorphML's optimization capabilities with:
1. **Bayesian Optimization** - Gaussian Process, TPE, SMAC
2. **Gradient-Based NAS** - DARTS, ENAS
3. **Multi-Objective Optimization** - NSGA-II, Pareto fronts
4. **Additional Evolutionary** - Differential Evolution, CMA-ES, Particle Swarm
5. **Benchmarking Suite** - OpenML, standard datasets

By the end of Phase 2, users can choose from multiple optimization strategies and optimize for multiple objectives simultaneously (accuracy, latency, memory).

---

## 📋 Success Criteria

### Functional Requirements
- ✅ Bayesian optimization converges faster than random/GA on benchmarks
- ✅ DARTS produces competitive architectures
- ✅ Multi-objective returns valid Pareto front
- ✅ All optimizers support same interface (ask/tell)
- ✅ Benchmarking suite runs on 5+ datasets

### Quality Requirements
- ✅ Test coverage >75% for new modules
- ✅ Type hints on all APIs
- ✅ Documentation with examples
- ✅ Performance: BO models train in <5s

### Performance Targets
- ✅ BO finds better solutions than GA in 50% fewer evaluations
- ✅ DARTS completes architecture search in <6 hours on single GPU
- ✅ Multi-objective discovers 20+ Pareto-optimal solutions

---

## 🏗️ Architecture Overview

### New Components

```
morphml/
├── optimizers/
│   ├── bayesian/              # NEW: Bayesian optimization
│   │   ├── gaussian_process.py
│   │   ├── tpe.py
│   │   ├── smac.py
│   │   └── acquisition.py
│   ├── gradient_based/        # NEW: Gradient-based NAS
│   │   ├── darts.py
│   │   ├── enas.py
│   │   └── differentiable_graph.py
│   ├── evolutionary/          # EXTEND: More algorithms
│   │   ├── differential_evolution.py
│   │   ├── cma_es.py
│   │   └── particle_swarm.py
│   └── multi_objective/       # NEW: Multi-objective
│       ├── nsga2.py
│       ├── pareto.py
│       └── indicators.py
│
├── core/
│   └── objectives/            # EXTEND: Multi-objective support
│       ├── multi_objective.py (expand)
│       ├── pareto_dominance.py
│       └── hypervolume.py
│
├── benchmarks/                # NEW: Benchmarking
│   ├── datasets.py
│   ├── runners.py
│   ├── metrics.py
│   └── openml_suite.py
│
└── visualization/             # NEW: Advanced plots
    ├── pareto_plot.py
    ├── convergence_plot.py
    └── architecture_plot.py
```

---

## 📦 Phase 2 Components

### Component 1: Bayesian Optimization (Weeks 1-2)
**File:** `01_bayesian_optimization.md`

**Scope:**
- Gaussian Process surrogate model
- Tree-structured Parzen Estimator (TPE)
- SMAC (Sequential Model-based Algorithm Configuration)
- Acquisition functions (EI, UCB, PI)
- Integration with search space

**LOC:** ~5,000

### Component 2: Gradient-Based NAS (Weeks 3-4)
**File:** `02_gradient_based_nas.md`

**Scope:**
- DARTS (Differentiable Architecture Search)
- ENAS (Efficient Neural Architecture Search)
- Differentiable graph representation
- Bilevel optimization
- Architecture discretization

**LOC:** ~6,000

### Component 3: Multi-Objective Optimization (Week 5)
**File:** `03_multi_objective.md`

**Scope:**
- NSGA-II implementation
- Pareto dominance and ranking
- Crowding distance
- Hypervolume indicator
- Multi-objective evaluator

**LOC:** ~4,000

### Component 4: Advanced Evolutionary (Week 6)
**File:** `04_advanced_evolutionary.md`

**Scope:**
- Differential Evolution
- CMA-ES (Covariance Matrix Adaptation)
- Particle Swarm Optimization
- Hybrid strategies

**LOC:** ~5,000

### Component 5: Benchmarking & Visualization (Weeks 7-8)
**File:** `05_benchmarking_visualization.md`

**Scope:**
- OpenML integration
- Standard dataset loaders (CIFAR-10, ImageNet, etc.)
- Benchmark runners
- Performance metrics
- Visualization tools (Pareto plots, convergence curves)

**LOC:** ~5,000

---

## 🔧 Key Technologies

### New Dependencies

```toml
# Bayesian Optimization
scikit-optimize = "^0.9.0"     # For GP and acquisition functions
gpytorch = "^1.9.0"            # Advanced GP models
botorch = "^0.8.0"             # Bayesian optimization library

# Gradient-Based
torch = "^2.0.0"               # For DARTS/ENAS
torch-geometric = "^2.3.0"     # Graph neural networks

# Multi-Objective
pymoo = "^0.6.0"               # Multi-objective optimization

# Benchmarking
openml = "^0.13.0"             # OpenML datasets
scikit-learn = "^1.3.0"        # Dataset utilities
```

---

## 📊 Optimizer Comparison Matrix

| Optimizer | Sample Efficiency | Scalability | Multi-Obj | GPU Required |
|-----------|------------------|-------------|-----------|--------------|
| Genetic Algorithm | ⭐⭐ | ⭐⭐⭐⭐ | ✅ | ❌ |
| Bayesian (GP) | ⭐⭐⭐⭐ | ⭐⭐ | ✅ | ❌ |
| Bayesian (TPE) | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | ❌ |
| DARTS | ⭐⭐⭐ | ⭐⭐⭐ | ❌ | ✅ |
| ENAS | ⭐⭐⭐⭐ | ⭐⭐ | ❌ | ✅ |
| Differential Evolution | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ❌ |
| CMA-ES | ⭐⭐⭐⭐ | ⭐⭐ | ❌ | ❌ |
| Particle Swarm | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ❌ |
| NSGA-II | ⭐⭐ | ⭐⭐⭐ | ✅ | ❌ |

---

## 🎯 Usage Examples

### Bayesian Optimization

```python
from morphml.optimizers.bayesian import GaussianProcessOptimizer

optimizer = GaussianProcessOptimizer(
    search_space=space,
    config={
        'acquisition': 'ei',  # Expected Improvement
        'kernel': 'matern',
        'n_initial_points': 10,
        'max_iterations': 100
    }
)

results = executor.run(space, optimizer, evaluator, max_evaluations=100)
```

### Multi-Objective

```python
from morphml.optimizers.multi_objective import NSGA2

optimizer = NSGA2(
    search_space=space,
    config={
        'population_size': 100,
        'objectives': ['maximize:accuracy', 'minimize:latency', 'minimize:params']
    }
)

results = executor.run(space, optimizer, evaluator, max_evaluations=500)
pareto_front = results['pareto_front']

# Visualize
from morphml.visualization import plot_pareto_front
plot_pareto_front(pareto_front, objectives=['accuracy', 'latency'])
```

### DARTS

```python
from morphml.optimizers.gradient_based import DARTS

optimizer = DARTS(
    search_space=space,
    config={
        'epochs': 50,
        'learning_rate': 0.025,
        'weight_decay': 3e-4
    }
)

# Requires GPU
results = executor.run(space, optimizer, evaluator, max_evaluations=50)
```

---

## 🧪 Testing Strategy

### Unit Tests
- Each optimizer: initialization, ask/tell, convergence
- Bayesian models: surrogate accuracy, acquisition optimization
- DARTS: architecture discretization, bilevel optimization
- Multi-objective: dominance, crowding, hypervolume

### Integration Tests
- Compare optimizers on toy problems (known optima)
- Benchmark suite end-to-end
- Multi-objective Pareto front validation

### Performance Tests
- Sample efficiency comparison
- Convergence rate analysis
- Computational cost benchmarks

---

## 📈 Benchmarking Plan

### Datasets
1. **CIFAR-10**: Image classification (50k train, 10k test)
2. **MNIST**: Digit recognition (baseline)
3. **Fashion-MNIST**: Fashion item classification
4. **SVHN**: Street View House Numbers
5. **Tiny ImageNet**: Subset of ImageNet

### Metrics
- **Accuracy**: Final test accuracy
- **Sample Efficiency**: Evaluations to reach target accuracy
- **Time Efficiency**: Wall-clock time
- **Pareto Quality**: Hypervolume indicator for multi-objective

### Experiments
```python
# Benchmark all optimizers on CIFAR-10
from morphml.benchmarks import run_benchmark

results = run_benchmark(
    dataset='cifar10',
    optimizers=['genetic', 'gp', 'tpe', 'darts', 'nsga2'],
    budget=500,
    num_runs=5  # For statistical significance
)

results.plot_comparison()
results.save_report('benchmark_results.pdf')
```

---

## 📝 Code Quality Standards

### Same as Phase 1
- Type hints on all APIs
- Docstrings with examples
- <50 lines per function
- >75% test coverage
- Black + Ruff + MyPy passing

### Additional for Phase 2
- Benchmark results must be reproducible (fixed seeds)
- GPU code must handle CUDA availability gracefully
- Multi-objective code must handle conflicting objectives
- All optimizers must implement BaseOptimizer interface

---

## 🔄 Integration with Phase 1

### Backward Compatibility
- All Phase 1 code continues to work
- No breaking changes to existing APIs
- New optimizers follow same interface

### Enhancements
- SearchSpace gains multi-objective support
- ModelGraph adds parameter counting
- Evaluator supports multiple objectives
- CLI adds optimizer selection flag

---

## 📚 Learning Resources

### Bayesian Optimization
- "A Tutorial on Bayesian Optimization" (Frazier, 2018)
- "Practical Bayesian Optimization of Machine Learning Algorithms" (Snoek et al., 2012)

### Gradient-Based NAS
- "DARTS: Differentiable Architecture Search" (Liu et al., 2019)
- "Efficient Neural Architecture Search via Parameter Sharing" (Pham et al., 2018)

### Multi-Objective
- "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II" (Deb et al., 2002)
- "Multiobjective Optimization" (pymoo documentation)

---

## ✅ Phase 2 Deliverables Checklist

### Code
- [ ] Bayesian optimization (GP, TPE, SMAC)
- [ ] Gradient-based NAS (DARTS, ENAS)
- [ ] Multi-objective (NSGA-II, Pareto)
- [ ] Advanced evolutionary (DE, CMA-ES, PSO)
- [ ] Benchmarking suite
- [ ] Visualization tools

### Tests
- [ ] Unit tests >75% coverage
- [ ] Integration tests for each optimizer
- [ ] Benchmark comparisons

### Documentation
- [ ] API docs for new modules
- [ ] Examples for each optimizer
- [ ] Benchmark report

### Performance
- [ ] BO achieves 2x sample efficiency vs GA
- [ ] DARTS runs on GPU
- [ ] Multi-objective discovers valid Pareto fronts

---

**Next:** Proceed to `01_bayesian_optimization.md` to begin Phase 2 implementation.
