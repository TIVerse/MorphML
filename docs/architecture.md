# 🏗️ MorphML Architecture Documentation

**Version:** 1.0  
**Last Updated:** October 31, 2025  
**Status:** Design Phase

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Design Principles](#design-principles)
3. [System Components](#system-components)
4. [Complete Directory Structure](#complete-directory-structure)
5. [Data Flow](#data-flow)
6. [API Specifications](#api-specifications)
7. [Integration Points](#integration-points)
8. [Deployment Architecture](#deployment-architecture)

---

## 🎯 Architecture Overview

MorphML follows a **layered, modular architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│           CLI • REST API • Dashboard • Notebooks            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                       │
│        Experiment Manager • Task Scheduler • Monitor        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Core Engine Layer                       │
│         DSL Compiler • Search Engine • Graph System         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Optimization Layer                        │
│    Evolutionary • Bayesian • Gradient • RL • Meta-Learn    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Execution Layer                           │
│      Local Executor • Distributed Workers • Evaluators      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Storage & Cache Layer                      │
│         Result DB • Checkpoint Store • Artifact Cache       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎨 Design Principles

### 1. **Modularity**
Every component is a pluggable module with well-defined interfaces.

### 2. **Extensibility**
Adding new optimizers, model types, or backends requires minimal code changes.

### 3. **Transparency**
All internal states are inspectable; no hidden magic.

### 4. **Composability**
Complex systems are built by composing simple primitives.

### 5. **Performance**
Critical paths are optimized; support for distributed execution.

### 6. **Reproducibility**
Every experiment is fully deterministic and replayable.

---

## 🧩 System Components

### Component Interaction Map

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ DSL Parser   │────▶│   Compiler   │────▶│ Experiment   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
                                                   ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Search Space │◀────│Search Engine │◀────│ Orchestrator │
└──────────────┘     └──────────────┘     └──────────────┘
       │                     │                     │
       ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Model Graph  │────▶│  Optimizer   │────▶│   Executor   │
└──────────────┘     └──────────────┘     └──────────────┘
       │                     │                     │
       ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Evaluator   │────▶│Result Logger │────▶│  Storage     │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## 📁 Complete Directory Structure

```
morphml/
│
├── README.md                           # 200 LOC
├── LICENSE                             # 50 LOC
├── pyproject.toml                      # 150 LOC
├── setup.py                            # 100 LOC
├── CONTRIBUTING.md                     # 300 LOC
├── CODE_OF_CONDUCT.md                  # 100 LOC
├── CHANGELOG.md                        # 500 LOC
│
├── morphml/                            # Main package
│   ├── __init__.py                     # 50 LOC
│   ├── version.py                      # 30 LOC
│   ├── config.py                       # 200 LOC
│   ├── exceptions.py                   # 150 LOC
│   ├── logging_config.py               # 100 LOC
│   │
│   ├── core/                           # Core abstractions (8,500 LOC)
│   │   ├── __init__.py                 # 100 LOC
│   │   │
│   │   ├── dsl/                        # DSL implementation (3,500 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── lexer.py                # 800 LOC - Tokenization
│   │   │   ├── parser.py               # 1200 LOC - AST generation
│   │   │   ├── ast_nodes.py            # 600 LOC - AST node definitions
│   │   │   ├── compiler.py             # 800 LOC - AST → Internal IR
│   │   │   ├── validator.py            # 400 LOC - Semantic validation
│   │   │   ├── type_system.py          # 350 LOC - Type checking
│   │   │   └── syntax.py               # 300 LOC - Grammar definitions
│   │   │
│   │   ├── search/                     # Search engine core (2,500 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── search_space.py         # 800 LOC - Space definition
│   │   │   ├── search_engine.py        # 600 LOC - Base engine
│   │   │   ├── population.py           # 400 LOC - Population management
│   │   │   ├── selection.py            # 300 LOC - Selection strategies
│   │   │   └── constraints.py          # 350 LOC - Constraint handling
│   │   │
│   │   ├── graph/                      # Model graph system (2,000 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── node.py                 # 400 LOC - Graph nodes
│   │   │   ├── edge.py                 # 200 LOC - Connections
│   │   │   ├── graph.py                # 600 LOC - Graph structure
│   │   │   ├── mutations.py            # 400 LOC - Graph mutations
│   │   │   ├── serialization.py        # 250 LOC - Save/load
│   │   │   └── visualization.py        # 100 LOC - Graph plotting
│   │   │
│   │   └── objectives/                 # Objective functions (500 LOC)
│   │       ├── __init__.py             # 50 LOC
│   │       ├── base.py                 # 150 LOC - Base classes
│   │       ├── single_objective.py     # 100 LOC
│   │       ├── multi_objective.py      # 150 LOC - Pareto optimization
│   │       └── constraints.py          # 50 LOC
│   │
│   ├── optimizers/                     # Optimization algorithms (25,000 LOC)
│   │   ├── __init__.py                 # 100 LOC
│   │   ├── base.py                     # 300 LOC - Base optimizer
│   │   │
│   │   ├── evolutionary/               # Evolutionary algorithms (8,000 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── genetic.py              # 1200 LOC - Genetic algorithm
│   │   │   ├── differential_evolution.py # 800 LOC - DE
│   │   │   ├── cma_es.py               # 1000 LOC - CMA-ES
│   │   │   ├── particle_swarm.py       # 700 LOC - PSO
│   │   │   ├── operators/              # Genetic operators (3,000 LOC)
│   │   │   │   ├── __init__.py         # 50 LOC
│   │   │   │   ├── mutation.py         # 800 LOC
│   │   │   │   ├── crossover.py        # 800 LOC
│   │   │   │   ├── selection.py        # 600 LOC
│   │   │   │   └── elitism.py          # 250 LOC
│   │   │   └── utils.py                # 500 LOC
│   │   │
│   │   ├── bayesian/                   # Bayesian optimization (7,000 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── gaussian_process.py     # 1500 LOC - GP implementation
│   │   │   ├── tpe.py                  # 1200 LOC - Tree-structured Parzen
│   │   │   ├── smac.py                 # 1000 LOC - SMAC
│   │   │   ├── acquisition/            # Acquisition functions (2,000 LOC)
│   │   │   │   ├── __init__.py         # 50 LOC
│   │   │   │   ├── expected_improvement.py # 400 LOC
│   │   │   │   ├── ucb.py              # 300 LOC
│   │   │   │   ├── probability_improvement.py # 300 LOC
│   │   │   │   └── entropy_search.py   # 450 LOC
│   │   │   ├── surrogate/              # Surrogate models (1,000 LOC)
│   │   │   │   ├── __init__.py         # 50 LOC
│   │   │   │   ├── gp_kernels.py       # 400 LOC
│   │   │   │   └── random_forest.py    # 350 LOC
│   │   │   └── utils.py                # 250 LOC
│   │   │
│   │   ├── gradient_based/             # Gradient-based NAS (5,000 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── darts.py                # 1500 LOC - DARTS
│   │   │   ├── enas.py                 # 1200 LOC - ENAS
│   │   │   ├── snas.py                 # 800 LOC - SNAS
│   │   │   ├── weight_sharing.py       # 600 LOC
│   │   │   ├── supernet.py             # 500 LOC
│   │   │   └── architecture_params.py  # 350 LOC
│   │   │
│   │   ├── reinforcement/              # RL-based search (4,000 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── ppo.py                  # 1200 LOC - PPO
│   │   │   ├── a3c.py                  # 1000 LOC - A3C
│   │   │   ├── policy_network.py       # 600 LOC
│   │   │   ├── value_network.py        # 400 LOC
│   │   │   ├── environment.py          # 500 LOC - RL env
│   │   │   └── replay_buffer.py        # 250 LOC
│   │   │
│   │   └── hybrid/                     # Hybrid methods (1,000 LOC)
│   │       ├── __init__.py             # 50 LOC
│   │       ├── local_search.py         # 400 LOC
│   │       └── multi_fidelity.py       # 550 LOC
│   │
│   ├── distributed/                    # Distributed execution (20,000 LOC)
│   │   ├── __init__.py                 # 100 LOC
│   │   │
│   │   ├── orchestrator/               # Master coordinator (5,000 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── master.py               # 1500 LOC - Master node
│   │   │   ├── experiment_manager.py   # 1000 LOC
│   │   │   ├── task_queue.py           # 800 LOC
│   │   │   ├── resource_manager.py     # 700 LOC
│   │   │   ├── heartbeat.py            # 400 LOC
│   │   │   └── failure_recovery.py     # 550 LOC
│   │   │
│   │   ├── worker/                     # Worker nodes (4,500 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── worker.py               # 1200 LOC - Worker process
│   │   │   ├── executor.py             # 1000 LOC - Task execution
│   │   │   ├── gpu_manager.py          # 600 LOC
│   │   │   ├── resource_monitor.py     # 500 LOC
│   │   │   └── sandbox.py              # 650 LOC - Isolated execution
│   │   │
│   │   ├── scheduler/                  # Task scheduling (4,000 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── base_scheduler.py       # 400 LOC
│   │   │   ├── fifo_scheduler.py       # 300 LOC
│   │   │   ├── priority_scheduler.py   # 500 LOC
│   │   │   ├── fair_share_scheduler.py # 600 LOC
│   │   │   ├── gang_scheduler.py       # 700 LOC
│   │   │   ├── load_balancer.py        # 800 LOC
│   │   │   └── placement_policy.py     # 650 LOC
│   │   │
│   │   ├── storage/                    # Distributed storage (3,500 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── result_store.py         # 800 LOC
│   │   │   ├── checkpoint_manager.py   # 700 LOC
│   │   │   ├── artifact_cache.py       # 600 LOC
│   │   │   ├── distributed_fs.py       # 500 LOC
│   │   │   └── compression.py          # 350 LOC
│   │   │
│   │   └── communication/              # Inter-node comm (3,000 LOC)
│   │       ├── __init__.py             # 50 LOC
│   │       ├── message_protocol.py     # 600 LOC
│   │       ├── rpc_server.py           # 700 LOC
│   │       ├── rpc_client.py           # 600 LOC
│   │       ├── serialization.py        # 500 LOC
│   │       └── encryption.py           # 550 LOC
│   │
│   ├── meta_learning/                  # Meta-learning (15,000 LOC)
│   │   ├── __init__.py                 # 100 LOC
│   │   │
│   │   ├── warmstart/                  # Warm-starting (4,500 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── transfer_optimizer.py   # 1200 LOC
│   │   │   ├── initial_population.py   # 800 LOC
│   │   │   ├── prior_extractor.py      # 900 LOC
│   │   │   ├── domain_adaptation.py    # 700 LOC
│   │   │   └── similarity_metrics.py   # 850 LOC
│   │   │
│   │   ├── predictors/                 # Performance prediction (5,000 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── base_predictor.py       # 400 LOC
│   │   │   ├── neural_predictor.py     # 1500 LOC - Neural net
│   │   │   ├── gbm_predictor.py        # 800 LOC - Gradient boosting
│   │   │   ├── early_stopping.py       # 600 LOC
│   │   │   ├── learning_curve.py       # 700 LOC
│   │   │   └── uncertainty.py          # 450 LOC
│   │   │
│   │   ├── strategy_evolution/         # Strategy optimization (3,500 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── strategy_optimizer.py   # 1000 LOC
│   │   │   ├── meta_features.py        # 800 LOC
│   │   │   ├── strategy_embedding.py   # 600 LOC
│   │   │   └── portfolio_selection.py  # 550 LOC
│   │   │
│   │   └── knowledge_base/             # Experience storage (2,000 LOC)
│   │       ├── __init__.py             # 50 LOC
│   │       ├── experience_db.py        # 800 LOC
│   │       ├── indexing.py             # 500 LOC
│   │       └── retrieval.py            # 650 LOC
│   │
│   ├── integrations/                   # Framework integrations (10,000 LOC)
│   │   ├── __init__.py                 # 100 LOC
│   │   │
│   │   ├── sklearn/                    # Scikit-learn (2,500 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── adapter.py              # 800 LOC
│   │   │   ├── pipeline_builder.py     # 600 LOC
│   │   │   ├── estimators.py           # 500 LOC
│   │   │   └── transformers.py         # 550 LOC
│   │   │
│   │   ├── pytorch/                    # PyTorch (2,500 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── module_adapter.py       # 800 LOC
│   │   │   ├── graph_to_module.py      # 700 LOC
│   │   │   ├── training_loop.py        # 500 LOC
│   │   │   └── optimization.py         # 450 LOC
│   │   │
│   │   ├── tensorflow/                 # TensorFlow (2,500 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── keras_adapter.py        # 800 LOC
│   │   │   ├── graph_to_keras.py       # 700 LOC
│   │   │   ├── training_loop.py        # 500 LOC
│   │   │   └── optimization.py         # 450 LOC
│   │   │
│   │   ├── jax/                        # JAX (1,500 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── flax_adapter.py         # 600 LOC
│   │   │   ├── graph_to_flax.py        # 500 LOC
│   │   │   └── training_loop.py        # 350 LOC
│   │   │
│   │   └── ray/                        # Ray Tune (1,000 LOC)
│   │       ├── __init__.py             # 50 LOC
│   │       ├── tune_adapter.py         # 500 LOC
│   │       └── trainable.py            # 450 LOC
│   │
│   ├── visualization/                  # Visualization tools (8,000 LOC)
│   │   ├── __init__.py                 # 100 LOC
│   │   │
│   │   ├── dashboard/                  # Web dashboard (4,000 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── app.py                  # 800 LOC - FastAPI app
│   │   │   ├── routes/
│   │   │   │   ├── experiments.py      # 400 LOC
│   │   │   │   ├── models.py           # 400 LOC
│   │   │   │   └── monitoring.py       # 350 LOC
│   │   │   ├── frontend/               # React app (1,500 LOC JS)
│   │   │   │   ├── src/
│   │   │   │   │   ├── App.jsx         # 200 LOC
│   │   │   │   │   ├── ExperimentView.jsx # 300 LOC
│   │   │   │   │   ├── GraphView.jsx   # 300 LOC
│   │   │   │   │   ├── MetricsView.jsx # 250 LOC
│   │   │   │   │   └── LiveMonitor.jsx # 250 LOC
│   │   │   │   └── package.json        # 50 LOC
│   │   │   └── websocket.py            # 500 LOC
│   │   │
│   │   ├── plotting/                   # Static plots (2,500 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── convergence.py          # 400 LOC
│   │   │   ├── pareto_front.py         # 350 LOC
│   │   │   ├── graph_viz.py            # 500 LOC
│   │   │   ├── heatmaps.py             # 300 LOC
│   │   │   ├── evolution_animation.py  # 400 LOC
│   │   │   └── reports.py              # 500 LOC
│   │   │
│   │   └── explainability/             # Model interpretation (1,500 LOC)
│   │       ├── __init__.py             # 50 LOC
│   │       ├── feature_importance.py   # 400 LOC
│   │       ├── sensitivity_analysis.py # 350 LOC
│   │       └── architecture_impact.py  # 350 LOC
│   │
│   ├── benchmarks/                     # Benchmarking suite (7,000 LOC)
│   │   ├── __init__.py                 # 100 LOC
│   │   │
│   │   ├── datasets/                   # Dataset loaders (2,000 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── openml.py               # 600 LOC
│   │   │   ├── cifar.py                # 300 LOC
│   │   │   ├── imagenet.py             # 400 LOC
│   │   │   ├── nas_bench.py            # 350 LOC
│   │   │   └── custom.py               # 300 LOC
│   │   │
│   │   ├── baselines/                  # Baseline comparisons (2,500 LOC)
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── auto_sklearn.py         # 600 LOC
│   │   │   ├── tpot.py                 # 500 LOC
│   │   │   ├── h2o_automl.py           # 450 LOC
│   │   │   ├── autogluon.py            # 400 LOC
│   │   │   └── comparison_report.py    # 500 LOC
│   │   │
│   │   └── metrics/                    # Evaluation metrics (2,500 LOC)
│   │       ├── __init__.py             # 50 LOC
│   │       ├── performance.py          # 600 LOC
│   │       ├── efficiency.py           # 500 LOC
│   │       ├── robustness.py           # 450 LOC
│   │       ├── statistical_tests.py    # 400 LOC
│   │       └── reporting.py            # 500 LOC
│   │
│   ├── cli/                            # Command-line interface (5,000 LOC)
│   │   ├── __init__.py                 # 50 LOC
│   │   ├── main.py                     # 800 LOC - Main CLI entry
│   │   ├── commands/
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── run.py                  # 600 LOC
│   │   │   ├── resume.py               # 400 LOC
│   │   │   ├── status.py               # 350 LOC
│   │   │   ├── results.py              # 400 LOC
│   │   │   ├── export.py               # 300 LOC
│   │   │   └── config.py               # 250 LOC
│   │   ├── formatting.py               # 400 LOC
│   │   ├── validation.py               # 300 LOC
│   │   └── interactive.py              # 600 LOC
│   │
│   ├── api/                            # REST API (4,000 LOC)
│   │   ├── __init__.py                 # 50 LOC
│   │   ├── app.py                      # 500 LOC - FastAPI app
│   │   ├── routers/
│   │   │   ├── __init__.py             # 50 LOC
│   │   │   ├── experiments.py          # 600 LOC
│   │   │   ├── models.py               # 500 LOC
│   │   │   ├── workers.py              # 400 LOC
│   │   │   └── admin.py                # 350 LOC
│   │   ├── models.py                   # 600 LOC - Pydantic models
│   │   ├── auth.py                     # 400 LOC
│   │   ├── middleware.py               # 300 LOC
│   │   └── websockets.py               # 250 LOC
│   │
│   └── utils/                          # Utilities (3,000 LOC)
│       ├── __init__.py                 # 50 LOC
│       ├── serialization.py            # 400 LOC
│       ├── hashing.py                  # 200 LOC
│       ├── random.py                   # 150 LOC
│       ├── validation.py               # 300 LOC
│       ├── monitoring.py               # 400 LOC
│       ├── profiling.py                # 350 LOC
│       ├── gpu_utils.py                # 300 LOC
│       ├── data_utils.py               # 400 LOC
│       └── math_utils.py               # 450 LOC
│
├── tests/                              # Test suite (12,000 LOC)
│   ├── __init__.py                     # 50 LOC
│   ├── conftest.py                     # 500 LOC - Pytest fixtures
│   │
│   ├── unit/                           # Unit tests (6,000 LOC)
│   │   ├── test_dsl/                   # 1,000 LOC
│   │   ├── test_search/                # 800 LOC
│   │   ├── test_graph/                 # 700 LOC
│   │   ├── test_optimizers/            # 1,500 LOC
│   │   ├── test_distributed/           # 1,000 LOC
│   │   └── test_meta_learning/         # 1,000 LOC
│   │
│   ├── integration/                    # Integration tests (4,000 LOC)
│   │   ├── test_end_to_end.py          # 800 LOC
│   │   ├── test_distributed_flow.py    # 600 LOC
│   │   ├── test_integrations.py        # 700 LOC
│   │   ├── test_meta_learning.py       # 500 LOC
│   │   └── test_benchmarks.py          # 400 LOC
│   │
│   └── performance/                    # Performance tests (2,000 LOC)
│       ├── test_scalability.py         # 500 LOC
│       ├── test_memory.py              # 400 LOC
│       ├── test_throughput.py          # 400 LOC
│       └── benchmark_suite.py          # 700 LOC
│
├── examples/                           # Example notebooks/scripts (3,000 LOC)
│   ├── quickstart.py                   # 150 LOC
│   ├── cifar10_classification.py       # 200 LOC
│   ├── custom_optimizer.py             # 250 LOC
│   ├── distributed_search.py           # 200 LOC
│   ├── meta_learning_demo.py           # 250 LOC
│   ├── multi_objective.py              # 200 LOC
│   ├── notebooks/
│   │   ├── 01_introduction.ipynb       # 300 LOC
│   │   ├── 02_dsl_tutorial.ipynb       # 350 LOC
│   │   ├── 03_custom_search.ipynb      # 300 LOC
│   │   ├── 04_distributed.ipynb        # 250 LOC
│   │   └── 05_visualization.ipynb      # 250 LOC
│   └── advanced/
│       ├── nas_from_scratch.py         # 300 LOC
│       └── hybrid_optimizer.py         # 300 LOC
│
├── docs/                               # Documentation (5,000 LOC)
│   ├── README.md                       # 100 LOC
│   ├── conf.py                         # 200 LOC - Sphinx config
│   │
│   ├── source/
│   │   ├── index.rst                   # 150 LOC
│   │   ├── installation.rst            # 200 LOC
│   │   ├── quickstart.rst              # 300 LOC
│   │   │
│   │   ├── user_guide/                 # User documentation
│   │   │   ├── dsl_reference.rst       # 400 LOC
│   │   │   ├── search_strategies.rst   # 350 LOC
│   │   │   ├── distributed_setup.rst   # 300 LOC
│   │   │   ├── meta_learning.rst       # 250 LOC
│   │   │   └── visualization.rst       # 200 LOC
│   │   │
│   │   ├── developer_guide/            # Developer docs
│   │   │   ├── architecture.rst        # 400 LOC
│   │   │   ├── contributing.rst        # 300 LOC
│   │   │   ├── custom_optimizers.rst   # 350 LOC
│   │   │   └── testing.rst             # 200 LOC
│   │   │
│   │   ├── api/                        # API documentation
│   │   │   ├── core.rst                # 300 LOC
│   │   │   ├── optimizers.rst          # 250 LOC
│   │   │   ├── distributed.rst         # 200 LOC
│   │   │   └── integrations.rst        # 150 LOC
│   │   │
│   │   └── tutorials/                  # Tutorials
│   │       ├── basic_workflow.rst      # 300 LOC
│   │       ├── custom_search_space.rst # 250 LOC
│   │       └── production_deployment.rst # 300 LOC
│   │
│   └── papers/                         # Research papers
│       ├── morphml_design.pdf          # Reference
│       └── benchmarks.pdf              # Reference
│
├── scripts/                            # Utility scripts (2,000 LOC)
│   ├── setup_dev_env.sh                # 100 LOC
│   ├── run_benchmarks.py               # 300 LOC
│   ├── generate_docs.sh                # 50 LOC
│   ├── deploy/
│   │   ├── docker_build.sh             # 100 LOC
│   │   ├── kubernetes_deploy.yaml      # 300 LOC
│   │   └── helm_chart/
│   │       ├── Chart.yaml              # 50 LOC
│   │       ├── values.yaml             # 200 LOC
│   │       └── templates/              # 400 LOC
│   └── data/
│       ├── download_datasets.py        # 200 LOC
│       └── preprocess.py               # 300 LOC
│
├── docker/                             # Docker configurations
│   ├── Dockerfile                      # 100 LOC
│   ├── Dockerfile.gpu                  # 120 LOC
│   ├── Dockerfile.worker               # 80 LOC
│   ├── docker-compose.yml              # 150 LOC
│   └── requirements/
│       ├── base.txt                    # 50 LOC
│       ├── dev.txt                     # 30 LOC
│       └── gpu.txt                     # 40 LOC
│
├── .github/                            # GitHub workflows
│   ├── workflows/
│   │   ├── ci.yml                      # 150 LOC
│   │   ├── tests.yml                   # 100 LOC
│   │   ├── docs.yml                    # 80 LOC
│   │   └── release.yml                 # 120 LOC
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md               # 80 LOC
│   │   └── feature_request.md          # 60 LOC
│   └── PULL_REQUEST_TEMPLATE.md        # 100 LOC
│
└── benchmarks_results/                 # Benchmark results (metadata)
    ├── openml_cc18/
    ├── nas_bench_201/
    └── comparison_with_baselines/

```

---

## 📊 Lines of Code Summary

### By Component

| Component | LOC | Percentage |
|-----------|-----|------------|
| **Core Systems** | 8,500 | 6.8% |
| - DSL | 3,500 | 2.8% |
| - Search Engine | 2,500 | 2.0% |
| - Graph System | 2,000 | 1.6% |
| - Objectives | 500 | 0.4% |
| **Optimizers** | 25,000 | 20.0% |
| - Evolutionary | 8,000 | 6.4% |
| - Bayesian | 7,000 | 5.6% |
| - Gradient-based | 5,000 | 4.0% |
| - Reinforcement | 4,000 | 3.2% |
| - Hybrid | 1,000 | 0.8% |
| **Distributed** | 20,000 | 16.0% |
| - Orchestrator | 5,000 | 4.0% |
| - Worker | 4,500 | 3.6% |
| - Scheduler | 4,000 | 3.2% |
| - Storage | 3,500 | 2.8% |
| - Communication | 3,000 | 2.4% |
| **Meta-Learning** | 15,000 | 12.0% |
| - Warmstart | 4,500 | 3.6% |
| - Predictors | 5,000 | 4.0% |
| - Strategy Evolution | 3,500 | 2.8% |
| - Knowledge Base | 2,000 | 1.6% |
| **Integrations** | 10,000 | 8.0% |
| **Visualization** | 8,000 | 6.4% |
| **Benchmarks** | 7,000 | 5.6% |
| **CLI** | 5,000 | 4.0% |
| **API** | 4,000 | 3.2% |
| **Utils** | 3,000 | 2.4% |
| **Tests** | 12,000 | 9.6% |
| **Examples** | 3,000 | 2.4% |
| **Docs** | 5,000 | 4.0% |
| **Scripts** | 2,000 | 1.6% |
| **Config Files** | 1,500 | 1.2% |
| **Total** | **125,000** | **100%** |

---

## 🔄 Data Flow Architecture

### 1. Experiment Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    1. DSL Definition                         │
│   User writes experiment specification in MorphML DSL       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    2. Compilation                            │
│   DSL → AST → Internal Representation → Validated Config    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    3. Experiment Setup                       │
│   Create search space, initialize optimizer, setup workers  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    4. Search Loop                            │
│   ┌──────────────────────────────────────────────────┐     │
│   │  a. Generate Candidates (Optimizer)              │     │
│   │  b. Distribute Tasks (Orchestrator)              │     │
│   │  c. Execute Evaluations (Workers)                │     │
│   │  d. Collect Results (Result Store)               │     │
│   │  e. Update Population (Optimizer)                │     │
│   │  f. Check Termination (Budget/Convergence)       │     │
│   └──────────────────────────────────────────────────┘     │
│                  Loop until budget exhausted                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    5. Result Analysis                        │
│   Generate reports, visualizations, export best models      │
└─────────────────────────────────────────────────────────────┘
```

### 2. Model Evaluation Flow

```
Worker receives task
       ↓
┌──────────────────┐
│ Parse model graph│
└──────────────────┘
       ↓
┌──────────────────┐
│ Compile to target│  (PyTorch/TF/JAX)
└──────────────────┘
       ↓
┌──────────────────┐
│ Load dataset     │
└──────────────────┘
       ↓
┌──────────────────┐
│ Train model      │
└──────────────────┘
       ↓
┌──────────────────┐
│ Evaluate metrics │
└──────────────────┘
       ↓
┌──────────────────┐
│ Return results   │
└──────────────────┘
```

### 3. Meta-Learning Flow

```
┌────────────────────────────────────────────────────────┐
│              Historical Experiments                    │
│  (Model architectures, hyperparameters, performance)   │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│              Feature Extraction                        │
│  (Dataset meta-features, task characteristics)         │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│              Performance Prediction                    │
│  (Train surrogate model on historical data)            │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│              Warm-Start Generation                     │
│  (Generate initial population for new experiment)      │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│              New Experiment Execution                  │
│  (Use warm-started population for faster convergence)  │
└────────────────────────────────────────────────────────┘
```

---

## 🔌 API Specifications

### Core DSL API

```python
# morphml/core/dsl/api.py (400 LOC)

from morphml import SearchSpace, Layer, Optimizer, Evolution

# Define search space
space = SearchSpace(
    name="cnn_search",
    input_shape=(32, 32, 3),
    output_shape=(10,)
)

# Add layers with options
space.add(Layer.conv2d(
    filters=[32, 64, 128],
    kernel_size=[3, 5, 7],
    activation=['relu', 'elu']
))

space.add(Layer.batch_norm())

space.add(Layer.pool(
    type=['max', 'avg'],
    pool_size=[2, 3]
))

space.add(Layer.dense(
    units=[128, 256, 512],
    activation=['relu', 'tanh']
))

# Configure optimizer
optimizer_config = Optimizer(
    type=['adam', 'sgd', 'rmsprop'],
    learning_rate=[1e-4, 1e-3, 1e-2],
    weight_decay=[0.0, 1e-5, 1e-4]
)

# Setup evolution
evolution = Evolution(
    strategy='genetic',
    population_size=50,
    generations=100,
    mutation_rate=0.15,
    crossover_rate=0.7,
    selection='tournament',
    elitism=0.1
)

# Create experiment
from morphml import Experiment

experiment = Experiment(
    name="cifar10_automl",
    search_space=space,
    optimizer_config=optimizer_config,
    evolution=evolution,
    objectives=['maximize:accuracy', 'minimize:params'],
    budget={'time': '24h', 'evals': 1000}
)

# Run experiment
results = experiment.run(
    dataset='cifar10',
    distributed=True,
    num_workers=10
)
```

### Search Engine API

```python
# morphml/core/search/api.py (300 LOC)

from morphml.optimizers import GeneticOptimizer

# Custom optimizer
class CustomOptimizer(GeneticOptimizer):
    def generate_offspring(self, population):
        # Custom generation logic
        offspring = []
        for parent in population:
            child = self.mutate(parent)
            offspring.append(child)
        return offspring
    
    def mutate(self, individual):
        # Custom mutation
        return mutated_individual

# Use custom optimizer
optimizer = CustomOptimizer(
    population_size=100,
    mutation_rate=0.2
)

experiment = Experiment(
    search_space=space,
    optimizer=optimizer
)
```

### Distributed API

```python
# morphml/distributed/api.py (400 LOC)

from morphml.distributed import DistributedOrchestrator, WorkerConfig

# Configure distributed execution
orchestrator = DistributedOrchestrator(
    master_address='localhost:8000',
    num_workers=10,
    worker_config=WorkerConfig(
        gpus_per_worker=1,
        cpus_per_worker=4,
        memory_per_worker='16GB'
    )
)

# Start workers
orchestrator.start_workers()

# Run distributed experiment
results = experiment.run(
    orchestrator=orchestrator,
    checkpoint_interval=100,
    fault_tolerance=True
)

# Monitor progress
for update in orchestrator.stream_progress():
    print(f"Progress: {update['completed']}/{update['total']}")
```

### Meta-Learning API

```python
# morphml/meta_learning/api.py (300 LOC)

from morphml.meta_learning import WarmStarter, PerformancePredictor

# Warm-start from previous experiments
warmstarter = WarmStarter(
    experience_db='experiments.db',
    similarity_metric='cosine',
    num_samples=10
)

initial_population = warmstarter.generate(
    task_features={'dataset': 'cifar10', 'n_classes': 10}
)

# Use performance predictor
predictor = PerformancePredictor(
    model='neural',
    features=['architecture', 'hyperparameters', 'dataset']
)

predictor.train(historical_experiments)

# Predict before training
predicted_acc = predictor.predict(candidate_model)
```

---

## 🔗 Integration Points

### 1. PyTorch Integration

```python
# morphml/integrations/pytorch/adapter.py

from morphml.core.graph import ModelGraph
import torch.nn as nn

class PyTorchAdapter:
    def graph_to_module(self, graph: ModelGraph) -> nn.Module:
        """Convert MorphML graph to PyTorch module"""
        pass
    
    def train(self, module: nn.Module, dataset, config):
        """Train PyTorch model"""
        pass
```

### 2. Scikit-learn Integration

```python
# morphml/integrations/sklearn/adapter.py

from morphml.core.graph import ModelGraph
from sklearn.pipeline import Pipeline

class SklearnAdapter:
    def graph_to_pipeline(self, graph: ModelGraph) -> Pipeline:
        """Convert MorphML graph to sklearn pipeline"""
        pass
    
    def fit(self, pipeline: Pipeline, X, y):
        """Fit sklearn pipeline"""
        pass
```

### 3. Ray Tune Integration

```python
# morphml/integrations/ray/tune_adapter.py

from ray import tune
from morphml import Experiment

def morphml_to_tune(experiment: Experiment):
    """Convert MorphML experiment to Ray Tune config"""
    
    config = {
        "search_space": experiment.search_space.to_dict(),
        "resources": {"cpu": 1, "gpu": 0.25}
    }
    
    return tune.run(
        trainable,
        config=config,
        num_samples=experiment.budget['evals']
    )
```

---

## 🚀 Deployment Architecture

### Single-Node Deployment

```
┌─────────────────────────────────────────┐
│         MorphML Master Process          │
│  ┌───────────────────────────────────┐  │
│  │  Orchestrator + Search Engine     │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │  Local Worker Pool (4-8 workers) │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │  SQLite Result Store              │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Multi-Node Cluster Deployment

```
                    ┌──────────────────┐
                    │  Master Node     │
                    │  - Orchestrator  │
                    │  - Web Dashboard │
                    │  - Result DB     │
                    └──────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
    ┌───────▼─────┐ ┌───────▼─────┐ ┌───────▼─────┐
    │ Worker Node │ │ Worker Node │ │ Worker Node │
    │ - 8 Workers │ │ - 8 Workers │ │ - 8 Workers │
    │ - 4 GPUs    │ │ - 4 GPUs    │ │ - 4 GPUs    │
    └─────────────┘ └─────────────┘ └─────────────┘
            │               │               │
            └───────────────┼───────────────┘
                            │
                    ┌───────▼──────┐
                    │ Shared Storage│
                    │ - Checkpoints │
                    │ - Datasets    │
                    └───────────────┘
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml (300 LOC)

apiVersion: apps/v1
kind: Deployment
metadata:
  name: morphml-master
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: master
        image: morphml/master:latest
        ports:
        - containerPort: 8000
        
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: morphml-workers
spec:
  replicas: 10
  template:
    spec:
      containers:
      - name: worker
        image: morphml/worker:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

---

## 🔐 Security Architecture

### Authentication Flow

```
User → API Gateway → JWT Validation → MorphML API
                          ↓
                    User Database
```

### Data Isolation

```
User A experiments → Namespace A → Isolated storage
User B experiments → Namespace B → Isolated storage
```

### Secure Communication

```
Master ←──TLS──→ Workers
Master ←──TLS──→ Database
Workers ←──mTLS──→ Workers
```

---

## 📈 Scalability Considerations

### Horizontal Scaling

- **Workers**: Add more worker nodes linearly
- **Master**: Single master with failover support
- **Storage**: Distributed file system (HDFS/S3)

### Vertical Scaling

- **Memory**: Support for large model graphs (>10GB)
- **GPU**: Multi-GPU support per worker
- **CPU**: Parallel evaluation on CPU clusters

### Performance Targets

| Metric | Target |
|--------|--------|
| Experiments/day | 10,000+ |
| Concurrent workers | 1,000+ |
| Models evaluated/hour | 100,000+ |
| Result query latency | <100ms |
| Dashboard update rate | 10 Hz |

---

## 🧪 Testing Strategy by Component

### Unit Tests (6,000 LOC)

```python
# tests/unit/test_dsl/test_parser.py (200 LOC)
def test_parse_valid_dsl():
    dsl_code = """
    space = SearchSpace()
    space.add(Layer.conv2d(filters=[32, 64]))
    """
    ast = parse(dsl_code)
    assert isinstance(ast, AST)
    assert len(ast.layers) == 1

# tests/unit/test_optimizers/test_genetic.py (300 LOC)
def test_genetic_mutation():
    optimizer = GeneticOptimizer(mutation_rate=0.1)
    parent = Individual(genes=[1, 2, 3, 4, 5])
    child = optimizer.mutate(parent)
    assert child != parent
    assert len(child.genes) == len(parent.genes)
```

### Integration Tests (4,000 LOC)

```python
# tests/integration/test_end_to_end.py (800 LOC)
def test_complete_experiment_flow():
    experiment = create_test_experiment()
    results = experiment.run(dataset='test_data', budget={'evals': 10})
    assert len(results.models) == 10
    assert results.best_model is not None
    assert results.best_model.accuracy > 0.5
```

### Performance Tests (2,000 LOC)

```python
# tests/performance/test_scalability.py (500 LOC)
@pytest.mark.parametrize("num_workers", [1, 10, 100])
def test_worker_scalability(num_workers):
    start_time = time.time()
    experiment.run(num_workers=num_workers, budget={'evals': 1000})
    elapsed = time.time() - start_time
    
    # Should scale sub-linearly
    expected_time = baseline_time / (num_workers ** 0.8)
    assert elapsed < expected_time * 1.2
```

---

## 📊 Monitoring & Observability

### Metrics Collection

```python
# morphml/utils/monitoring.py (400 LOC)

from prometheus_client import Counter, Histogram, Gauge

# Define metrics
experiments_started = Counter('morphml_experiments_started_total')
experiments_completed = Counter('morphml_experiments_completed_total')
evaluation_duration = Histogram('morphml_evaluation_duration_seconds')
active_workers = Gauge('morphml_active_workers')

# Instrument code
@evaluation_duration.time()
def evaluate_model(model):
    # Evaluation logic
    pass
```

### Logging Structure

```python
# All logs follow structured format
{
    "timestamp": "2025-10-31T10:30:00Z",
    "level": "INFO",
    "component": "orchestrator",
    "experiment_id": "exp_12345",
    "message": "Started distributed search",
    "metadata": {
        "num_workers": 10,
        "search_strategy": "genetic"
    }
}
```

---

## 🔄 Version Control & Release Strategy

### Semantic Versioning

- **Major**: Breaking API changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes

### Release Cycle

- **Nightly**: Automated builds from `main`
- **Beta**: Monthly releases from `develop`
- **Stable**: Quarterly releases with full testing

---

## 🎯 Critical Path Components

### Must-Build-First (Priority 1)

1. DSL Parser (3,500 LOC) - **Months 1-2**
2. Model Graph System (2,000 LOC) - **Month 2**
3. Basic Genetic Optimizer (1,200 LOC) - **Month 3**
4. Local Executor (800 LOC) - **Month 3**

### Essential Features (Priority 2)

1. Bayesian Optimizer (7,000 LOC) - **Months 4-5**
2. Distributed Orchestrator (5,000 LOC) - **Months 6-7**
3. PyTorch Integration (2,500 LOC) - **Month 8**

### Advanced Features (Priority 3)

1. Meta-Learning (15,000 LOC) - **Months 9-12**
2. Web Dashboard (4,000 LOC) - **Months 13-14**
3. All Integrations (10,000 LOC) - **Months 15-18**

---

## 📦 Dependencies

### Core Dependencies

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.10"
numpy = "^1.24.0"
scipy = "^1.10.0"
networkx = "^3.1"
pydantic = "^2.0.0"
torch = "^2.0.0"
ray = "^2.5.0"
fastapi = "^0.100.0"
sqlalchemy = "^2.0.0"
redis = "^4.5.0"
plotly = "^5.14.0"
scikit-learn = "^1.3.0"
```

### Optional Dependencies

```toml
[tool.poetry.extras]
tensorflow = ["tensorflow"]
jax = ["jax", "flax"]
gpu = ["torch-cuda"]
docs = ["sphinx", "sphinx-rtd-theme"]
dev = ["pytest", "black", "mypy", "pre-commit"]
```

---

## 🎓 Architecture Decision Records (ADRs)

### ADR-001: Why Python as Primary Language?

**Decision**: Use Python 3.10+ as the primary language

**Reasoning**:
- ML ecosystem is Python-centric
- Rich library ecosystem
- Easy prototyping and iteration
- Strong typing support via type hints

**Consequences**:
- Performance bottlenecks will need C++/Rust extensions
- GIL limitations for CPU-bound tasks

### ADR-002: Why Ray for Distributed Execution?

**Decision**: Use Ray as the distributed execution backend

**Reasoning**:
- Mature, battle-tested framework
- Native Python support
- Built-in fault tolerance
- Active community

**Consequences**:
- Additional dependency
- Learning curve for contributors

### ADR-003: Graph-Based Model Representation

**Decision**: Use DAG (Directed Acyclic Graph) for model representation

**Reasoning**:
- Flexible, supports arbitrary architectures
- Easy to mutate and evolve
- Framework-agnostic intermediate representation

**Consequences**:
- More complex than linear layer stacking
- Requires graph compilation for each framework

---

## 🏁 Success Criteria

### Technical Metrics

- ✅ DSL can express 95%+ of common architectures
- ✅ Search finds models within 5% of SOTA on benchmarks
- ✅ Distributed scaling efficiency >80% up to 100 workers
- ✅ Meta-learning reduces search time by 30%+

### Code Quality Metrics

- ✅ Test coverage >75%
- ✅ Type hint coverage >90%
- ✅ Documentation coverage 100% of public APIs
- ✅ Zero critical security vulnerabilities

---

## 📚 References

### Papers
- Elsken et al. "Neural Architecture Search: A Survey" (2019)
- Liu et al. "DARTS: Differentiable Architecture Search" (2019)
- Hutter et al. "Automated Machine Learning" (2019)

### Similar Projects
- Auto-Sklearn: github.com/automl/auto-sklearn
- TPOT: github.com/EpistasisLab/tpot
- Ray Tune: docs.ray.io/en/latest/tune
- NNI: github.com/microsoft/nni

---

## 📞 Architecture Review Process

### Monthly Architecture Review

1. Review new components against design principles
2. Identify technical debt
3. Plan refactoring sprints
4. Update this document

### Change Approval

- **Minor changes** (<500 LOC): Tech lead approval
- **Major changes** (>500 LOC): Team review + RFC
- **Breaking changes**: Community RFC + voting

---

**Document Maintained By**: Tech Lead  
**Last Architecture Review**: Month 0 (Initial Design)  
**Next Review**: End of Month 3

---

## 📊 Total Lines of Code: ~125,000

**Breakdown**:
- Production Code: 108,000 LOC (86.4%)
- Test Code: 12,000 LOC (9.6%)
- Documentation: 5,000 LOC (4.0%)

**Estimated Development Time**: 18-24 months with 10-person team

---

*This architecture is a living document and will evolve as the project progresses.*