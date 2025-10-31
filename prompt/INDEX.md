# MorphML Development Prompts - Complete Index

**Author:** Eshan Roy <eshanized@proton.me>  
**Organization:** TONMOY INFRASTRUCTURE & VISION  
**Repository:** https://github.com/TIVerse/MorphML  
**Last Updated:** November 1, 2025

---

## 📚 How to Use This Prompt Library

This directory contains **comprehensive, LLM-ready prompts** for building MorphML across 5 development phases. Each prompt is engineered for use with Large Language Models (e.g., Claude Sonnet 4.5) to generate production-grade code.

### Prompt Engineering Principles

✅ **Human-Readable Code** - All code should look like it was written by an experienced developer  
✅ **Simple Functions** - No overly complex functions (<50 lines each)  
✅ **Production-Grade** - Fully tested, documented, and type-hinted  
✅ **Incremental** - Each component builds on previous ones  
✅ **Testable** - >75% test coverage target

---

## 🗺️ Complete Prompt Structure

### Phase 1: Foundation (Months 1-6) - ~20,000 LOC
**Status:** ✅ Prompts Complete  
**Target:** Core DSL, Graph System, Genetic Algorithm, Local Execution, CLI

```
phase_1/
├── 00_overview.md                    # Phase objectives and architecture
├── 01_project_setup.md               # Poetry, CI/CD, config (~500 LOC)
├── 02_dsl_implementation.md          # Lexer, Parser, Compiler (~3,500 LOC)
├── 03_graph_system.md                # DAG, Nodes, Edges, Mutations (~2,000 LOC)
├── 04_search_engine.md               # SearchSpace, Population (~2,500 LOC)
├── 05_genetic_algorithm.md           # GA, Selection, Crossover (~3,000 LOC)
└── 06_execution_cli.md               # Executor, Evaluator, CLI (~3,000 LOC)
```

**Deliverables:**
- ✅ Pythonic DSL for search spaces
- ✅ Model graph DAG with mutations
- ✅ Genetic algorithm optimizer
- ✅ Local execution engine
- ✅ CLI with Rich output
- ✅ >75% test coverage

---

### Phase 2: Advanced Search (Months 7-12) - ~25,000 LOC
**Status:** ✅ Prompts Complete  
**Target:** Bayesian Optimization, DARTS, Multi-Objective, Benchmarks

```
phase_2/
├── 00_overview.md                    # Phase objectives
├── 01_bayesian_optimization.md       # GP, TPE, SMAC (~5,000 LOC)
├── 02_gradient_based_nas.md          # DARTS, ENAS (~6,000 LOC)
├── 03_multi_objective.md             # NSGA-II, Pareto (~4,000 LOC)
├── 04_advanced_evolutionary.md       # DE, CMA-ES, PSO (~5,000 LOC)
└── 05_benchmarking_visualization.md  # OpenML, Plots (~5,000 LOC)
```

**Deliverables:**
- ✅ Multiple optimization backends
- ✅ Gradient-based NAS with GPU support
- ✅ Multi-objective optimization
- ✅ Comprehensive benchmarking suite
- ✅ Advanced visualizations

---

### Phase 3: Distribution (Months 13-18) - ~20,000 LOC
**Status:** ✅ Prompts Complete  
**Target:** Master-Worker, Task Scheduling, Fault Tolerance, Kubernetes

```
phase_3/
├── 00_overview.md                    # Phase objectives
├── 01_master_worker.md               # Orchestrator, Workers (~5,000 LOC)
├── 02_task_scheduling.md             # Queue, Load Balancing (~4,000 LOC)
├── 03_storage.md                     # Distributed DB, Cache (~4,000 LOC)
├── 04_fault_tolerance.md             # Recovery, Checkpoints (~3,000 LOC)
└── 05_kubernetes.md                  # Docker, K8s, Helm (~4,000 LOC)
```

**Deliverables:**
- ✅ Distributed execution across multiple nodes
- ✅ 80%+ scaling efficiency to 50 workers
- ✅ Fault tolerance with automatic recovery
- ✅ Kubernetes deployment ready
- ✅ Production monitoring

---

### Phase 4: Meta-Learning (Months 19-24) - ~15,000 LOC
**Status:** ✅ Prompts Complete  
**Target:** Warm-Starting, Performance Prediction, Knowledge Base, Transfer

```
phase_4/
├── 00_overview.md                    # Phase objectives
├── 01_warm_starting.md               # Initialize from history (~3,000 LOC)
├── 02_performance_prediction.md      # GNN predictors (~4,000 LOC)
├── 03_knowledge_base.md              # Experience storage (~3,000 LOC)
├── 04_strategy_evolution.md          # Learn strategies (~3,000 LOC)
└── 05_transfer_learning.md           # Cross-task transfer (~2,000 LOC)
```

**Deliverables:**
- ✅ 30%+ search time reduction via warm-starting
- ✅ 75%+ performance prediction accuracy
- ✅ Knowledge base with 10,000+ experiments
- ✅ Adaptive strategy selection
- ✅ Transfer learning across datasets

---

### Phase 5: Ecosystem & Polish (Months 25-30) - ~15,000 LOC
**Status:** ✅ Prompts Complete  
**Target:** Dashboard, Integrations, API, Documentation

```
phase_5/
├── 00_overview.md                    # Phase objectives
├── 01_web_dashboard.md               # React + FastAPI (~5,000 LOC)
├── 02_framework_integrations.md      # PyTorch, TF, JAX (~4,000 LOC)
├── 03_rest_api.md                    # REST + WebSocket (~3,000 LOC)
├── 04_visualization.md               # Plotly, D3.js (~2,000 LOC)
└── 05_plugins_docs.md                # Plugin system, Docs (~1,000 LOC)
```

**Deliverables:**
- ✅ Full web dashboard
- ✅ 4 framework integrations (PyTorch, TF, JAX, Sklearn)
- ✅ REST API with WebSocket
- ✅ Plugin system
- ✅ Comprehensive documentation site

---

## 🚀 Development Workflow

### For LLM-Assisted Development

**Step 1: Start with Phase 1**
```bash
# Read the overview
cat prompt/phase_1/00_overview.md

# Then proceed component by component
cat prompt/phase_1/01_project_setup.md
# Generate code following the specifications
# Test and validate before moving to next component
```

**Step 2: Sequential Prompting**
- Feed one component prompt at a time to your LLM
- Generate code for that component
- Run tests to validate
- Fix any issues before proceeding
- Move to next component

**Step 3: Integration Testing**
- After completing all components in a phase
- Run integration tests
- Validate phase success criteria
- Move to next phase

**Step 4: Iterate if Needed**
- If generated code doesn't meet standards
- Refine the prompt or regenerate
- Ensure >75% test coverage
- Ensure all code passes linting (black, ruff, mypy)

---

## 📊 Project Statistics

### Total Scope
- **Total LOC:** ~107,000 (production) + ~12,000 (tests)
- **Total Phases:** 5
- **Total Components:** 26
- **Timeline:** 30 months (2.5 years)
- **Team Size:** 2-4 developers (or 1 developer + LLMs)

### Phase Breakdown
| Phase | Duration | LOC | Components | Key Deliverables |
|-------|----------|-----|------------|------------------|
| Phase 1 | 6 months | 20,000 | 6 | DSL, Graph, GA, CLI |
| Phase 2 | 6 months | 25,000 | 5 | BO, DARTS, Multi-Obj |
| Phase 3 | 6 months | 20,000 | 5 | Distributed, K8s |
| Phase 4 | 6 months | 15,000 | 5 | Meta-Learning |
| Phase 5 | 6 months | 15,000 | 5 | Dashboard, Integrations |
| **Total** | **30 months** | **95,000** | **26** | **Complete System** |

---

## ✅ Quality Checklist (Per Component)

Before marking any component complete:

### Code Quality
- [ ] All functions have docstrings
- [ ] Type hints on all public APIs
- [ ] No function exceeds 50 lines
- [ ] Follows PEP 8 (enforced by black)
- [ ] No linting errors (ruff)
- [ ] No type errors (mypy)

### Testing
- [ ] Unit tests written
- [ ] Test coverage >75%
- [ ] Integration tests pass
- [ ] Edge cases covered
- [ ] Performance tests (if applicable)

### Documentation
- [ ] Module docstring explaining purpose
- [ ] Class docstrings with examples
- [ ] Function docstrings with Args/Returns/Raises
- [ ] Inline comments for complex logic

### Integration
- [ ] Works with previous components
- [ ] Doesn't break existing tests
- [ ] Follows established patterns
- [ ] Proper error handling

---

## 🎓 Best Practices for LLM Prompting

### 1. Context Management
- Start each session by providing relevant previous components
- Reference the architecture docs (`docs/architecture.md`)
- Point to existing code when extending functionality

### 2. Iterative Refinement
```
First pass: Generate basic structure
Second pass: Add error handling
Third pass: Add tests
Fourth pass: Add documentation
Fifth pass: Optimize and refine
```

### 3. Testing-Driven
- Always ask the LLM to generate tests alongside code
- Run tests immediately after generation
- Fix failing tests before moving forward

### 4. Code Review
- Review generated code for:
  - Clarity and readability
  - Edge case handling
  - Performance concerns
  - Security issues
- Refine prompts based on review findings

---

## 🔧 Tools and Environment

### Required Tools
```bash
# Python environment
python >= 3.10

# Package manager
poetry

# Code quality
black, ruff, mypy, pre-commit

# Testing
pytest, pytest-cov
```

### Development Setup
```bash
# Clone repository
git clone https://github.com/TIVerse/MorphML.git
cd MorphML

# Install dependencies
poetry install

# Setup pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest
```

---

## 📞 Support and Resources

### Documentation
- **Architecture:** `docs/architecture.md`
- **Flows:** `docs/flows.md`
- **Research:** `docs/research.md`
- **Info:** `docs/info.md`

### Contact
- **Author:** Eshan Roy
- **Email:** eshanized@proton.me
- **GitHub:** https://github.com/eshanized
- **Organization:** TONMOY INFRASTRUCTURE & VISION
- **Repository:** https://github.com/TIVerse/MorphML

---

## 🎉 Success Milestones

### Phase 1 Complete ✓
- [ ] Can define search spaces with DSL
- [ ] Can run genetic algorithm locally
- [ ] CLI functional
- [ ] >75% test coverage

### Phase 2 Complete ✓
- [ ] Multiple optimizers available
- [ ] Bayesian optimization working
- [ ] Multi-objective optimization working
- [ ] Benchmark suite complete

### Phase 3 Complete ✓
- [ ] Distributed execution working
- [ ] 50+ workers supported
- [ ] Fault tolerance validated
- [ ] Kubernetes deployment successful

### Phase 4 Complete ✓
- [ ] Warm-starting reduces search time
- [ ] Performance predictor accurate
- [ ] Knowledge base operational
- [ ] Transfer learning working

### Phase 5 Complete ✓
- [ ] Dashboard deployed
- [ ] All integrations working
- [ ] API functional
- [ ] Documentation complete

---

## 🚀 Ready to Start?

1. **Read:** `prompt/README.md` (if you haven't already)
2. **Start:** `prompt/phase_1/00_overview.md`
3. **Build:** Follow prompts sequentially
4. **Test:** Validate after each component
5. **Iterate:** Refine as needed

**Good luck building MorphML! 🧬**

---

*This prompt library is designed to work with Claude Sonnet 4.5, GPT-4, and other advanced LLMs. Adjust prompts as needed for your specific LLM.*
