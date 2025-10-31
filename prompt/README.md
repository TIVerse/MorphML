# MorphML Development Prompts

**Author:** Eshan Roy <eshanized@proton.me>  
**Organization:** TONMOY INFRASTRUCTURE & VISION  
**Repository:** https://github.com/TIVerse/MorphML  
**Last Updated:** November 1, 2025

---

## Overview

This directory contains comprehensive, production-ready prompts for building MorphML using Large Language Models (LLMs) such as Claude Sonnet 4.5. Each prompt is carefully engineered to guide the development of production-grade, human-readable code that can be released directly to GitHub.

## Prompt Engineering Principles

All prompts follow these core principles:

1. **Human-Readable Code**: All generated code should look like it was written by an experienced developer, with clear variable names, proper documentation, and thoughtful comments.

2. **Simple Functions**: No overly complex functions. Each function should have a single, clear responsibility with manageable complexity.

3. **Production-Grade Quality**: Code must be:
   - Well-tested with comprehensive unit and integration tests
   - Properly documented with docstrings and inline comments
   - Type-hinted for clarity and IDE support
   - Following Python best practices (PEP 8, PEP 257)
   - Secure and handling edge cases

4. **Incremental & Testable**: Each phase builds on the previous one, with clear integration points and comprehensive testing.

5. **Documentation-First**: Every component includes clear documentation explaining the "why" behind design decisions.

## Phase Structure

MorphML is built across 5 major phases, aligned with the 18-24 month development roadmap:

### Phase 1: Foundation (Months 1-6)
**File:** `phase_1_prompt.md`

**Scope:**
- Project structure and configuration
- DSL lexer, parser, and AST
- Basic search space representation
- Simple model graph system
- Genetic algorithm optimizer
- Local execution engine
- CLI foundation
- Unit tests and CI/CD setup

**Deliverables:**
- ~20,000 LOC of production code
- Complete DSL implementation
- Working genetic algorithm
- Local execution capability
- Basic CLI interface

**Key Components:**
- `morphml/core/dsl/` - DSL implementation
- `morphml/core/search/` - Search space abstractions
- `morphml/core/graph/` - Model graph system
- `morphml/optimizers/evolutionary/genetic.py` - GA implementation
- `morphml/cli/` - Command-line interface
- `tests/` - Comprehensive test suite

---

### Phase 2: Advanced Search (Months 7-12)
**File:** `phase_2_prompt.md`

**Scope:**
- Bayesian optimization (GP, TPE, SMAC)
- Gradient-based NAS (DARTS, ENAS)
- Multi-objective optimization (NSGA-II)
- Additional evolutionary algorithms (DE, CMA-ES, PSO)
- Enhanced search engine
- Performance metrics and benchmarking

**Deliverables:**
- ~25,000 LOC additional
- Multiple optimization backends
- Multi-objective support
- Comprehensive benchmarks

**Key Components:**
- `morphml/optimizers/bayesian/` - BO implementations
- `morphml/optimizers/gradient_based/` - DARTS/ENAS
- `morphml/core/objectives/` - Multi-objective framework
- `morphml/benchmarks/` - Benchmark suite

---

### Phase 3: Distribution (Months 13-18)
**File:** `phase_3_prompt.md`

**Scope:**
- Master-worker orchestration
- Task scheduling and load balancing
- Distributed storage and caching
- Fault tolerance and checkpointing
- Kubernetes deployment
- Result aggregation
- Resource management

**Deliverables:**
- ~20,000 LOC additional
- Distributed execution capability
- Kubernetes support
- Fault tolerance mechanisms

**Key Components:**
- `morphml/distributed/orchestrator/` - Master coordination
- `morphml/distributed/worker/` - Worker implementation
- `morphml/distributed/scheduler/` - Task scheduling
- `morphml/distributed/storage/` - Distributed storage
- `docker/` and `kubernetes/` - Deployment configs

---

### Phase 4: Meta-Learning (Months 19-24)
**File:** `phase_4_prompt.md`

**Scope:**
- Warm-starting from past experiments
- Performance prediction models
- Transfer learning across tasks
- Knowledge base and retrieval
- Strategy evolution
- Meta-feature extraction

**Deliverables:**
- ~15,000 LOC additional
- Meta-learning engine
- 30%+ search time reduction
- Performance predictors

**Key Components:**
- `morphml/meta_learning/warmstart/` - Warm-starting
- `morphml/meta_learning/predictors/` - Performance prediction
- `morphml/meta_learning/strategy_evolution/` - Strategy optimization
- `morphml/meta_learning/knowledge_base/` - Experience storage

---

### Phase 5: Ecosystem & Polish (Months 25-30)
**File:** `phase_5_prompt.md`

**Scope:**
- Web dashboard (React + FastAPI)
- Framework integrations (PyTorch, TensorFlow, JAX, Scikit-learn)
- Visualization tools
- REST API
- Plugin system
- Documentation site
- Example notebooks

**Deliverables:**
- ~15,000 LOC additional
- Full web dashboard
- Complete integrations
- Production documentation

**Key Components:**
- `morphml/visualization/dashboard/` - Web UI
- `morphml/integrations/` - Framework adapters
- `morphml/api/` - REST API
- `docs/` - Sphinx documentation
- `examples/` - Notebooks and scripts

---

## How to Use These Prompts

### For Claude Sonnet 4.5 (or similar LLMs):

1. **Read the Prompt**: Each prompt is comprehensive (>600 lines) and self-contained.

2. **Follow the Structure**: Prompts are organized into:
   - Context and background
   - Objectives and scope
   - Component specifications
   - Implementation guidelines
   - Testing requirements
   - Success criteria

3. **Generate Code Iteratively**: Don't try to generate everything at once. Follow the sections in order.

4. **Review and Refine**: After generation, review for:
   - Code quality and readability
   - Test coverage
   - Documentation completeness
   - Edge case handling

5. **Integration**: Ensure each phase integrates cleanly with previous phases.

### Sequential Development:

```bash
# Phase 1: Foundation
1. Read prompt/phase_1_prompt.md
2. Generate code following the specifications
3. Run tests and ensure 80%+ coverage
4. Review code quality
5. Commit to GitHub

# Phase 2: Advanced Search
1. Read prompt/phase_2_prompt.md
2. Ensure Phase 1 is complete
3. Generate code for Phase 2
4. Run integration tests with Phase 1
5. Commit to GitHub

# ... Continue for Phases 3, 4, 5
```

---

## Code Quality Checklist

Before considering any phase complete, ensure:

- [ ] All functions have docstrings with parameter and return type documentation
- [ ] Type hints on all function signatures
- [ ] Inline comments explain "why", not "what"
- [ ] No function exceeds 50 lines (split complex logic)
- [ ] All edge cases handled with appropriate error messages
- [ ] Unit test coverage >75%
- [ ] Integration tests for all major workflows
- [ ] No hardcoded values (use configuration)
- [ ] Logging at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- [ ] Security considerations addressed (input validation, sanitization)
- [ ] Performance profiling for critical paths
- [ ] Documentation updated in `docs/`

---

## Prompt Maintenance

These prompts are living documents. As the project evolves:

1. **Update Prompts**: When architecture changes, update relevant prompts
2. **Version Control**: Keep prompts in sync with code versions
3. **Lessons Learned**: Add notes about what worked well or needed adjustment
4. **Community Feedback**: Incorporate feedback from developers using these prompts

---

## Additional Resources

- **Architecture**: See `docs/architecture.md` for detailed system design
- **Flows**: See `docs/flows.md` for algorithmic diagrams
- **Research**: See `docs/research.md` for theoretical background
- **Project Info**: See `docs/info.md` for project overview

---

## Estimated Effort

| Phase | Lines of Code | Core Components | Estimated Weeks |
|-------|--------------|-----------------|-----------------|
| Phase 1 | ~20,000 | DSL, Graph, GA, CLI | 8-12 weeks |
| Phase 2 | ~25,000 | BO, DARTS, Multi-obj | 8-12 weeks |
| Phase 3 | ~20,000 | Distributed, K8s | 8-12 weeks |
| Phase 4 | ~15,000 | Meta-learning | 8-10 weeks |
| Phase 5 | ~15,000 | Dashboard, Integrations | 8-10 weeks |
| Tests | ~12,000 | Unit + Integration | Throughout |
| **Total** | **~107,000** | + ~12,000 tests | **40-56 weeks** |

---

## Success Metrics

### Phase 1 Success:
- DSL can parse and compile valid input
- Genetic algorithm finds improving solutions
- Local execution completes end-to-end
- CLI functional for basic workflows

### Phase 2 Success:
- Multiple optimizers available and swappable
- Bayesian optimization converges faster than random
- DARTS produces valid architectures
- Multi-objective discovers Pareto fronts

### Phase 3 Success:
- 80%+ scaling efficiency up to 50 workers
- Automatic fault recovery works
- Kubernetes deployment successful
- No data loss on worker failure

### Phase 4 Success:
- Warm-starting reduces search time 30%+
- Performance predictor accuracy >75%
- Transfer learning across similar tasks works
- Knowledge base stores and retrieves effectively

### Phase 5 Success:
- Dashboard displays real-time experiment data
- All 4 framework integrations working
- REST API handles concurrent requests
- Documentation complete and published

---

## Contributing

These prompts are designed for LLM-assisted development, but human developers can also use them as detailed specifications. If you improve a prompt or find issues:

1. Create an issue describing the problem/improvement
2. Submit a PR with updated prompt
3. Include rationale for changes
4. Update this README if structure changes

---

## License

These prompts are part of the MorphML project and follow the same Apache 2.0 license.

---

**Questions or Issues?**  
Contact: eshanized@proton.me  
Repository: https://github.com/TIVerse/MorphML
