# Changelog

All notable changes to MorphML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-11-11

### 🎉 First Stable Release

MorphML v1.0.0 marks the first production-ready release with complete ecosystem features.

### Added

#### Complete Ecosystem (Phase 5)
- Web dashboard with real-time monitoring (React + FastAPI)
- REST API with JWT authentication and rate limiting
- Framework integrations (PyTorch, TensorFlow, JAX, Scikit-learn)
- Interactive Plotly dashboards for visualization
- Professional architecture diagram generator with Graphviz
- Performance analytics and statistical analysis
- Extensible plugin system
- Enhanced CLI with interactive commands
- Python API client library
- Complete documentation site with MkDocs Material

#### Production Features
- Automatic shape inference for all frameworks
- GPU acceleration support
- Multiple installation options with extras
- Comprehensive error handling
- Type hints throughout codebase
- PyPI publishing configuration

### Changed
- Updated authors information (Vedanth & Eshan Roy)
- Improved documentation structure
- Enhanced code organization

### Infrastructure
- Added publish.sh for automated PyPI publishing
- Added MANIFEST.in for package data
- Added setup.py for backwards compatibility
- Complete PyPI metadata in pyproject.toml

---

## [0.1.0] - 2024-11-11

### Initial Beta Release

### Added

#### Core Features (Phase 1)
- Neural Architecture Search framework with pythonic DSL
- `Layer.flatten()` method for CNN-to-Dense transitions
- Genetic Algorithm optimizer with true crossover implementation
- Random Search and Hill Climbing optimizers
- Simulated Annealing optimizer
- Differential Evolution optimizer
- `ModelGraph` class for architecture representation
- `SearchSpace` class for defining search spaces
- `HeuristicEvaluator` for fast architecture assessment
- Constraint system (MaxParameters, MinParameters, Depth, Width)
- Architecture export to PyTorch and Keras code
- Checkpointing system for long-running searches

#### Enhanced Features (Phase 2)
- Automatic shape inference in code export
- Enhanced constraint violation messages with detailed reporting
- Comprehensive DSL layer documentation
- Integrated crossover operators in GeneticAlgorithm
- Support for all layer types in exporter (avgpool, batchnorm, flatten)
- Import aliases for backward compatibility

#### Advanced Features (Phase 3)
- Crossover visualization utilities
- Adaptive crossover and mutation rate managers
- Custom layer handler support in exporter
- Flatten layer examples and tutorials
- Advanced features showcase examples

#### Quality of Life (Phase 4)
- Progress tracking system with rich output
- Quick-start CLI helper with templates
- Architecture comparison utilities
- Search space validation system
- Updated README with all features

#### Ecosystem (Phase 5)
- REST API foundation with FastAPI
- Web dashboard with React frontend
- Real-time monitoring via WebSocket
- Experiment management interface
- Interactive convergence visualization
- Architecture graph viewer

### Documentation
- Complete README with examples
- API documentation
- DSL layer reference guide
- Contributing guidelines
- Phase summaries (P1-P5)
- Dashboard setup guide

### Infrastructure
- Poetry-based dependency management
- Pre-commit hooks
- Black code formatting
- Ruff linting
- MyPy type checking
- Pytest test suite
- PyPI packaging configuration

## [Unreleased]

### Planned
- Database persistence for experiments
- Authentication system for dashboard
- PyTorch training integration
- TensorFlow/Keras integration
- JAX/Flax integration
- Plugin system
- Comprehensive documentation site
- Docker images
- Kubernetes deployment configs

---

## Version History

- **0.1.0** (2024-11-11) - Initial release with core NAS functionality, web dashboard, and comprehensive tooling

[0.1.0]: https://github.com/TIVerse/MorphML/releases/tag/v0.1.0
