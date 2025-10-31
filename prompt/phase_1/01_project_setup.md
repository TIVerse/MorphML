# Component 1: Project Infrastructure Setup

**Duration:** Week 1  
**LOC Target:** ~500 setup + config

---

## 🎯 Objective

Set up professional Python project with Poetry, CI/CD, code quality tools, logging, and configuration.

---

## 📋 Tasks

### 1. Create `pyproject.toml` with Poetry

**Dependencies:**
- Core: numpy, scipy, networkx, pydantic, click, rich, pyyaml
- Dev: pytest, pytest-cov, black, mypy, ruff, pre-commit

**Tool configurations:**
- black: line-length=100
- mypy: strict mode with disallow_untyped_defs=true
- pytest: coverage threshold 75%
- CLI entry: `morphml = "morphml.cli.main:cli"`

### 2. Create `.gitignore`

Ignore: `__pycache__/`, `*.egg-info`, `venv/`, `.vscode/`, `.pytest_cache/`, `htmlcov/`, `logs/`, `checkpoints/`

### 3. Create `.github/workflows/ci.yml`

**Pipeline:**
- Lint job: black, ruff, mypy
- Test job: pytest on matrix (Python 3.10, 3.11) x (ubuntu, macos, windows)
- Coverage: upload to Codecov
- Security: bandit scan

### 4. Core Package Files

**`morphml/version.py`** - Version constants

**`morphml/exceptions.py`** - Exception hierarchy:
- `MorphMLError` (base)
- `DSLError(line, column)` - parsing errors
- `ValidationError` - invalid values
- `SearchSpaceError` - space definition errors
- `GraphError` - DAG violations
- `OptimizerError` - optimization failures
- `ExecutionError` - runtime errors

**`morphml/config.py`** (~200 LOC):
```python
class Config:
    """Configuration with dot notation access."""
    def get(key: str, default=None) -> Any
    def set(key: str, value: Any) -> None
    def _load_from_file(path: str) -> None
    def _apply_env_overrides() -> None  # MORPHML_* env vars

DEFAULT_CONFIG = {
    "logging": {"level": "INFO", "log_dir": "~/.morphml/logs"},
    "execution": {"max_workers": 4, "timeout": 3600},
    "search": {"population_size": 50, "max_generations": 100},
    "cache": {"enabled": True, "max_size_gb": 10}
}
```

**`morphml/logging_config.py`** (~100 LOC):
```python
def setup_logging(level=None, log_file=None, verbose=False) -> None
def get_logger(name: str) -> logging.Logger

# Use Rich for colored console output
# Support both console and file logging
```

### 5. Directory Structure

Create empty `__init__.py` in:
- `morphml/core/{dsl,search,graph,objectives}/`
- `morphml/optimizers/evolutionary/`
- `morphml/execution/`
- `morphml/cli/commands/`
- `morphml/utils/`
- `tests/{unit,integration}/`

### 6. Test Infrastructure

**`tests/conftest.py`** - Shared fixtures:
```python
@pytest.fixture
def sample_config() -> Config

@pytest.fixture
def temp_dir(tmp_path) -> Path

@pytest.fixture
def logger() -> logging.Logger
```

---

## ✅ Validation

After setup, verify:

```bash
# Install dependencies
poetry install

# Run linters (should pass on empty code)
poetry run black --check morphml
poetry run ruff morphml
poetry run mypy morphml

# Install pre-commit hooks
poetry run pre-commit install

# Test imports
poetry run python -c "import morphml; print(morphml.__version__)"
```

---

## 📦 Deliverables

- [ ] `pyproject.toml` with all dependencies
- [ ] `.gitignore` comprehensive patterns
- [ ] `.github/workflows/ci.yml` complete pipeline
- [ ] `morphml/{version,exceptions,config,logging_config}.py`
- [ ] All directory structure created
- [ ] `tests/conftest.py` with fixtures
- [ ] CI pipeline passing (green checkmark on GitHub)

---

**Next:** Proceed to `02_dsl_implementation.md`
