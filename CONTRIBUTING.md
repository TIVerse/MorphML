# Contributing to MorphML

Thank you for your interest in contributing to MorphML! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your contribution
5. Make your changes
6. Submit a pull request

## Development Setup

MorphML uses Poetry for dependency management. To set up your development environment:

```bash
# Clone the repository
git clone https://github.com/TIVerse/MorphML.git
cd MorphML

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

## How to Contribute

### Types of Contributions

- **Bug fixes**: Fix issues reported in the issue tracker
- **Features**: Implement new features or improve existing ones
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Examples**: Add examples demonstrating MorphML usage

### Before Starting

1. Check the [issue tracker](https://github.com/TIVerse/MorphML/issues) for existing issues
2. If you're planning a major change, open an issue first to discuss it
3. Make sure your contribution aligns with the project's goals

## Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use type hints for function signatures
- Write descriptive docstrings (Google style)
- Keep functions focused and concise

### Code Quality Tools

We use several tools to maintain code quality:

- **Ruff**: For linting and formatting
- **MyPy**: For static type checking
- **Pytest**: For testing

Run these checks before submitting:

```bash
# Format code
poetry run ruff format .

# Lint code
poetry run ruff check .

# Type check
poetry run mypy morphml

# Run tests
poetry run pytest
```

### Commit Messages

- Use clear and descriptive commit messages
- Start with a verb in the imperative mood (e.g., "Add", "Fix", "Update")
- Keep the first line under 72 characters
- Reference relevant issues (e.g., "Fixes #123")

Example:
```
Add support for custom loss functions

- Implement LossFunction base class
- Add validation for loss function parameters
- Update documentation with examples

Fixes #123
```

## Testing

All contributions should include appropriate tests:

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test interactions between components
- **Regression tests**: Ensure bugs don't reappear

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=morphml --cov-report=html

# Run specific test file
poetry run pytest tests/test_phase1_syntax.py

# Run tests matching a pattern
poetry run pytest -k "test_optimizer"
```

## Documentation

Good documentation is crucial:

- Update docstrings when changing function signatures
- Add examples for new features
- Update README.md if adding significant functionality
- Keep documentation clear and concise

### Docstring Format

We use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Short description of the function.
    
    Longer description with more details about what the function
    does and how it works.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
        
    Examples:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
    pass
```

## Pull Request Process

1. **Update your branch**: Ensure your branch is up to date with main
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-branch
   git rebase main
   ```

2. **Run all tests and checks**: Make sure everything passes
   ```bash
   poetry run pytest
   poetry run ruff check .
   poetry run mypy morphml
   ```

3. **Update documentation**: If needed, update docs and examples

4. **Create pull request**:
   - Use a clear and descriptive title
   - Fill out the PR template completely
   - Link related issues
   - Add screenshots/examples if relevant

5. **Code review**:
   - Address review comments promptly
   - Be open to feedback and suggestions
   - Update your PR as needed

6. **Merge**: Once approved, a maintainer will merge your PR

## Reporting Bugs

When reporting bugs, please use the bug report template and include:

- **Description**: Clear description of the bug
- **Steps to reproduce**: Detailed steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: Python version, OS, MorphML version
- **Code sample**: Minimal code to reproduce the issue
- **Error messages**: Full error messages and stack traces

## Suggesting Enhancements

When suggesting enhancements, please:

- Use the feature request template
- Explain the motivation and use case
- Provide examples of how it would be used
- Consider backwards compatibility
- Discuss alternative approaches

## Questions?

If you have questions:

- Check the [documentation](docs/)
- Search existing [issues](https://github.com/TIVerse/MorphML/issues)
- Ask in discussions or open a new issue

## Recognition

Contributors will be:
- Listed in the project's contributors page
- Acknowledged in release notes for significant contributions
- Part of a growing community building the future of ML frameworks

Thank you for contributing to MorphML! 🎉
