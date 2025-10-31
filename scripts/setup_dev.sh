#!/bin/bash
# Development environment setup script for MorphML

set -e

echo "🧬 Setting up MorphML development environment..."

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Installing..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "✅ Poetry found"

# Install dependencies
echo "📦 Installing dependencies..."
poetry install

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
poetry run pre-commit install

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "Quick start commands:"
echo "  poetry run pytest                    # Run all tests"
echo "  poetry run pytest tests/test_graph.py -v  # Run graph tests"
echo "  poetry run black morphml tests       # Format code"
echo "  poetry run mypy morphml              # Type checking"
echo ""
echo "Happy coding! 🚀"
