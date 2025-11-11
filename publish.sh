#!/bin/bash
# MorphML PyPI Publication Script
# Usage: ./publish.sh

set -e  # Exit on error

echo "🚀 MorphML PyPI Publication Script"
echo "=================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}❌ Poetry not found. Please install poetry first.${NC}"
    echo "   Install with: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

echo -e "${GREEN}✓ Poetry found${NC}"

# Check if token is configured
if ! poetry config pypi-token.pypi &> /dev/null; then
    echo -e "${YELLOW}⚠ PyPI token not configured${NC}"
    echo ""
    echo "Please configure your PyPI token:"
    echo "  1. Get token from: https://pypi.org/manage/account/"
    echo "  2. Run: poetry config pypi-token.pypi pypi-YOUR_TOKEN"
    echo ""
    read -p "Have you configured the token? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}✓ PyPI token configured${NC}"

# Get current version
VERSION=$(poetry version -s)
echo ""
echo "Current version: $VERSION"
echo ""

# Confirm publication
read -p "Publish morphml v$VERSION to PyPI? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Publication cancelled."
    exit 0
fi

echo ""
echo "📦 Step 1: Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info
echo -e "${GREEN}✓ Cleaned${NC}"

echo ""
echo "🔨 Step 2: Building package..."
poetry build
echo -e "${GREEN}✓ Built${NC}"

echo ""
echo "📋 Step 3: Checking package contents..."
ls -lh dist/
echo ""

echo "🧪 Step 4: Testing package..."
# Create temporary venv
python -m venv .test_env
source .test_env/bin/activate

# Install and test
pip install -q dist/morphml-$VERSION-py3-none-any.whl

# Test import
if python -c "import morphml; print('✓ Import successful')" 2>/dev/null; then
    echo -e "${GREEN}✓ Package test passed${NC}"
else
    echo -e "${RED}❌ Package test failed${NC}"
    deactivate
    rm -rf .test_env
    exit 1
fi

# Cleanup test env
deactivate
rm -rf .test_env

echo ""
echo "🚀 Step 5: Publishing to PyPI..."
poetry publish

echo ""
echo -e "${GREEN}✅ Successfully published morphml v$VERSION to PyPI!${NC}"
echo ""
echo "📝 Next steps:"
echo "  1. Wait 2-3 minutes for PyPI to index"
echo "  2. Test installation: pip install morphml"
echo "  3. Create GitHub release: git tag -a v$VERSION -m 'Release v$VERSION'"
echo "  4. Push tag: git push origin v$VERSION"
echo ""
echo "🎉 Congratulations on publishing MorphML!"
