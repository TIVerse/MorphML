#!/usr/bin/env python3
"""Test MorphML installation and dependencies.

Verifies that all components can be imported and basic functionality works.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dependencies():
    """Check required dependencies."""
    print("\n" + "="*80)
    print(" "*25 + "Checking Dependencies")
    print("="*80 + "\n")
    
    required = {
        "numpy": "Core numerical library",
        "rich": "Beautiful terminal output",
    }
    
    optional = {
        "torch": "PyTorch (for gradient-based optimizers)",
        "gpytorch": "Gaussian Processes (for Bayesian optimization)",
        "botorch": "Bayesian Optimization",
        "cma": "CMA-ES optimizer",
        "grpcio": "gRPC (for distributed)",
        "redis": "Redis cache",
        "sqlalchemy": "Database ORM",
        "boto3": "AWS S3 storage",
        "matplotlib": "Visualization",
    }
    
    print("Required Dependencies:")
    print("-" * 80)
    
    all_required_met = True
    for package, description in required.items():
        try:
            __import__(package)
            print(f"  ✅ {package:<15} - {description}")
        except ImportError:
            print(f"  ❌ {package:<15} - {description} (MISSING)")
            all_required_met = False
    
    print("\nOptional Dependencies:")
    print("-" * 80)
    
    for package, description in optional.items():
        try:
            __import__(package)
            print(f"  ✅ {package:<15} - {description}")
        except ImportError:
            print(f"  ⚠️  {package:<15} - {description} (optional)")
    
    print("\n" + "="*80 + "\n")
    
    if not all_required_met:
        print("❌ Some required dependencies are missing!")
        print("\nInstall with:")
        print("  pip install numpy rich")
        print("\nOr install MorphML with Poetry:")
        print("  poetry install")
        print("\n")
        return False
    
    return True


def test_imports():
    """Test that core modules can be imported."""
    print("Testing Core Imports...")
    print("-" * 80)
    
    tests = [
        ("morphml.core.dsl", "DSL"),
        ("morphml.core.graph", "Graph"),
        ("morphml.core.search", "Search"),
        ("morphml.optimizers", "Optimizers"),
        ("morphml.distributed", "Distributed"),
        ("morphml.benchmarks", "Benchmarks"),
    ]
    
    passed = 0
    failed = 0
    
    for module, name in tests:
        try:
            __import__(module)
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name} - {e}")
            failed += 1
    
    print(f"\n  Passed: {passed}/{len(tests)}")
    print("="*80 + "\n")
    
    return failed == 0


def test_basic_functionality():
    """Test basic functionality."""
    print("Testing Basic Functionality...")
    print("-" * 80)
    
    try:
        from morphml.core.dsl import Layer, SearchSpace
        
        # Create search space
        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64),
            Layer.output(units=10)
        )
        
        # Sample architecture
        graph = space.sample()
        
        print(f"  ✅ Created search space with {len(graph.layers)} layers")
        print(f"  ✅ Graph serialization works")
        
        return True
    
    except Exception as e:
        print(f"  ❌ Basic functionality failed: {e}")
        return False


def print_next_steps():
    """Print next steps."""
    print("\n" + "="*80)
    print(" "*25 + "Next Steps")
    print("="*80 + "\n")
    
    print("1. Run comprehensive tests:")
    print("   python tests/run_local_tests.py")
    print()
    print("2. Run existing test suite:")
    print("   pytest tests/ -v")
    print()
    print("3. Run benchmarks:")
    print("   python benchmarks/run_all_benchmarks.py")
    print()
    print("4. Try examples:")
    print("   python examples/quickstart.py")
    print()
    print("5. Deploy to Kubernetes:")
    print("   ./deployment/scripts/deploy.sh")
    print("\n" + "="*80 + "\n")


def main():
    """Run installation tests."""
    print("\n" + "🚀"*40)
    print(" "*25 + "MorphML Installation Test")
    print("🚀"*40 + "\n")
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        sys.exit(1)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("⚠️  Some imports failed. Check error messages above.")
        sys.exit(1)
    
    # Test basic functionality
    func_ok = test_basic_functionality()
    
    if not func_ok:
        print("⚠️  Basic functionality test failed.")
        sys.exit(1)
    
    # Success!
    print("\n" + "="*80)
    print("🎉 MorphML is installed correctly!")
    print("="*80)
    
    print_next_steps()


if __name__ == "__main__":
    main()
