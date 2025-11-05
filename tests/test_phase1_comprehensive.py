"""
Comprehensive Phase 1 Test - Check Everything

This test systematically checks every component and requirement
from the Phase 1 prompts to ensure nothing is missing.

Author: Eshan Roy <eshanized@proton.me>
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("PHASE 1 COMPREHENSIVE VERIFICATION TEST")
print("=" * 80)
print()

tests_passed = 0
tests_failed = 0
warnings = []
errors = []


def test_pass(msg):
    global tests_passed
    tests_passed += 1
    print(f"  ✓ {msg}")


def test_fail(msg, error=None):
    global tests_failed
    tests_failed += 1
    print(f"  ✗ {msg}")
    if error:
        errors.append((msg, str(error)))


def test_warn(msg):
    global warnings
    warnings.append(msg)
    print(f"  ⚠ {msg}")


def section(name):
    print(f"\n{'=' * 80}")
    print(f"{name}")
    print("=" * 80)


# =============================================================================
# COMPONENT 1: PROJECT INFRASTRUCTURE
# =============================================================================
section("COMPONENT 1: Project Infrastructure (01_project_setup.md)")

files_to_check = [
    ("pyproject.toml", "Poetry configuration"),
    (".gitignore", "Git ignore file"),
    ("morphml/version.py", "Version module"),
    ("morphml/exceptions.py", "Exception hierarchy"),
    ("morphml/config.py", "Configuration system"),
    ("morphml/logging_config.py", "Logging configuration"),
]

for filepath, desc in files_to_check:
    p = Path(filepath)
    if p.exists():
        test_pass(f"{desc}: {filepath}")
    else:
        test_fail(f"{desc}: {filepath} NOT FOUND")

# =============================================================================
# COMPONENT 2: DSL IMPLEMENTATION
# =============================================================================
section("COMPONENT 2: DSL Implementation (02_dsl_implementation.md)")

print("\n📋 Text-based DSL Files:")
dsl_files = [
    "morphml/core/dsl/syntax.py",
    "morphml/core/dsl/lexer.py",
    "morphml/core/dsl/ast_nodes.py",
    "morphml/core/dsl/parser.py",
    "morphml/core/dsl/compiler.py",
    "morphml/core/dsl/validator.py",
    "morphml/core/dsl/type_system.py",
]

for filepath in dsl_files:
    if Path(filepath).exists():
        test_pass(f"{Path(filepath).name}")
    else:
        test_fail(f"{Path(filepath).name} NOT FOUND")

print("\n📋 Pythonic DSL Files:")
pythonic_files = [
    "morphml/core/dsl/layers.py",
    "morphml/core/dsl/search_space.py",
]

for filepath in pythonic_files:
    if Path(filepath).exists():
        test_pass(f"{Path(filepath).name}")
    else:
        test_fail(f"{Path(filepath).name} NOT FOUND")

# =============================================================================
# COMPONENT 3: GRAPH SYSTEM
# =============================================================================
section("COMPONENT 3: Graph System (03_graph_system.md)")

graph_files = [
    ("morphml/core/graph/node.py", "GraphNode"),
    ("morphml/core/graph/edge.py", "GraphEdge"),
    ("morphml/core/graph/graph.py", "ModelGraph"),
    ("morphml/core/graph/mutations.py", "Mutations"),
    ("morphml/core/graph/serialization.py", "Serialization"),
    ("morphml/core/graph/visualization.py", "Visualization"),
]

for filepath, desc in graph_files:
    if Path(filepath).exists():
        test_pass(f"{desc}: {Path(filepath).name}")
    else:
        test_fail(f"{desc}: {filepath} NOT FOUND")

# =============================================================================
# COMPONENT 4: SEARCH SPACE & ENGINE
# =============================================================================
section("COMPONENT 4: Search Space & Engine (04_search_engine.md)")

print("\n📋 Required Classes:")

# Check for parameter classes
try:
    from morphml.core.search.parameters import (
        Parameter,
        CategoricalParameter,
        IntegerParameter,
        FloatParameter,
        BooleanParameter,
    )
    test_pass("Parameter classes (CategoricalParameter, IntegerParameter, FloatParameter, BooleanParameter)")
except Exception as e:
    test_fail("Parameter classes", e)

# Check for search engine
try:
    from morphml.core.search.search_engine import SearchEngine
    test_pass("SearchEngine base class")
except Exception as e:
    test_fail("SearchEngine base class", e)

# Check for population management
try:
    from morphml.core.search.individual import Individual
    from morphml.core.search.population import Population
    test_pass("Individual and Population classes")
except Exception as e:
    test_fail("Individual and Population classes", e)

# Check for search space
try:
    from morphml.core.dsl.search_space import SearchSpace
    test_pass("SearchSpace class")
except Exception as e:
    test_fail("SearchSpace class", e)

print("\n📋 Selection Strategies:")
try:
    from morphml.core.search.population import Population
    p = Population()
    
    # Check if selection methods exist
    if hasattr(p, 'select'):
        test_pass("Selection methods available")
    else:
        test_fail("Selection methods not found")
except Exception as e:
    test_fail("Selection strategies", e)

# =============================================================================
# COMPONENT 5: GENETIC ALGORITHM
# =============================================================================
section("COMPONENT 5: Genetic Algorithm (05_genetic_algorithm.md)")

optimizers = [
    ("morphml/optimizers/genetic_algorithm.py", "GeneticAlgorithm"),
    ("morphml/optimizers/random_search.py", "RandomSearch"),
    ("morphml/optimizers/hill_climbing.py", "HillClimbing"),
]

for filepath, name in optimizers:
    if Path(filepath).exists():
        test_pass(f"{name}: {Path(filepath).name}")
    else:
        test_fail(f"{name}: {filepath} NOT FOUND")

# Check crossover
if Path("../morphml/core/crossover.py").exists():
    test_pass("Crossover operators")
else:
    test_fail("Crossover operators NOT FOUND")

# =============================================================================
# COMPONENT 6: EXECUTION & CLI
# =============================================================================
section("COMPONENT 6: Execution & CLI (06_execution_cli.md)")

exec_files = [
    ("morphml/execution/local_executor.py", "LocalExecutor"),
    ("morphml/evaluation/heuristic.py", "HeuristicEvaluator"),
    ("morphml/utils/export.py", "ArchitectureExporter"),
    ("morphml/utils/checkpoint.py", "Checkpoint"),
    ("morphml/cli/main.py", "CLI"),
]

for filepath, desc in exec_files:
    if Path(filepath).exists():
        test_pass(f"{desc}: {Path(filepath).name}")
    else:
        test_fail(f"{desc}: {filepath} NOT FOUND")

# =============================================================================
# EXAMPLES & DOCUMENTATION
# =============================================================================
section("Examples & Documentation")

docs = [
    ("examples/quickstart.py", "Quickstart example"),
    ("examples/dsl_example.py", "DSL example"),
    ("README.md", "Main README"),
    ("morphml/core/dsl/README.md", "DSL documentation"),
]

for filepath, desc in docs:
    if Path(filepath).exists():
        test_pass(f"{desc}")
    else:
        test_fail(f"{desc} NOT FOUND")

# =============================================================================
# SYNTAX COMPILATION TEST
# =============================================================================
section("Syntax Compilation Test")

print("\n📋 Testing file compilation:")

important_files = [
    "morphml/core/dsl/syntax.py",
    "morphml/core/dsl/lexer.py",
    "morphml/core/dsl/parser.py",
    "morphml/core/dsl/compiler.py",
    "morphml/core/search/parameters.py",
    "morphml/core/search/search_engine.py",
    "morphml/execution/local_executor.py",
    "morphml/core/graph/serialization.py",
    "morphml/core/graph/visualization.py",
]

import py_compile
compilation_errors = []

for filepath in important_files:
    try:
        if Path(filepath).exists():
            py_compile.compile(filepath, doraise=True)
            test_pass(f"Compiles: {Path(filepath).name}")
        else:
            test_warn(f"File not found: {filepath}")
    except Exception as e:
        test_fail(f"Compilation error: {Path(filepath).name}", e)
        compilation_errors.append((filepath, str(e)))

# =============================================================================
# IMPORT TEST
# =============================================================================
section("Import Test (May fail without dependencies)")

print("\n📋 Testing imports:")

imports_to_test = [
    ("morphml.core.dsl", "Layer, SearchSpace"),
    ("morphml.core.graph", "ModelGraph, GraphNode"),
    ("morphml.core.search", "Individual, Population"),
]

for module, items in imports_to_test:
    try:
        exec(f"from {module} import {items}")
        test_pass(f"Import {module}: {items}")
    except Exception as e:
        test_warn(f"Import {module} failed (may need dependencies): {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"Tests Passed:  {tests_passed}")
print(f"Tests Failed:  {tests_failed}")
print(f"Warnings:      {len(warnings)}")
print(f"Total Tests:   {tests_passed + tests_failed}")
print(f"Success Rate:  {tests_passed / (tests_passed + tests_failed) * 100:.1f}%")

if compilation_errors:
    print("\n" + "=" * 80)
    print("COMPILATION ERRORS:")
    print("=" * 80)
    for filepath, error in compilation_errors:
        print(f"\n{filepath}:")
        print(f"  {error}")

if errors:
    print("\n" + "=" * 80)
    print("FAILED TESTS:")
    print("=" * 80)
    for msg, error in errors:
        print(f"\n{msg}:")
        print(f"  {error}")

if warnings:
    print("\n" + "=" * 80)
    print("WARNINGS:")
    print("=" * 80)
    for warning in warnings:
        print(f"  ⚠ {warning}")

print("\n" + "=" * 80)
if tests_failed == 0:
    print("✓ ALL STRUCTURAL TESTS PASSED!")
    print("  Phase 1 is structurally complete.")
    print("  Run 'poetry install' then test_phase1_complete.py for functional tests.")
else:
    print(f"⚠ {tests_failed} TESTS FAILED")
    print("  Some components may be missing or have issues.")
print("=" * 80)
print()

sys.exit(0 if tests_failed == 0 else 1)
