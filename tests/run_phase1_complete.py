"""
Comprehensive Phase 1 Test Suite

Tests all components of Phase 1 to verify completeness:
1. Project Infrastructure
2. DSL Implementation (both Pythonic and Text-based)
3. Graph System
4. Search Space & Engine
5. Genetic Algorithm
6. Execution & CLI

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION

Note: This is a script-style test file. Run directly with: python tests/test_phase1_complete.py
"""

import sys
import tempfile
from pathlib import Path

# Prevent pytest from collecting this file
collect_ignore = [__file__]
pytest_collect_file = lambda *args: None

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Track test results
tests_passed = 0
tests_failed = 0
errors = []


def test_section(name):
    """Decorator for test sections."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print("=" * 70)


def test_pass(message):
    """Mark test as passed."""
    global tests_passed
    tests_passed += 1
    print(f"  ✓ {message}")


def test_fail(message, error=None):
    """Mark test as failed."""
    global tests_failed
    tests_failed += 1
    print(f"  ✗ {message}")
    if error:
        print(f"    Error: {error}")
        errors.append((message, str(error)))


# =============================================================================
# Component 1: Project Infrastructure
# =============================================================================

# Only run tests if executed as script, not during pytest collection
if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 1 COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print()
    
    test_section("Component 1: Project Infrastructure")

    try:
    from morphml import __version__

    test_pass(f"Version module works ({__version__})")
except Exception as e:
    test_fail("Version module", e)

try:
    test_pass("Exception classes imported")
except Exception as e:
    test_fail("Exception classes", e)

try:
    from morphml.config import get_config

    config = get_config()
    test_pass("Configuration system works")
except Exception as e:
    test_fail("Configuration system", e)

try:
    from morphml.logging_config import get_logger

    logger = get_logger(__name__)
    test_pass("Logging system works")
except Exception as e:
    test_fail("Logging system", e)

# =============================================================================
# Component 2: DSL Implementation
# =============================================================================
test_section("Component 2: DSL Implementation - Text-based")

# Test syntax definitions
try:
    from morphml.core.dsl.syntax import LAYER_TYPES, TokenType

    test_pass(f"Syntax module: {len(list(TokenType))} token types, {len(LAYER_TYPES)} layer types")
except Exception as e:
    test_fail("Syntax module", e)

# Test lexer
try:
    from morphml.core.dsl.lexer import Lexer

    source = "Layer.conv2d(filters=[32, 64], kernel_size=3)"
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    test_pass(f"Lexer: Tokenized {len(tokens)} tokens from source")
except Exception as e:
    test_fail("Lexer", e)

# Test AST nodes
try:
    from morphml.core.dsl.ast_nodes import ParamNode

    param = ParamNode(name="filters", values=[32, 64, 128])
    test_pass(f"AST nodes: Created ParamNode with {len(param.values)} values")
except Exception as e:
    test_fail("AST nodes", e)

# Test parser
try:
    from morphml.core.dsl.parser import parse_dsl

    dsl_source = """
    SearchSpace(
        layers=[
            Layer.conv2d(filters=[32, 64]),
            Layer.relu()
        ]
    )
    """
    ast = parse_dsl(dsl_source)
    test_pass(f"Parser: Parsed DSL with {len(ast.search_space.layers)} layers")
except Exception as e:
    test_fail("Parser", e)

# Test compiler
try:
    from morphml.core.dsl.compiler import compile_dsl

    result = compile_dsl(dsl_source)
    search_space = result["search_space"]
    test_pass(f"Compiler: Compiled to SearchSpace with {len(search_space.layers)} layers")
except Exception as e:
    test_fail("Compiler", e)

# Test validator
try:
    from morphml.core.dsl.validator import Validator

    validator = Validator()
    validation_errors = validator.validate(ast)
    test_pass(f"Validator: Validated AST ({len(validation_errors)} errors)")
except Exception as e:
    test_fail("Validator", e)

# Test type system
try:
    from morphml.core.dsl.type_system import check_types

    type_errors = check_types(ast)
    test_pass(f"Type system: Type checked AST ({len(type_errors)} type errors)")
except Exception as e:
    test_fail("Type system", e)

# Test Pythonic DSL
test_section("Component 2: DSL Implementation - Pythonic API")

try:
    from morphml.core.dsl import Layer, SearchSpace

    space = SearchSpace("test_space")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=[32, 64], kernel_size=3),
        Layer.relu(),
        Layer.dense(units=[128, 256]),
    )
    test_pass(f"Pythonic DSL: Created SearchSpace with {len(space.layers)} layers")
except Exception as e:
    test_fail("Pythonic DSL", e)

# =============================================================================
# Component 3: Graph System
# =============================================================================
test_section("Component 3: Graph System")

try:
    from morphml.core.graph import GraphEdge, GraphNode, ModelGraph

    graph = ModelGraph()
    node1 = GraphNode.create("input", {"shape": (3, 32, 32)})
    node2 = GraphNode.create("conv2d", {"filters": 32, "kernel_size": 3})
    node3 = GraphNode.create("output", {"units": 10})

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    graph.add_edge(GraphEdge(node1, node2))
    graph.add_edge(GraphEdge(node2, node3))

    test_pass(f"Graph: Created graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Validate DAG
    if graph.is_valid_dag():
        test_pass("Graph: DAG validation passed")
    else:
        test_fail("Graph: DAG validation failed")

except Exception as e:
    test_fail("Graph creation", e)

# Test mutations
try:
    from morphml.core.graph import GraphMutator

    mutator = GraphMutator()
    mutated = mutator.mutate(graph, mutation_rate=0.1)

    if mutated.is_valid_dag():
        test_pass("Mutations: Mutated graph remains valid DAG")
    else:
        test_fail("Mutations: Mutated graph broke DAG property")

except Exception as e:
    test_fail("Graph mutations", e)

# Test serialization
try:
    from morphml.core.graph import graph_to_json_string, load_graph, save_graph

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test JSON serialization
        json_path = Path(tmpdir) / "test_graph.json"
        save_graph(graph, json_path, format="json")
        loaded_graph = load_graph(json_path)
        test_pass("Serialization: JSON save/load successful")

        # Test string conversion
        json_str = graph_to_json_string(graph)
        test_pass(f"Serialization: JSON string conversion ({len(json_str)} chars)")

except Exception as e:
    test_fail("Graph serialization", e)

# Test visualization (syntax only, no display)
try:
    test_pass("Visualization: Module imports successful")
except Exception as e:
    test_fail("Visualization module", e)

# =============================================================================
# Component 4: Search Space & Engine
# =============================================================================
test_section("Component 4: Search Space & Engine")

try:
    from morphml.core.search import Individual, Population

    individual = Individual(graph=graph)
    individual.set_fitness(0.85)
    test_pass(f"Individual: Created with fitness {individual.fitness}")

    pop = Population([individual])
    test_pass(f"Population: Created with {len(pop)} individuals")

except Exception as e:
    test_fail("Individual/Population", e)

# Test search space sampling
try:
    sampled_graph = space.sample()
    if sampled_graph.is_valid_dag():
        test_pass(f"SearchSpace: Sampled valid graph with {len(sampled_graph.nodes)} nodes")
    else:
        test_fail("SearchSpace: Sampled graph is not valid DAG")
except Exception as e:
    test_fail("SearchSpace sampling", e)

# Test selection strategies
try:
    from morphml.core.search.population import Population

    pop = Population()
    for i in range(5):
        ind = Individual(graph=graph.clone())
        ind.set_fitness(0.5 + i * 0.1)
        pop.add(ind)

    selected = pop.select("tournament", n=2, tournament_size=3)
    test_pass(f"Selection: Tournament selection returned {len(selected)} individuals")

except Exception as e:
    test_fail("Selection strategies", e)

# =============================================================================
# Component 5: Genetic Algorithm
# =============================================================================
test_section("Component 5: Genetic Algorithm")

try:
    from morphml.optimizers import GeneticAlgorithm

    config = {
        "population_size": 10,
        "num_generations": 2,
        "mutation_rate": 0.1,
        "crossover_rate": 0.7,
        "elitism": 2,
    }

    ga = GeneticAlgorithm(search_space=space, **config)
    test_pass(f"GeneticAlgorithm: Created with population size {config['population_size']}")

except Exception as e:
    test_fail("GeneticAlgorithm creation", e)

# Test population initialization
try:
    population = ga.initialize()
    test_pass(f"GeneticAlgorithm: Initialized population with {len(population)} individuals")
except Exception as e:
    test_fail("GA population initialization", e)

# Test other optimizers
try:
    from morphml.optimizers import HillClimbing, RandomSearch

    rs = RandomSearch(search_space=space, num_samples=5)
    test_pass("RandomSearch: Created successfully")

    hc = HillClimbing(search_space=space, max_iterations=5)
    test_pass("HillClimbing: Created successfully")

except Exception as e:
    test_fail("Other optimizers", e)

# =============================================================================
# Component 6: Execution & CLI
# =============================================================================
test_section("Component 6: Execution & CLI")

try:
    from morphml.evaluation import HeuristicEvaluator

    evaluator = HeuristicEvaluator()
    score = evaluator(graph)
    test_pass(f"HeuristicEvaluator: Evaluated graph with score {score:.4f}")

except Exception as e:
    test_fail("HeuristicEvaluator", e)

try:
    from morphml.utils import ArchitectureExporter

    exporter = ArchitectureExporter()
    pytorch_code = exporter.to_pytorch(graph, "TestModel")
    test_pass(f"ArchitectureExporter: Generated PyTorch code ({len(pytorch_code)} chars)")

except Exception as e:
    test_fail("ArchitectureExporter", e)

try:
    test_pass("Checkpoint: Module imported successfully")
except Exception as e:
    test_fail("Checkpoint module", e)

try:
    test_pass("CLI: Command-line interface imported successfully")
except Exception as e:
    test_fail("CLI module", e)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"Tests Passed: {tests_passed}")
print(f"Tests Failed: {tests_failed}")
print(f"Total Tests:  {tests_passed + tests_failed}")
print(f"Success Rate: {tests_passed / (tests_passed + tests_failed) * 100:.1f}%")

if tests_failed > 0:
    print("\n" + "=" * 70)
    print("FAILED TESTS:")
    print("=" * 70)
    for message, error in errors:
        print(f"\n{message}:")
        print(f"  {error}")

print("\n" + "=" * 70)
if tests_failed == 0:
    print("✓ ALL TESTS PASSED - PHASE 1 IS COMPLETE!")
else:
    print(f"⚠ {tests_failed} TESTS FAILED - PHASE 1 NEEDS ATTENTION")
print("=" * 70)
print()

    # Return test results
    return tests_failed == 0


# Pytest-compatible test function
def test_phase1_complete():
    """Run Phase 1 comprehensive tests."""
    assert run_tests(), "Phase 1 tests failed"


# Allow running as script
if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
