"""
MorphML Text-based DSL Example

Demonstrates the complete DSL pipeline:
Lexer → Parser → AST → Validator → Type Checker → Compiler

This example shows how to use the text-based DSL to define search spaces
declaratively, which can be useful for:
- External configuration files
- Dynamic search space generation
- Integration with other tools

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from morphml.core.dsl import (
    Lexer,
    Parser,
    Compiler,
    Validator,
    TypeChecker,
    parse_dsl,
    compile_dsl,
    validate_ast,
    check_types,
)


def example_1_basic_parsing():
    """Example 1: Basic lexing and parsing."""
    print("=" * 60)
    print("Example 1: Basic Lexing and Parsing")
    print("=" * 60)

    # Define search space in DSL syntax
    source = """
    SearchSpace(
        layers=[
            Layer.conv2d(filters=[32, 64, 128], kernel_size=3),
            Layer.relu(),
            Layer.maxpool(pool_size=2),
            Layer.dense(units=[128, 256])
        ]
    )
    """

    print("\nSource Code:")
    print(source)

    # Step 1: Lexical analysis
    print("\n1. Lexing...")
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    print(f"   Generated {len(tokens)} tokens")
    print(f"   First 5 tokens: {tokens[:5]}")

    # Step 2: Parsing
    print("\n2. Parsing...")
    parser = Parser(tokens)
    ast = parser.parse()
    print(f"   AST: {ast}")
    print(f"   Layers in search space: {len(ast.search_space.layers)}")

    return ast


def example_2_full_pipeline():
    """Example 2: Complete DSL pipeline with validation and compilation."""
    print("\n" + "=" * 60)
    print("Example 2: Full Pipeline (Parse → Validate → Type Check → Compile)")
    print("=" * 60)

    # Define a complete experiment with evolution config
    source = """
    SearchSpace(
        name="cifar10_cnn",
        layers=[
            Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5]),
            Layer.relu(),
            Layer.batch_norm(),
            Layer.maxpool(pool_size=2),
            Layer.conv2d(filters=[64, 128, 256], kernel_size=3),
            Layer.relu(),
            Layer.maxpool(pool_size=2),
            Layer.dense(units=[128, 256, 512]),
            Layer.dropout(rate=[0.3, 0.5]),
            Layer.dense(units=[64, 128]),
            Layer.output()
        ]
    )
    
    Evolution(
        strategy="genetic",
        population_size=50,
        num_generations=100,
        mutation_rate=0.15,
        crossover_rate=0.7,
        elite_size=5
    )
    """

    print("\nSource Code:")
    print(source)

    # Step 1: Parse
    print("\n1. Parsing...")
    ast = parse_dsl(source)
    print(f"   ✓ Parsed successfully")
    print(f"   ✓ Search space: {ast.search_space.name}")
    print(f"   ✓ Layers: {len(ast.search_space.layers)}")
    print(f"   ✓ Evolution: {ast.evolution.strategy if ast.evolution else 'None'}")

    # Step 2: Validate
    print("\n2. Validating...")
    validator = Validator()
    errors = validator.validate(ast)
    if errors:
        print(f"   ✗ Validation errors found: {len(errors)}")
        for error in errors:
            print(f"     - {error}")
    else:
        print(f"   ✓ No validation errors")
    
    if validator.warnings:
        print(f"   ⚠ Warnings: {len(validator.warnings)}")
        for warning in validator.warnings:
            print(f"     - {warning}")

    # Step 3: Type check
    print("\n3. Type checking...")
    type_checker = TypeChecker()
    type_errors = type_checker.check(ast)
    if type_errors:
        print(f"   ✗ Type errors found: {len(type_errors)}")
        for error in type_errors:
            print(f"     - {error}")
    else:
        print(f"   ✓ No type errors")

    # Step 4: Compile
    print("\n4. Compiling...")
    compiler = Compiler()
    result = compiler.compile(ast)
    search_space = result["search_space"]
    evolution_config = result["evolution"]
    
    print(f"   ✓ Compiled successfully")
    print(f"   ✓ Search space: {search_space}")
    print(f"   ✓ Evolution config: {evolution_config}")
    
    return result


def example_3_convenience_function():
    """Example 3: Using convenience functions."""
    print("\n" + "=" * 60)
    print("Example 3: Convenience Functions")
    print("=" * 60)

    source = """
    SearchSpace(
        layers=[
            Layer.dense(units=[64, 128, 256]),
            Layer.relu(),
            Layer.dropout(rate=[0.2, 0.5])
        ]
    )
    """

    print("\nUsing compile_dsl() convenience function:")
    print(source)

    # One-step compilation
    result = compile_dsl(source)
    
    print(f"\n✓ Compiled in one step!")
    print(f"  Search space: {result['search_space']}")
    print(f"  Can be used directly with optimizers")

    return result


def example_4_error_handling():
    """Example 4: Error detection and reporting."""
    print("\n" + "=" * 60)
    print("Example 4: Error Detection")
    print("=" * 60)

    # Invalid DSL (missing required parameter)
    invalid_source = """
    SearchSpace(
        layers=[
            Layer.conv2d(kernel_size=3),
            Layer.dense()
        ]
    )
    """

    print("\nInvalid Source (missing required params):")
    print(invalid_source)

    try:
        ast = parse_dsl(invalid_source)
        
        print("\n✓ Parsing succeeded (syntax is valid)")
        
        # But validation should catch semantic errors
        print("\nRunning validator...")
        errors = validate_ast(ast)
        if errors:
            print(f"✓ Validator caught {len(errors)} error(s):")
            for error in errors:
                print(f"  - {error}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_5_comparing_approaches():
    """Example 5: Compare Pythonic vs Text-based DSL."""
    print("\n" + "=" * 60)
    print("Example 5: Comparing Both Approaches")
    print("=" * 60)

    # Approach 1: Pythonic API
    print("\nApproach 1: Pythonic Builder Pattern")
    print("-" * 40)
    from morphml.core.dsl import SearchSpace, Layer
    
    space_pythonic = SearchSpace("my_space")
    space_pythonic.add_layers(
        Layer.conv2d(filters=[32, 64], kernel_size=3),
        Layer.relu(),
        Layer.dense(units=[128, 256])
    )
    print("space = SearchSpace('my_space')")
    print("space.add_layers(")
    print("    Layer.conv2d(filters=[32, 64], kernel_size=3),")
    print("    Layer.relu(),")
    print("    Layer.dense(units=[128, 256])")
    print(")")
    print(f"Result: {space_pythonic}")

    # Approach 2: Text-based DSL
    print("\n\nApproach 2: Text-based DSL")
    print("-" * 40)
    dsl_source = """
    SearchSpace(
        name="my_space",
        layers=[
            Layer.conv2d(filters=[32, 64], kernel_size=3),
            Layer.relu(),
            Layer.dense(units=[128, 256])
        ]
    )
    """
    print(dsl_source)
    
    result = compile_dsl(dsl_source)
    space_dsl = result["search_space"]
    print(f"Result: {space_dsl}")

    print("\n\nComparison:")
    print(f"  Pythonic layers: {len(space_pythonic.layers)}")
    print(f"  DSL layers:      {len(space_dsl.layers)}")
    print(f"  Both approaches create equivalent structures!")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "MorphML Text-based DSL Examples" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Run examples
    try:
        example_1_basic_parsing()
        example_2_full_pipeline()
        example_3_convenience_function()
        example_4_error_handling()
        example_5_comparing_approaches()
        
        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
