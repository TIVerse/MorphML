"""Domain-Specific Language for search space definition.

MorphML provides two approaches to defining search spaces:

1. Pythonic Builder Pattern (Recommended):
   - Direct Python API
   - Better IDE support
   - Type checking
   - Example: space.add_layers(Layer.conv2d(filters=[32, 64]))

2. Text-based DSL (Advanced):
   - Declarative syntax
   - Lexer → Parser → AST → Compiler
   - Useful for external configuration
   - Example: parse_dsl("SearchSpace(layers=[Layer.conv2d(...)])")
"""

# Pythonic API (Primary interface)
from morphml.core.dsl.layers import Layer, LayerSpec
from morphml.core.dsl.search_space import SearchSpace, create_cnn_space, create_mlp_space

# Text-based DSL components
from morphml.core.dsl.syntax import TokenType, KEYWORDS, LAYER_TYPES, OPERATORS
from morphml.core.dsl.lexer import Lexer, Token
from morphml.core.dsl.parser import Parser, parse_dsl
from morphml.core.dsl.ast_nodes import (
    ASTNode,
    ParamNode,
    LayerNode,
    SearchSpaceNode,
    EvolutionNode,
    ConstraintNode,
    ExperimentNode,
    ASTVisitor,
)
from morphml.core.dsl.compiler import Compiler, compile_dsl, compile_to_search_space
from morphml.core.dsl.validator import Validator, validate_ast
from morphml.core.dsl.type_system import TypeChecker, Type, check_types

__all__ = [
    # Pythonic API
    "Layer",
    "LayerSpec",
    "SearchSpace",
    "create_cnn_space",
    "create_mlp_space",
    # Text-based DSL
    "TokenType",
    "KEYWORDS",
    "LAYER_TYPES",
    "OPERATORS",
    "Lexer",
    "Token",
    "Parser",
    "parse_dsl",
    "ASTNode",
    "ParamNode",
    "LayerNode",
    "SearchSpaceNode",
    "EvolutionNode",
    "ConstraintNode",
    "ExperimentNode",
    "ASTVisitor",
    "Compiler",
    "compile_dsl",
    "compile_to_search_space",
    "Validator",
    "validate_ast",
    "TypeChecker",
    "Type",
    "check_types",
]
