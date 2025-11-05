"""Token definitions and grammar for MorphML DSL.

This module defines all tokens, keywords, and grammar rules for the
text-based DSL that allows users to define search spaces using a
declarative syntax.

Grammar (EBNF):
    experiment := search_space_def evolution_def
    search_space_def := "SearchSpace" "(" param_list ")"
    layer_def := "Layer" "." layer_type "(" param_list ")"
    param_list := param ("," param)*
    param := IDENTIFIER "=" value_list
    value_list := "[" value ("," value)* "]" | value
    value := NUMBER | STRING | BOOLEAN | IDENTIFIER

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from enum import Enum, auto


class TokenType(Enum):
    """All token types in the MorphML DSL."""

    # Literals
    NUMBER = auto()
    STRING = auto()
    BOOLEAN = auto()

    # Identifiers and keywords
    IDENTIFIER = auto()
    SEARCHSPACE = auto()
    LAYER = auto()
    EVOLUTION = auto()
    EXPERIMENT = auto()
    CONSTRAINT = auto()

    # Operators and delimiters
    ASSIGN = auto()
    COMMA = auto()
    DOT = auto()
    COLON = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()

    # Special
    EOF = auto()
    NEWLINE = auto()
    COMMENT = auto()


# Keywords that are recognized as special tokens
KEYWORDS = {
    "SearchSpace": TokenType.SEARCHSPACE,
    "Layer": TokenType.LAYER,
    "Evolution": TokenType.EVOLUTION,
    "Experiment": TokenType.EXPERIMENT,
    "Constraint": TokenType.CONSTRAINT,
    "True": TokenType.BOOLEAN,
    "False": TokenType.BOOLEAN,
    "true": TokenType.BOOLEAN,
    "false": TokenType.BOOLEAN,
}

# Supported layer types in the DSL
LAYER_TYPES = [
    "conv2d",
    "conv3d",
    "dense",
    "linear",
    "batch_norm",
    "layer_norm",
    "dropout",
    "max_pool",
    "avg_pool",
    "global_avg_pool",
    "activation",
    "flatten",
    "relu",
    "elu",
    "gelu",
    "tanh",
    "sigmoid",
    "leaky_relu",
    "softmax",
    "input",
    "output",
]

# Optimizer types
OPTIMIZER_TYPES = [
    "adam",
    "sgd",
    "rmsprop",
    "adamw",
    "adagrad",
]

# Activation functions
ACTIVATION_TYPES = [
    "relu",
    "elu",
    "gelu",
    "tanh",
    "sigmoid",
    "leaky_relu",
    "softmax",
    "selu",
    "swish",
]

# Evolution strategies
EVOLUTION_STRATEGIES = [
    "genetic",
    "random_search",
    "hill_climbing",
    "differential_evolution",
    "cma_es",
    "bayesian",
    "nsga2",
]

# Operator symbols
OPERATORS = {
    "=": TokenType.ASSIGN,
    ",": TokenType.COMMA,
    ".": TokenType.DOT,
    ":": TokenType.COLON,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
}


def is_layer_type(name: str) -> bool:
    """Check if name is a valid layer type."""
    return name in LAYER_TYPES


def is_optimizer_type(name: str) -> bool:
    """Check if name is a valid optimizer type."""
    return name in OPTIMIZER_TYPES


def is_activation_type(name: str) -> bool:
    """Check if name is a valid activation function."""
    return name in ACTIVATION_TYPES


def is_evolution_strategy(name: str) -> bool:
    """Check if name is a valid evolution strategy."""
    return name in EVOLUTION_STRATEGIES


# Grammar documentation
GRAMMAR_DOCS = """
MorphML DSL Grammar (Extended Backus-Naur Form)

experiment        := search_space_def [evolution_def] [constraint_list]
search_space_def  := "SearchSpace" "(" [space_params] ")"
space_params      := "layers" "=" layer_list ["," param_list]
layer_list        := "[" layer_def ("," layer_def)* "]"
layer_def         := "Layer" "." layer_type "(" [param_list] ")"
evolution_def     := "Evolution" "(" param_list ")"
constraint_list   := "Constraint" "(" constraint_expr ("," constraint_expr)* ")"

param_list        := param ("," param)*
param             := IDENTIFIER "=" value_expr
value_expr        := value | value_list
value_list        := "[" value ("," value)* "]"
value             := NUMBER | STRING | BOOLEAN | IDENTIFIER

layer_type        := IDENTIFIER  # Must be in LAYER_TYPES
constraint_expr   := STRING | function_call

Examples:
    SearchSpace(
        layers=[
            Layer.conv2d(filters=[32, 64, 128], kernel_size=3),
            Layer.relu(),
            Layer.maxpool(pool_size=2),
            Layer.dense(units=[128, 256])
        ]
    )

    Evolution(
        strategy="genetic",
        population_size=50,
        num_generations=100
    )
"""
