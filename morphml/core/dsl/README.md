# MorphML DSL Implementation

This directory contains the complete text-based Domain-Specific Language (DSL) implementation for MorphML, as specified in Phase 1, Component 2.

## Architecture

```
Text DSL Source
      ↓
   syntax.py (Token definitions, grammar)
      ↓
   lexer.py (Tokenization)
      ↓
   parser.py (AST construction)
      ↓
   ast_nodes.py (AST node classes)
      ↓
   validator.py (Semantic validation)
      ↓
   type_system.py (Type checking)
      ↓
   compiler.py (Compile to internal IR)
      ↓
SearchSpace + Config
```

## Files Implemented

### 1. `syntax.py` (~300 LOC)
**Purpose:** Token definitions and grammar rules

**Key Components:**
- `TokenType` enum - All token types (NUMBER, STRING, IDENTIFIER, etc.)
- `KEYWORDS` - Reserved words (SearchSpace, Layer, Evolution, etc.)
- `LAYER_TYPES` - Supported layer types (conv2d, dense, dropout, etc.)
- `OPERATORS` - Operators and delimiters (=, ,, ., etc.)
- Grammar documentation in EBNF notation

### 2. `lexer.py` (~800 LOC)
**Purpose:** Convert source code into token stream

**Key Classes:**
- `Token` - Represents a single token with type, value, line, column
- `Lexer` - Main lexical analyzer

**Features:**
- Handles numbers (int, float, scientific notation)
- Handles strings (single/double quotes, escape sequences)
- Handles comments (# style)
- Tracks line and column for error messages
- Comprehensive error reporting with context

**Example:**
```python
from morphml.core.dsl import Lexer

source = "Layer.conv2d(filters=[32, 64])"
lexer = Lexer(source)
tokens = lexer.tokenize()
# Returns: [Token(LAYER, 'Layer'), Token(DOT, '.'), ...]
```

### 3. `ast_nodes.py` (~600 LOC)
**Purpose:** Define Abstract Syntax Tree node classes

**Key Classes:**
- `ASTNode` - Base class for all AST nodes
- `ParamNode` - Hyperparameter specification
- `LayerNode` - Layer specification
- `SearchSpaceNode` - Complete search space
- `EvolutionNode` - Evolution configuration
- `ConstraintNode` - Constraint specification
- `ExperimentNode` - Root node (complete experiment)
- `ASTVisitor` - Visitor pattern for AST traversal

**Features:**
- Immutable dataclasses (frozen=True)
- Automatic validation in __post_init__
- Type inference for parameters
- Visitor pattern support

**Example:**
```python
layer = LayerNode(
    layer_type='conv2d',
    params={'filters': ParamNode('filters', [32, 64])}
)
```

### 4. `parser.py` (~1,200 LOC)
**Purpose:** Parse token stream into AST

**Key Classes:**
- `Parser` - Recursive descent parser

**Methods:**
- `parse()` - Parse complete experiment
- `parse_search_space()` - Parse search space definition
- `parse_layer()` - Parse layer specification
- `parse_evolution()` - Parse evolution config
- `_parse_param_list()` - Parse parameters
- `_parse_value_expr()` - Parse values or value lists

**Features:**
- Recursive descent parsing
- Clear error messages with position
- Helper methods for token manipulation
- Convenience function `parse_dsl(source)`

**Example:**
```python
from morphml.core.dsl import Parser, parse_dsl

source = '''
SearchSpace(
    layers=[Layer.conv2d(filters=[32, 64])]
)
'''
ast = parse_dsl(source)
```

### 5. `compiler.py` (~800 LOC)
**Purpose:** Compile AST into executable internal representation

**Key Classes:**
- `Compiler` - Main compiler
- `CompilationContext` - Tracks compilation state

**Methods:**
- `compile()` - Compile complete experiment
- `_compile_search_space()` - Convert SearchSpaceNode to SearchSpace
- `_compile_layer()` - Convert LayerNode to LayerSpec
- `_compile_evolution()` - Convert to optimizer config
- `_compile_constraint()` - Convert to constraint config

**Features:**
- Type inference from values
- Parameter mapping (DSL names → internal names)
- Symbol table management
- Convenience function `compile_dsl(source)`

**Example:**
```python
from morphml.core.dsl import compile_dsl

source = "SearchSpace(layers=[Layer.dense(units=[128])])"
result = compile_dsl(source)
search_space = result['search_space']
```

### 6. `validator.py` (~400 LOC)
**Purpose:** Semantic validation of AST

**Key Classes:**
- `ValidationError` - Represents a validation error
- `Validator` - Validates AST using visitor pattern

**Validations:**
- Required parameters present
- Parameter values are valid
- Layer types are supported
- Evolution strategy is valid
- Type consistency
- Architectural patterns (conv + pool, etc.)

**Features:**
- Errors and warnings
- Parameter-specific validation (filters, kernel_size, etc.)
- Layer diversity checking
- Helpful suggestions

**Example:**
```python
from morphml.core.dsl import validate_ast

errors = validate_ast(ast)
if errors:
    for error in errors:
        print(error)
```

### 7. `type_system.py` (~350 LOC)
**Purpose:** Type checking for DSL

**Key Classes:**
- `Type` - Type enum (INT, FLOAT, STRING, BOOL, etc.)
- `TypeEnvironment` - Symbol table for types
- `TypeError` - Represents a type error
- `TypeChecker` - Type checker using visitor pattern

**Features:**
- Type inference from values
- Type compatibility checking
- Parameter type requirements per layer
- Scoped type environments

**Example:**
```python
from morphml.core.dsl import check_types

type_errors = check_types(ast)
if type_errors:
    for error in type_errors:
        print(error)
```

## Two Approaches to Defining Search Spaces

### Approach 1: Pythonic Builder Pattern (Recommended)
```python
from morphml.core.dsl import SearchSpace, Layer

space = SearchSpace("my_cnn")
space.add_layers(
    Layer.conv2d(filters=[32, 64, 128], kernel_size=3),
    Layer.relu(),
    Layer.maxpool(pool_size=2),
    Layer.dense(units=[128, 256])
)
```

**Advantages:**
- Better IDE support (autocomplete, type hints)
- Easier debugging (Python stack traces)
- No context switching
- Direct Python integration

### Approach 2: Text-based DSL
```python
from morphml.core.dsl import compile_dsl

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

result = compile_dsl(source)
space = result['search_space']
```

**Advantages:**
- Declarative syntax
- External configuration files
- Dynamic generation
- Tool integration
- Human-readable serialization

## Complete Pipeline Example

```python
from morphml.core.dsl import (
    Lexer, Parser, Compiler, 
    Validator, TypeChecker,
    parse_dsl, compile_dsl
)

# Define experiment in DSL
source = """
SearchSpace(
    name="cifar10_cnn",
    layers=[
        Layer.conv2d(filters=[32, 64], kernel_size=3),
        Layer.relu(),
        Layer.dense(units=[128, 256])
    ]
)

Evolution(
    strategy="genetic",
    population_size=50,
    num_generations=100
)
"""

# Method 1: Manual pipeline
lexer = Lexer(source)
tokens = lexer.tokenize()

parser = Parser(tokens)
ast = parser.parse()

validator = Validator()
errors = validator.validate(ast)

checker = TypeChecker()
type_errors = checker.check(ast)

compiler = Compiler()
result = compiler.compile(ast)

# Method 2: Convenience functions
ast = parse_dsl(source)              # Lex + Parse
result = compile_dsl(source)         # Full pipeline

# Use the compiled result
search_space = result['search_space']
evolution_config = result['evolution']
```

## Grammar Reference

```ebnf
experiment        := search_space_def [evolution_def] [constraint_list]
search_space_def  := "SearchSpace" "(" [space_params] ")"
space_params      := "layers" "=" layer_list ["," param_list]
layer_list        := "[" layer_def ("," layer_def)* "]"
layer_def         := "Layer" "." layer_type "(" [param_list] ")"
evolution_def     := "Evolution" "(" param_list ")"

param_list        := param ("," param)*
param             := IDENTIFIER "=" value_expr
value_expr        := value | value_list
value_list        := "[" value ("," value)* "]"
value             := NUMBER | STRING | BOOLEAN | IDENTIFIER
```

## Testing

Run the example file to see the DSL in action:

```bash
python examples/dsl_example.py
```

## Status

✅ **Complete Implementation** (as per Phase 1, Component 2 specification)

- All 7 files implemented
- ~3,500 lines of code total
- Full lexer/parser/compiler pipeline
- Semantic validation
- Type checking
- Error handling with line/column tracking
- Comprehensive examples

## Integration

Both DSL approaches are fully integrated and exported from `morphml.core.dsl`:

```python
from morphml.core.dsl import (
    # Pythonic API
    Layer, LayerSpec, SearchSpace,
    
    # Text-based DSL
    Lexer, Parser, Compiler,
    parse_dsl, compile_dsl,
    validate_ast, check_types
)
```

Users can choose either approach based on their needs!
