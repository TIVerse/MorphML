# Component 2: DSL Implementation

**Duration:** Weeks 2-3  
**LOC Target:** ~3,500  
**Dependencies:** Component 1 complete

---

## 🎯 Objective

Implement a Pythonic Domain-Specific Language (DSL) that allows users to declaratively define search spaces, layer specifications, and evolution strategies. The DSL should feel natural to Python developers while providing strong validation and clear error messages.

---

## 📋 Architecture

```
morphml/core/dsl/
├── __init__.py           # Public API exports
├── syntax.py             # Token definitions and grammar (~300 LOC)
├── lexer.py              # Tokenization (~800 LOC)
├── parser.py             # AST construction (~1,200 LOC)
├── ast_nodes.py          # AST node classes (~600 LOC)
├── compiler.py           # AST to internal IR (~800 LOC)
├── validator.py          # Semantic validation (~400 LOC)
└── type_system.py        # Type checking (~350 LOC)
```

---

## 🔧 Implementation Guide

### File 1: `syntax.py` (~300 LOC)

**Purpose:** Define tokens, keywords, and grammar rules.

**Key Components:**

```python
from enum import Enum

class TokenType(Enum):
    """All token types in the DSL."""
    # Literals
    NUMBER = "NUMBER"
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"
    
    # Identifiers and keywords
    IDENTIFIER = "IDENTIFIER"
    SEARCHSPACE = "SearchSpace"
    LAYER = "Layer"
    EVOLUTION = "Evolution"
    EXPERIMENT = "Experiment"
    
    # Operators and delimiters
    ASSIGN = "="
    COMMA = ","
    DOT = "."
    COLON = ":"
    LPAREN = "("
    RPAREN = ")"
    LBRACKET = "["
    RBRACKET = "]"
    
    # Special
    EOF = "EOF"
    NEWLINE = "NEWLINE"

# Supported layer types
LAYER_TYPES = [
    "conv2d", "conv3d", "dense", "linear",
    "batch_norm", "layer_norm", "dropout",
    "max_pool", "avg_pool", "activation", "flatten"
]

# Optimizer types
OPTIMIZER_TYPES = ["adam", "sgd", "rmsprop", "adamw"]

# Activation functions
ACTIVATION_TYPES = ["relu", "elu", "gelu", "tanh", "sigmoid", "leaky_relu"]

# Evolution strategies
EVOLUTION_STRATEGIES = ["genetic", "differential_evolution", "cma_es"]
```

**Grammar (EBNF notation in docstring):**
```
experiment := search_space_def evolution_def
search_space_def := "SearchSpace" "(" layer_list ["," param_list] ")"
layer_def := "Layer" "." layer_type "(" param_list ")"
param_list := param ("," param)*
param := IDENTIFIER "=" value_list
value_list := "[" value ("," value)* "]" | value
```

---

### File 2: `lexer.py` (~800 LOC)

**Purpose:** Convert source code into token stream.

**Key Classes:**

```python
@dataclass
class Token:
    """Represents a single token."""
    type: TokenType
    value: Any
    line: int
    column: int
    
    def __repr__(self) -> str:
        return f"Token({self.type.value}, {self.value!r}, {self.line}:{self.column})"

class Lexer:
    """Tokenizes MorphML DSL source code."""
    
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
    
    def tokenize(self) -> List[Token]:
        """Main tokenization method."""
        while self.position < len(self.source):
            self._skip_whitespace()
            if self._match_comment():
                continue
            elif self._match_number():
                continue
            elif self._match_string():
                continue
            elif self._match_keyword_or_identifier():
                continue
            elif self._match_operator():
                continue
            else:
                self._error(f"Unexpected character: {self._current_char()}")
        
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens
```

**Helper Methods to Implement:**

- `_current_char() -> str | None` - Get current character
- `_peek_char(offset=1) -> str | None` - Look ahead
- `_advance() -> str | None` - Move forward, track line/column
- `_skip_whitespace()` - Skip spaces, tabs, newlines
- `_match_comment() -> bool` - Handle `#` comments
- `_match_number() -> bool` - Match integers, floats, scientific notation (123, 3.14, 1e-5)
- `_match_string() -> bool` - Match quoted strings with escape sequences
- `_match_keyword_or_identifier() -> bool` - Distinguish keywords from identifiers
- `_match_operator() -> bool` - Match operators and delimiters
- `_error(message: str)` - Raise DSLError with line/column

**Number Matching Logic:**
```python
def _match_number(self) -> bool:
    """Match integer, float, or scientific notation."""
    start_pos = self.position
    start_col = self.column
    
    # Handle negative sign
    if self._current_char() == '-':
        self._advance()
    
    # Match digits before decimal point
    if not self._current_char().isdigit():
        return False
    
    while self._current_char() and self._current_char().isdigit():
        self._advance()
    
    # Match decimal point and fractional part
    if self._current_char() == '.':
        self._advance()
        while self._current_char() and self._current_char().isdigit():
            self._advance()
    
    # Match scientific notation (e or E)
    if self._current_char() in ('e', 'E'):
        self._advance()
        if self._current_char() in ('+', '-'):
            self._advance()
        while self._current_char() and self._current_char().isdigit():
            self._advance()
    
    value_str = self.source[start_pos:self.position]
    value = float(value_str) if '.' in value_str or 'e' in value_str else int(value_str)
    
    self.tokens.append(Token(TokenType.NUMBER, value, self.line, start_col))
    return True
```

**Error Handling:**
- Track position carefully for error messages
- Example: `"Unexpected character '}' at line 5, column 12"`
- Provide suggestions for common mistakes

---

### File 3: `ast_nodes.py` (~600 LOC)

**Purpose:** Define Abstract Syntax Tree node classes.

**Base Class:**
```python
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass(frozen=True)
class ASTNode:
    """Base class for all AST nodes."""
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        """Visitor pattern support for traversal."""
        raise NotImplementedError
```

**Node Classes to Implement:**

```python
@dataclass(frozen=True)
class SearchSpaceNode(ASTNode):
    """Represents a search space definition."""
    layers: List['LayerNode']
    global_params: Dict[str, 'ParamNode']
    name: str | None = None
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_search_space(self)

@dataclass(frozen=True)
class LayerNode(ASTNode):
    """Represents a layer specification."""
    layer_type: str  # 'conv2d', 'dense', etc.
    params: Dict[str, 'ParamNode']
    
    def __post_init__(self):
        # Validate layer type
        if self.layer_type not in LAYER_TYPES:
            raise ValidationError(f"Unknown layer type: {self.layer_type}")

@dataclass(frozen=True)
class ParamNode(ASTNode):
    """Represents a hyperparameter specification."""
    name: str
    values: List[Any]  # Possible values
    param_type: str  # 'categorical', 'integer', 'float'
    
    def __post_init__(self):
        # Infer type if not specified
        if self.param_type is None:
            self.param_type = self._infer_type()

@dataclass(frozen=True)
class EvolutionNode(ASTNode):
    """Represents evolution configuration."""
    strategy: str  # 'genetic', 'differential_evolution', etc.
    params: Dict[str, Any]
    
    def __post_init__(self):
        if self.strategy not in EVOLUTION_STRATEGIES:
            raise ValidationError(f"Unknown strategy: {self.strategy}")

@dataclass(frozen=True)
class ExperimentNode(ASTNode):
    """Root node representing complete experiment."""
    search_space: SearchSpaceNode
    evolution: EvolutionNode
    objectives: List[str]
    constraints: List[Any] = field(default_factory=list)
```

**Implementation Notes:**
- Use `@dataclass(frozen=True)` for immutability
- Implement `__post_init__` for validation
- Add helpful `__repr__` for debugging
- Use type hints extensively

---

### File 4: `parser.py` (~1,200 LOC)

**Purpose:** Parse token stream into AST using recursive descent.

**Key Class:**

```python
class Parser:
    """Recursive descent parser for MorphML DSL."""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
        self.current_token = tokens[0] if tokens else None
    
    def parse(self) -> ExperimentNode:
        """Parse complete experiment definition."""
        search_space = self.parse_search_space()
        evolution = self.parse_evolution()
        return ExperimentNode(search_space, evolution, [])
    
    def parse_search_space(self) -> SearchSpaceNode:
        """Parse: SearchSpace(layers=[...], ...)"""
        self._expect(TokenType.SEARCHSPACE)
        self._expect(TokenType.LPAREN)
        
        layers = []
        params = {}
        
        # Parse keyword arguments
        while self._current_token.type != TokenType.RPAREN:
            param_name = self._expect(TokenType.IDENTIFIER).value
            self._expect(TokenType.ASSIGN)
            
            if param_name == "layers":
                layers = self._parse_layer_list()
            else:
                params[param_name] = self._parse_value()
            
            if self._current_token.type == TokenType.COMMA:
                self._advance()
        
        self._expect(TokenType.RPAREN)
        return SearchSpaceNode(layers, params)
    
    def _parse_layer_list(self) -> List[LayerNode]:
        """Parse: [Layer.conv2d(...), Layer.dense(...)]"""
        self._expect(TokenType.LBRACKET)
        layers = []
        
        while self._current_token.type != TokenType.RBRACKET:
            layers.append(self.parse_layer())
            if self._current_token.type == TokenType.COMMA:
                self._advance()
        
        self._expect(TokenType.RBRACKET)
        return layers
    
    def parse_layer(self) -> LayerNode:
        """Parse: Layer.conv2d(filters=[32, 64], kernel_size=3)"""
        self._expect(TokenType.LAYER)
        self._expect(TokenType.DOT)
        
        layer_type = self._expect(TokenType.IDENTIFIER).value
        self._expect(TokenType.LPAREN)
        
        params = self._parse_param_list()
        
        self._expect(TokenType.RPAREN)
        return LayerNode(layer_type, params)
    
    def _parse_param_list(self) -> Dict[str, ParamNode]:
        """Parse: filters=[32, 64], kernel_size=3"""
        params = {}
        
        while self._current_token.type == TokenType.IDENTIFIER:
            name = self._advance().value
            self._expect(TokenType.ASSIGN)
            
            # Parse value or value list
            if self._current_token.type == TokenType.LBRACKET:
                values = self._parse_value_list()
            else:
                values = [self._parse_value()]
            
            params[name] = ParamNode(name, values, None)
            
            if self._current_token.type == TokenType.COMMA:
                self._advance()
        
        return params
```

**Helper Methods:**

- `_current_token() -> Token` - Get current token
- `_peek_token(offset=1) -> Token` - Look ahead
- `_advance() -> Token` - Move to next token
- `_expect(token_type: TokenType) -> Token` - Consume expected token or error
- `_match(*token_types: TokenType) -> bool` - Check if current token matches
- `_error(message: str)` - Raise DSLError with context

**Error Handling:**
```python
def _expect(self, token_type: TokenType) -> Token:
    """Expect specific token type, raise error if not found."""
    if self.current_token.type != token_type:
        raise DSLError(
            f"Expected {token_type.value}, got {self.current_token.type.value}",
            line=self.current_token.line,
            column=self.current_token.column
        )
    token = self.current_token
    self._advance()
    return token
```

---

### File 5: `compiler.py` (~800 LOC)

**Purpose:** Compile AST into executable internal representation.

```python
class Compiler:
    """Compiles AST into internal representation."""
    
    def compile(self, ast: ExperimentNode) -> Dict[str, Any]:
        """
        Compile AST to internal representation.
        
        Returns:
            Dictionary with 'search_space', 'evolution', 'objectives'
        """
        search_space = self._compile_search_space(ast.search_space)
        evolution = self._compile_evolution(ast.evolution)
        
        return {
            'search_space': search_space,
            'evolution': evolution,
            'objectives': ast.objectives
        }
    
    def _compile_search_space(self, node: SearchSpaceNode) -> 'SearchSpace':
        """Convert SearchSpaceNode to SearchSpace object."""
        from morphml.core.search import SearchSpace, Layer, Parameter
        
        layers = [self._compile_layer(layer_node) for layer_node in node.layers]
        global_params = {
            name: self._compile_param(param_node)
            for name, param_node in node.global_params.items()
        }
        
        return SearchSpace(layers=layers, global_params=global_params)
    
    def _compile_layer(self, node: LayerNode) -> 'Layer':
        """Convert LayerNode to Layer object."""
        params = {
            name: self._compile_param(param_node)
            for name, param_node in node.params.items()
        }
        return Layer(layer_type=node.layer_type, params=params)
    
    def _compile_param(self, node: ParamNode) -> 'Parameter':
        """Convert ParamNode to Parameter object."""
        # Determine parameter type
        if node.param_type == 'categorical' or isinstance(node.values[0], str):
            return CategoricalParameter(node.name, node.values)
        elif all(isinstance(v, int) for v in node.values):
            return IntegerParameter(node.name, min(node.values), max(node.values))
        elif all(isinstance(v, float) for v in node.values):
            return FloatParameter(node.name, min(node.values), max(node.values))
        else:
            raise ValidationError(f"Cannot infer parameter type for {node.name}")
```

**Key Responsibilities:**
- Type inference from values
- Default value assignment
- Reference resolution
- Validation during compilation

---

### File 6: `validator.py` (~400 LOC)

**Purpose:** Semantic validation of AST.

```python
class Validator:
    """Validates AST for semantic errors."""
    
    def validate(self, ast: ExperimentNode) -> List[str]:
        """
        Validate AST and return list of errors.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        errors.extend(self._validate_search_space(ast.search_space))
        errors.extend(self._validate_evolution(ast.evolution))
        errors.extend(self._validate_objectives(ast.objectives))
        return errors
    
    def _validate_search_space(self, node: SearchSpaceNode) -> List[str]:
        """Validate search space definition."""
        errors = []
        
        # Check at least one layer
        if not node.layers:
            errors.append("SearchSpace must contain at least one layer")
        
        # Validate each layer
        for layer_node in node.layers:
            errors.extend(self._validate_layer(layer_node))
        
        return errors
    
    def _validate_layer(self, node: LayerNode) -> List[str]:
        """Validate layer specification."""
        errors = []
        
        # Check layer type is supported
        if node.layer_type not in LAYER_TYPES:
            errors.append(f"Unsupported layer type: {node.layer_type}")
        
        # Validate required parameters for layer type
        required_params = self._get_required_params(node.layer_type)
        for param in required_params:
            if param not in node.params:
                errors.append(f"Missing required parameter '{param}' for {node.layer_type}")
        
        # Validate parameter values
        for name, param_node in node.params.items():
            if not param_node.values:
                errors.append(f"Parameter '{name}' has no values")
        
        return errors
```

**Validation Checks:**
- Required parameters present
- Value ranges are valid
- No circular dependencies
- Type consistency
- Constraint satisfaction

---

### File 7: `type_system.py` (~350 LOC)

**Purpose:** Type checking for DSL.

```python
class Type(Enum):
    """DSL type system."""
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    LIST = "list"
    ANY = "any"

class TypeChecker:
    """Type checker for DSL."""
    
    def check(self, ast: ExperimentNode) -> List[str]:
        """
        Type check AST.
        
        Returns:
            List of type errors
        """
        errors = []
        errors.extend(self._check_search_space(ast.search_space))
        return errors
    
    def _infer_type(self, value: Any) -> Type:
        """Infer type from Python value."""
        if isinstance(value, bool):
            return Type.BOOL
        elif isinstance(value, int):
            return Type.INT
        elif isinstance(value, float):
            return Type.FLOAT
        elif isinstance(value, str):
            return Type.STRING
        elif isinstance(value, list):
            return Type.LIST
        else:
            return Type.ANY
```

---

### File 8: `__init__.py`

**Public API exports:**

```python
"""MorphML DSL - Pythonic interface for defining search spaces."""

from morphml.core.dsl.lexer import Lexer, Token
from morphml.core.dsl.parser import Parser
from morphml.core.dsl.compiler import Compiler
from morphml.core.dsl.validator import Validator
from morphml.core.dsl.ast_nodes import (
    ASTNode,
    SearchSpaceNode,
    LayerNode,
    ParamNode,
    EvolutionNode,
    ExperimentNode,
)

__all__ = [
    "Lexer",
    "Token",
    "Parser",
    "Compiler",
    "Validator",
    "ASTNode",
    "SearchSpaceNode",
    "LayerNode",
    "ParamNode",
    "EvolutionNode",
    "ExperimentNode",
]
```

---

## 🧪 Testing Strategy

**`tests/unit/test_dsl/test_lexer.py`** (~200 LOC):

```python
def test_lexer_tokenizes_integers():
    lexer = Lexer("42")
    tokens = lexer.tokenize()
    assert len(tokens) == 2  # NUMBER + EOF
    assert tokens[0].type == TokenType.NUMBER
    assert tokens[0].value == 42

def test_lexer_tokenizes_floats():
    lexer = Lexer("3.14")
    tokens = lexer.tokenize()
    assert tokens[0].value == 3.14

def test_lexer_tokenizes_scientific_notation():
    lexer = Lexer("1.5e-3")
    tokens = lexer.tokenize()
    assert abs(tokens[0].value - 0.0015) < 1e-10

def test_lexer_tracks_line_numbers():
    source = "SearchSpace\n(layers=[Layer.conv2d()])"
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    assert tokens[0].line == 1
    assert tokens[1].line == 2

def test_lexer_raises_error_on_invalid_character():
    with pytest.raises(DSLError, match="Unexpected character"):
        Lexer("@invalid").tokenize()
```

**`tests/unit/test_dsl/test_parser.py`** (~250 LOC):

```python
def test_parser_builds_layer_node():
    tokens = [
        Token(TokenType.LAYER, "Layer", 1, 1),
        Token(TokenType.DOT, ".", 1, 6),
        Token(TokenType.IDENTIFIER, "conv2d", 1, 7),
        Token(TokenType.LPAREN, "(", 1, 13),
        Token(TokenType.RPAREN, ")", 1, 14),
        Token(TokenType.EOF, None, 1, 15),
    ]
    parser = Parser(tokens)
    layer = parser.parse_layer()
    assert layer.layer_type == "conv2d"
    assert layer.params == {}

def test_parser_handles_parameter_lists():
    # Test parsing Layer.conv2d(filters=[32, 64])
    pass

def test_parser_raises_error_on_missing_paren():
    tokens = [Token(TokenType.LAYER, "Layer", 1, 1), Token(TokenType.EOF, None, 1, 6)]
    parser = Parser(tokens)
    with pytest.raises(DSLError, match="Expected DOT"):
        parser.parse_layer()
```

---

## ✅ Deliverables Checklist

- [ ] All 7 files implemented with proper structure
- [ ] Type hints on all public methods
- [ ] Docstrings with examples
- [ ] Error messages include line/column numbers
- [ ] Unit tests >85% coverage
- [ ] Integration test: parse sample search space end-to-end
- [ ] Code passes black, ruff, mypy

---

**Next:** Proceed to `03_graph_system.md`
