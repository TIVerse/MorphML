"""Recursive descent parser for MorphML DSL.

Parses token stream into Abstract Syntax Tree (AST).

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Any, Dict, List, Optional

from morphml.core.dsl.ast_nodes import (
    ConstraintNode,
    EvolutionNode,
    ExperimentNode,
    LayerNode,
    ParamNode,
    SearchSpaceNode,
)
from morphml.core.dsl.lexer import Token
from morphml.core.dsl.syntax import TokenType
from morphml.exceptions import DSLError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class Parser:
    """
    Recursive descent parser for MorphML DSL.

    Converts a stream of tokens into an Abstract Syntax Tree (AST)
    that can be compiled into executable search space definitions.

    Grammar:
        experiment := search_space_def [evolution_def] [constraint_list]
        search_space_def := "SearchSpace" "(" param_list ")"
        layer_def := "Layer" "." layer_type "(" param_list ")"
        evolution_def := "Evolution" "(" param_list ")"

    Example:
        >>> tokens = lexer.tokenize()
        >>> parser = Parser(tokens)
        >>> ast = parser.parse()
    """

    def __init__(self, tokens: List[Token]):
        """
        Initialize parser with token stream.

        Args:
            tokens: List of tokens from lexer
        """
        self.tokens = tokens
        self.position = 0
        self.current_token = tokens[0] if tokens else None

    def parse(self) -> ExperimentNode:
        """
        Parse complete experiment definition.

        Returns:
            ExperimentNode representing the entire experiment

        Raises:
            DSLError: If parsing fails
        """
        logger.debug("Starting parse")

        # Parse required search space
        search_space = self.parse_search_space()

        # Parse optional evolution config
        evolution = None
        if self._match(TokenType.EVOLUTION):
            evolution = self.parse_evolution()

        # Parse optional constraints
        constraints = []
        while self._match(TokenType.CONSTRAINT):
            constraints.append(self.parse_constraint())

        # Expect EOF
        self._expect(TokenType.EOF)

        logger.debug(f"Parse complete: {len(search_space.layers)} layers")

        return ExperimentNode(
            search_space=search_space, evolution=evolution, constraints=constraints
        )

    def parse_search_space(self) -> SearchSpaceNode:
        """
        Parse search space definition.

        Grammar: "SearchSpace" "(" param_list ")"

        Returns:
            SearchSpaceNode
        """
        self._expect(TokenType.SEARCHSPACE)
        self._expect(TokenType.LPAREN)

        layers: List[LayerNode] = []
        global_params: Dict[str, ParamNode] = {}
        name: Optional[str] = None

        # Parse keyword arguments
        while not self._match(TokenType.RPAREN):
            param_name_token = self._expect(TokenType.IDENTIFIER)
            param_name = param_name_token.value
            self._expect(TokenType.ASSIGN)

            if param_name == "layers":
                layers = self._parse_layer_list()
            elif param_name == "name":
                name_token = self._expect(TokenType.STRING)
                name = name_token.value
            else:
                # Global parameter
                param_value = self._parse_value_expr()
                global_params[param_name] = ParamNode(
                    name=param_name, values=param_value if isinstance(param_value, list) else [param_value]
                )

            # Optional comma
            if self._match(TokenType.COMMA):
                self._advance()

        self._expect(TokenType.RPAREN)

        return SearchSpaceNode(layers=layers, global_params=global_params, name=name)

    def _parse_layer_list(self) -> List[LayerNode]:
        """
        Parse list of layers.

        Grammar: "[" layer_def ("," layer_def)* "]"

        Returns:
            List of LayerNode
        """
        self._expect(TokenType.LBRACKET)
        layers = []

        while not self._match(TokenType.RBRACKET):
            layers.append(self.parse_layer())

            # Optional comma
            if self._match(TokenType.COMMA):
                self._advance()
            elif not self._match(TokenType.RBRACKET):
                self._error("Expected ',' or ']' in layer list")

        self._expect(TokenType.RBRACKET)
        return layers

    def parse_layer(self) -> LayerNode:
        """
        Parse layer definition.

        Grammar: "Layer" "." layer_type "(" param_list ")"

        Returns:
            LayerNode
        """
        self._expect(TokenType.LAYER)
        self._expect(TokenType.DOT)

        layer_type_token = self._expect(TokenType.IDENTIFIER)
        layer_type = layer_type_token.value

        self._expect(TokenType.LPAREN)

        # Parse parameters
        params = self._parse_param_list()

        self._expect(TokenType.RPAREN)

        return LayerNode(layer_type=layer_type, params=params)

    def _parse_param_list(self) -> Dict[str, ParamNode]:
        """
        Parse parameter list.

        Grammar: param ("," param)*
        param := IDENTIFIER "=" value_expr

        Returns:
            Dictionary mapping parameter names to ParamNode
        """
        params: Dict[str, ParamNode] = {}

        while self._match(TokenType.IDENTIFIER):
            name_token = self._advance()
            name = name_token.value

            self._expect(TokenType.ASSIGN)

            value_expr = self._parse_value_expr()

            # Create ParamNode
            if isinstance(value_expr, list):
                params[name] = ParamNode(name=name, values=value_expr)
            else:
                params[name] = ParamNode(name=name, values=[value_expr])

            # Optional comma
            if self._match(TokenType.COMMA):
                self._advance()
            else:
                break

        return params

    def _parse_value_expr(self) -> Any:
        """
        Parse value expression (single value or list).

        Grammar: value | "[" value ("," value)* "]"

        Returns:
            Single value or list of values
        """
        if self._match(TokenType.LBRACKET):
            return self._parse_value_list()
        else:
            return self._parse_value()

    def _parse_value_list(self) -> List[Any]:
        """
        Parse list of values.

        Grammar: "[" value ("," value)* "]"

        Returns:
            List of values
        """
        self._expect(TokenType.LBRACKET)
        values = []

        while not self._match(TokenType.RBRACKET):
            values.append(self._parse_value())

            # Optional comma
            if self._match(TokenType.COMMA):
                self._advance()
            elif not self._match(TokenType.RBRACKET):
                self._error("Expected ',' or ']' in value list")

        self._expect(TokenType.RBRACKET)
        return values

    def _parse_value(self) -> Any:
        """
        Parse a single value.

        Grammar: NUMBER | STRING | BOOLEAN | IDENTIFIER

        Returns:
            Value (int, float, str, bool)
        """
        if self._match(TokenType.NUMBER):
            token = self._advance()
            return token.value
        elif self._match(TokenType.STRING):
            token = self._advance()
            return token.value
        elif self._match(TokenType.BOOLEAN):
            token = self._advance()
            return token.value
        elif self._match(TokenType.IDENTIFIER):
            token = self._advance()
            return token.value
        else:
            self._error(
                f"Expected value (number, string, boolean, or identifier), "
                f"got {self.current_token.type.name}"
            )

    def parse_evolution(self) -> EvolutionNode:
        """
        Parse evolution configuration.

        Grammar: "Evolution" "(" param_list ")"

        Returns:
            EvolutionNode
        """
        self._expect(TokenType.EVOLUTION)
        self._expect(TokenType.LPAREN)

        params: Dict[str, Any] = {}
        strategy: Optional[str] = None

        # Parse parameters
        while self._match(TokenType.IDENTIFIER):
            name_token = self._advance()
            name = name_token.value

            self._expect(TokenType.ASSIGN)

            value = self._parse_value_expr()

            if name == "strategy":
                strategy = value if isinstance(value, str) else value[0]
            else:
                params[name] = value

            # Optional comma
            if self._match(TokenType.COMMA):
                self._advance()

        self._expect(TokenType.RPAREN)

        if not strategy:
            self._error("Evolution must specify 'strategy' parameter")

        return EvolutionNode(strategy=strategy, params=params)

    def parse_constraint(self) -> ConstraintNode:
        """
        Parse constraint definition.

        Grammar: "Constraint" "(" param_list ")"

        Returns:
            ConstraintNode
        """
        self._expect(TokenType.CONSTRAINT)
        self._expect(TokenType.LPAREN)

        params: Dict[str, Any] = {}
        constraint_type: Optional[str] = None

        # Parse parameters
        while self._match(TokenType.IDENTIFIER):
            name_token = self._advance()
            name = name_token.value

            self._expect(TokenType.ASSIGN)

            value = self._parse_value_expr()

            if name == "type":
                constraint_type = value if isinstance(value, str) else value[0]
            else:
                params[name] = value

            # Optional comma
            if self._match(TokenType.COMMA):
                self._advance()

        self._expect(TokenType.RPAREN)

        if not constraint_type:
            self._error("Constraint must specify 'type' parameter")

        return ConstraintNode(constraint_type=constraint_type, params=params)

    # Helper methods for token manipulation

    def _current_token(self) -> Token:
        """Get current token."""
        return self.current_token

    def _peek_token(self, offset: int = 1) -> Optional[Token]:
        """Look ahead at token."""
        pos = self.position + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None

    def _advance(self) -> Token:
        """Move to next token and return current."""
        token = self.current_token
        self.position += 1
        if self.position < len(self.tokens):
            self.current_token = self.tokens[self.position]
        else:
            self.current_token = None
        return token

    def _match(self, *token_types: TokenType) -> bool:
        """Check if current token matches any of the given types."""
        if self.current_token is None:
            return False
        return self.current_token.type in token_types

    def _expect(self, token_type: TokenType) -> Token:
        """
        Expect specific token type, raise error if not found.

        Args:
            token_type: Expected token type

        Returns:
            Current token

        Raises:
            DSLError: If current token doesn't match expected type
        """
        if not self._match(token_type):
            self._error(
                f"Expected {token_type.name}, got {self.current_token.type.name if self.current_token else 'EOF'}"
            )

        return self._advance()

    def _error(self, message: str) -> None:
        """
        Raise DSLError with current position information.

        Args:
            message: Error message

        Raises:
            DSLError: With formatted error message
        """
        if self.current_token:
            raise DSLError(
                f"Parse error: {message}\n"
                f"  at line {self.current_token.line}, column {self.current_token.column}\n"
                f"  near token: {self.current_token}",
                line=self.current_token.line,
                column=self.current_token.column,
            )
        else:
            raise DSLError(f"Parse error: {message}\n  at end of input")


def parse_dsl(source: str) -> ExperimentNode:
    """
    Convenience function to lex and parse DSL source.

    Args:
        source: DSL source code

    Returns:
        ExperimentNode AST

    Example:
        >>> source = '''
        ... SearchSpace(
        ...     layers=[Layer.conv2d(filters=[32, 64])]
        ... )
        ... '''
        >>> ast = parse_dsl(source)
    """
    from morphml.core.dsl.lexer import Lexer

    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()
