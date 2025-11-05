"""Lexical analyzer for MorphML DSL.

Converts source code into a stream of tokens for parsing.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from dataclasses import dataclass
from typing import Any, List, Optional

from morphml.core.dsl.syntax import KEYWORDS, OPERATORS, TokenType
from morphml.exceptions import DSLError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Token:
    """Represents a single token in the source code."""

    type: TokenType
    value: Any
    line: int
    column: int

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"


class Lexer:
    """
    Tokenizes MorphML DSL source code.

    Converts a string of source code into a stream of tokens that can be
    parsed into an abstract syntax tree.

    Example:
        >>> source = 'Layer.conv2d(filters=[32, 64])'
        >>> lexer = Lexer(source)
        >>> tokens = lexer.tokenize()
        >>> for token in tokens:
        ...     print(token)
    """

    def __init__(self, source: str):
        """
        Initialize lexer with source code.

        Args:
            source: Source code string to tokenize
        """
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []

    def tokenize(self) -> List[Token]:
        """
        Main tokenization method.

        Returns:
            List of tokens representing the source code

        Raises:
            DSLError: If invalid characters or syntax is encountered
        """
        while self.position < len(self.source):
            # Skip whitespace
            if self._skip_whitespace():
                continue

            # Skip comments
            if self._skip_comment():
                continue

            # Match tokens
            if self._match_number():
                continue
            elif self._match_string():
                continue
            elif self._match_keyword_or_identifier():
                continue
            elif self._match_operator():
                continue
            else:
                char = self._current_char()
                self._error(f"Unexpected character: '{char}' (ASCII {ord(char)})")

        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        logger.debug(f"Tokenized {len(self.tokens)} tokens from {self.line} lines")
        return self.tokens

    def _current_char(self) -> Optional[str]:
        """Get current character without advancing."""
        if self.position >= len(self.source):
            return None
        return self.source[self.position]

    def _peek_char(self, offset: int = 1) -> Optional[str]:
        """Look ahead at character at offset from current position."""
        pos = self.position + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]

    def _advance(self) -> Optional[str]:
        """Move to next character and return current one."""
        if self.position >= len(self.source):
            return None

        char = self.source[self.position]
        self.position += 1

        # Track line and column for error messages
        if char == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1

        return char

    def _skip_whitespace(self) -> bool:
        """Skip whitespace characters (space, tab, newline)."""
        char = self._current_char()
        if char and char in " \t\n\r":
            self._advance()
            return True
        return False

    def _skip_comment(self) -> bool:
        """Skip comments starting with #."""
        if self._current_char() == "#":
            # Skip until end of line
            while self._current_char() and self._current_char() != "\n":
                self._advance()
            return True
        return False

    def _match_number(self) -> bool:
        """
        Match integer, float, or scientific notation.

        Formats supported:
        - Integer: 42, -123
        - Float: 3.14, -0.5
        - Scientific: 1.5e-3, 2E+10
        """
        start_pos = self.position
        start_col = self.column

        # Handle negative sign
        if self._current_char() == "-":
            # Check if next char is digit
            if not (self._peek_char() and self._peek_char().isdigit()):
                return False
            self._advance()

        # Must start with digit
        if not (self._current_char() and self._current_char().isdigit()):
            return False

        # Match integer part
        while self._current_char() and self._current_char().isdigit():
            self._advance()

        # Match decimal point and fractional part
        if self._current_char() == ".":
            # Peek ahead to ensure it's a decimal, not a method call
            if self._peek_char() and self._peek_char().isdigit():
                self._advance()  # consume '.'
                while self._current_char() and self._current_char().isdigit():
                    self._advance()

        # Match scientific notation
        if self._current_char() and self._current_char() in ("e", "E"):
            self._advance()
            # Optional sign
            if self._current_char() and self._current_char() in ("+", "-"):
                self._advance()
            # Exponent digits
            if not (self._current_char() and self._current_char().isdigit()):
                self._error("Invalid scientific notation: expected digits after exponent")
            while self._current_char() and self._current_char().isdigit():
                self._advance()

        # Extract value and convert
        value_str = self.source[start_pos : self.position]
        try:
            if "." in value_str or "e" in value_str or "E" in value_str:
                value = float(value_str)
            else:
                value = int(value_str)
        except ValueError:
            self._error(f"Invalid number format: {value_str}")

        self.tokens.append(Token(TokenType.NUMBER, value, self.line, start_col))
        return True

    def _match_string(self) -> bool:
        """
        Match quoted strings with escape sequences.

        Supports both single and double quotes.
        Handles escape sequences: \\n, \\t, \\', \\", \\\\
        """
        start_col = self.column
        quote_char = self._current_char()

        if quote_char not in ('"', "'"):
            return False

        self._advance()  # consume opening quote
        chars = []

        while self._current_char() and self._current_char() != quote_char:
            if self._current_char() == "\\":
                # Handle escape sequences
                self._advance()
                escape_char = self._current_char()
                if escape_char == "n":
                    chars.append("\n")
                elif escape_char == "t":
                    chars.append("\t")
                elif escape_char == "r":
                    chars.append("\r")
                elif escape_char == "\\":
                    chars.append("\\")
                elif escape_char == quote_char:
                    chars.append(quote_char)
                else:
                    chars.append(escape_char)
                self._advance()
            else:
                chars.append(self._current_char())
                self._advance()

        if self._current_char() != quote_char:
            self._error(f"Unterminated string starting at line {self.line}, column {start_col}")

        self._advance()  # consume closing quote

        value = "".join(chars)
        self.tokens.append(Token(TokenType.STRING, value, self.line, start_col))
        return True

    def _match_keyword_or_identifier(self) -> bool:
        """
        Match keywords or identifiers.

        Identifiers: [a-zA-Z_][a-zA-Z0-9_]*
        Keywords: SearchSpace, Layer, Evolution, etc.
        """
        start_col = self.column
        char = self._current_char()

        # Must start with letter or underscore
        if not (char and (char.isalpha() or char == "_")):
            return False

        # Collect identifier characters
        chars = []
        while self._current_char() and (
            self._current_char().isalnum() or self._current_char() == "_"
        ):
            chars.append(self._current_char())
            self._advance()

        identifier = "".join(chars)

        # Check if it's a keyword
        if identifier in KEYWORDS:
            token_type = KEYWORDS[identifier]
            # For boolean keywords, store the boolean value
            if token_type == TokenType.BOOLEAN:
                value = identifier in ("True", "true")
            else:
                value = identifier
            self.tokens.append(Token(token_type, value, self.line, start_col))
        else:
            # Regular identifier
            self.tokens.append(Token(TokenType.IDENTIFIER, identifier, self.line, start_col))

        return True

    def _match_operator(self) -> bool:
        """Match operators and delimiters."""
        start_col = self.column
        char = self._current_char()

        if char in OPERATORS:
            token_type = OPERATORS[char]
            self._advance()
            self.tokens.append(Token(token_type, char, self.line, start_col))
            return True

        return False

    def _error(self, message: str) -> None:
        """
        Raise DSLError with line and column information.

        Args:
            message: Error message

        Raises:
            DSLError: With formatted error message
        """
        raise DSLError(
            f"{message}\n"
            f"  at line {self.line}, column {self.column}\n"
            f"  {self._get_error_context()}",
            line=self.line,
            column=self.column,
        )

    def _get_error_context(self) -> str:
        """Get source code context around error for display."""
        # Find start and end of current line
        line_start = self.position
        while line_start > 0 and self.source[line_start - 1] != "\n":
            line_start -= 1

        line_end = self.position
        while line_end < len(self.source) and self.source[line_end] != "\n":
            line_end += 1

        line_content = self.source[line_start:line_end]
        pointer = " " * (self.column - 1) + "^"

        return f"{line_content}\n  {pointer}"
