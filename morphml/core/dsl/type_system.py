"""Type system for MorphML DSL.

Provides type checking and type inference for DSL constructs.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from morphml.core.dsl.ast_nodes import (
    ASTVisitor,
    ExperimentNode,
    LayerNode,
    ParamNode,
    SearchSpaceNode,
)
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class Type(Enum):
    """Type system for DSL values."""

    # Primitive types
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"

    # Collection types
    LIST = "list"
    DICT = "dict"

    # Special types
    ANY = "any"
    NONE = "none"
    LAYER = "layer"
    PARAM = "param"

    def __repr__(self) -> str:
        """String representation."""
        return f"Type.{self.name}"


class TypeEnvironment:
    """
    Type environment for type checking.

    Maintains a symbol table mapping names to types.
    """

    def __init__(self, parent: Optional["TypeEnvironment"] = None):
        """
        Initialize type environment.

        Args:
            parent: Parent environment for nested scopes
        """
        self.parent = parent
        self.symbols: Dict[str, Type] = {}

    def define(self, name: str, type_: Type) -> None:
        """Define a symbol with its type."""
        self.symbols[name] = type_

    def lookup(self, name: str) -> Optional[Type]:
        """Look up symbol type."""
        if name in self.symbols:
            return self.symbols[name]
        elif self.parent:
            return self.parent.lookup(name)
        return None

    def __repr__(self) -> str:
        """String representation."""
        return f"TypeEnvironment({self.symbols})"


class TypeError:
    """Represents a type error."""

    def __init__(
        self,
        message: str,
        node: str = "Unknown",
        expected: Optional[Type] = None,
        actual: Optional[Type] = None,
    ):
        """
        Initialize type error.

        Args:
            message: Error message
            node: Node type where error occurred
            expected: Expected type
            actual: Actual type
        """
        self.message = message
        self.node = node
        self.expected = expected
        self.actual = actual

    def __str__(self) -> str:
        """String representation."""
        if self.expected and self.actual:
            return (
                f"[{self.node}] {self.message}\n"
                f"  Expected: {self.expected}\n"
                f"  Actual:   {self.actual}"
            )
        return f"[{self.node}] {self.message}"


class TypeChecker(ASTVisitor):
    """
    Type checker for MorphML DSL.

    Performs static type checking on the AST to catch type errors
    before compilation.

    Example:
        >>> checker = TypeChecker()
        >>> errors = checker.check(ast)
        >>> if errors:
        ...     for error in errors:
        ...         print(error)
    """

    def __init__(self) -> None:
        """Initialize type checker."""
        self.errors: List[TypeError] = []
        self.env = TypeEnvironment()
        self.current_layer_type: Optional[str] = None

        # Parameter type requirements for each layer type
        self.layer_param_types: Dict[str, Dict[str, Type]] = {
            "conv2d": {
                "filters": Type.INT,
                "kernel_size": Type.INT,
                "strides": Type.INT,
                "padding": Type.STRING,
            },
            "dense": {"units": Type.INT, "activation": Type.STRING},
            "dropout": {"rate": Type.FLOAT},
            "max_pool": {"pool_size": Type.INT},
            "avg_pool": {"pool_size": Type.INT},
        }

    def check(self, ast: ExperimentNode) -> List[TypeError]:
        """
        Type check complete experiment AST.

        Args:
            ast: ExperimentNode to type check

        Returns:
            List of type errors (empty if type-safe)
        """
        logger.info("Starting type checking")

        # Reset state
        self.errors.clear()
        self.env = TypeEnvironment()

        # Type check via visitor pattern
        ast.accept(self)

        if self.errors:
            logger.warning(f"Type checking found {len(self.errors)} errors")
        else:
            logger.info("Type checking successful")

        return self.errors

    def visit_search_space(self, node: SearchSpaceNode) -> None:
        """Type check search space."""
        # Create new scope for search space
        old_env = self.env
        self.env = TypeEnvironment(parent=old_env)

        # Type check global parameters
        for param_name, param_node in node.global_params.items():
            param_type = self._infer_param_type(param_node)
            self.env.define(param_name, param_type)

        # Type check layers
        super().visit_search_space(node)

        # Restore environment
        self.env = old_env

    def visit_layer(self, node: LayerNode) -> None:
        """Type check layer."""
        self.current_layer_type = node.layer_type

        # Get expected parameter types for this layer
        expected_types = self.layer_param_types.get(node.layer_type, {})

        # Check each parameter
        for param_name, param_node in node.params.items():
            # Infer actual type
            actual_type = self._infer_param_type(param_node)

            # Check against expected type if known
            if param_name in expected_types:
                expected_type = expected_types[param_name]

                if not self._is_compatible(actual_type, expected_type):
                    self.errors.append(
                        TypeError(
                            f"Type mismatch for parameter '{param_name}' in layer '{node.layer_type}'",
                            node="Layer",
                            expected=expected_type,
                            actual=actual_type,
                        )
                    )

        super().visit_layer(node)

        self.current_layer_type = None

    def visit_param(self, node: ParamNode) -> None:
        """Type check parameter."""
        # Check type consistency within parameter values
        if len(node.values) > 1:
            types = [self._infer_type(v) for v in node.values]
            if len(set(types)) > 1:
                # Allow int/float mixing
                if not (set(types) <= {Type.INT, Type.FLOAT}):
                    self.errors.append(
                        TypeError(
                            f"Inconsistent types in parameter '{node.name}': {types}",
                            node="Parameter",
                        )
                    )

    def _infer_type(self, value: Any) -> Type:
        """
        Infer type from Python value.

        Args:
            value: Python value

        Returns:
            Type enum
        """
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
        elif isinstance(value, dict):
            return Type.DICT
        elif value is None:
            return Type.NONE
        else:
            return Type.ANY

    def _infer_param_type(self, param_node: ParamNode) -> Type:
        """
        Infer type for a parameter node.

        Args:
            param_node: Parameter node

        Returns:
            Inferred type
        """
        if not param_node.values:
            return Type.NONE

        # Infer from first value
        first_type = self._infer_type(param_node.values[0])

        # Check if all values have same type
        all_same = all(self._infer_type(v) == first_type for v in param_node.values)

        if all_same:
            return first_type

        # Mixed types - check for numeric
        types = [self._infer_type(v) for v in param_node.values]
        if set(types) <= {Type.INT, Type.FLOAT}:
            return Type.FLOAT

        return Type.ANY

    def _is_compatible(self, actual: Type, expected: Type) -> bool:
        """
        Check if actual type is compatible with expected type.

        Args:
            actual: Actual type
            expected: Expected type

        Returns:
            True if compatible
        """
        # Same type is compatible
        if actual == expected:
            return True

        # ANY is compatible with everything
        if actual == Type.ANY or expected == Type.ANY:
            return True

        # INT is compatible with FLOAT
        if actual == Type.INT and expected == Type.FLOAT:
            return True

        return False


def check_types(ast: ExperimentNode) -> List[TypeError]:
    """
    Convenience function to type check AST.

    Args:
        ast: ExperimentNode to check

    Returns:
        List of type errors
    """
    checker = TypeChecker()
    return checker.check(ast)


def infer_value_type(value: Any) -> str:
    """
    Infer type name from Python value.

    Args:
        value: Python value

    Returns:
        Type name as string
    """
    if isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, str):
        return "string"
    elif isinstance(value, list):
        return "list"
    elif isinstance(value, dict):
        return "dict"
    elif value is None:
        return "none"
    else:
        return "any"
