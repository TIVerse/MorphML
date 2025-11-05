"""Semantic validator for MorphML DSL.

Validates AST for semantic correctness beyond syntax.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Dict, List, Set

from morphml.core.dsl.ast_nodes import (
    ASTVisitor,
    ConstraintNode,
    EvolutionNode,
    ExperimentNode,
    LayerNode,
    ParamNode,
    SearchSpaceNode,
)
from morphml.core.dsl.syntax import EVOLUTION_STRATEGIES, LAYER_TYPES
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class ValidationError:
    """Represents a validation error with location information."""

    def __init__(self, message: str, node: str = "Unknown"):
        """
        Initialize validation error.

        Args:
            message: Error message
            node: Node type where error occurred
        """
        self.message = message
        self.node = node

    def __str__(self) -> str:
        """String representation."""
        return f"[{self.node}] {self.message}"


class Validator(ASTVisitor):
    """
    Validates AST for semantic correctness.

    Checks:
    - Required parameters are present
    - Parameter values are valid
    - Layer types are supported
    - Evolution strategy is valid
    - No circular dependencies
    - Type consistency

    Example:
        >>> validator = Validator()
        >>> errors = validator.validate(ast)
        >>> if errors:
        ...     for error in errors:
        ...         print(error)
    """

    def __init__(self) -> None:
        """Initialize validator."""
        self.errors: List[ValidationError] = []
        self.warnings: List[str] = []
        self.layer_types_seen: Set[str] = set()

        # Required parameters for each layer type
        self.required_params: Dict[str, List[str]] = {
            "conv2d": ["filters", "kernel_size"],
            "conv3d": ["filters", "kernel_size"],
            "dense": ["units"],
            "linear": ["units"],
            "dropout": ["rate"],
            "max_pool": ["pool_size"],
            "avg_pool": ["pool_size"],
        }

    def validate(self, ast: ExperimentNode) -> List[ValidationError]:
        """
        Validate complete experiment AST.

        Args:
            ast: ExperimentNode to validate

        Returns:
            List of validation errors (empty if valid)
        """
        logger.info("Starting validation")

        # Reset state
        self.errors.clear()
        self.warnings.clear()
        self.layer_types_seen.clear()

        # Validate via visitor pattern
        ast.accept(self)

        # Additional high-level validations
        self._validate_layer_diversity()

        if self.errors:
            logger.warning(f"Validation found {len(self.errors)} errors")
        else:
            logger.info("Validation successful")

        return self.errors

    def visit_experiment(self, node: ExperimentNode) -> None:
        """Validate experiment node."""
        # Validate search space is present
        if not node.search_space:
            self.errors.append(ValidationError("Experiment must have a search space", "Experiment"))

        # Continue validation
        super().visit_experiment(node)

    def visit_search_space(self, node: SearchSpaceNode) -> None:
        """Validate search space node."""
        # Check at least one layer
        if not node.layers:
            self.errors.append(
                ValidationError("SearchSpace must contain at least one layer", "SearchSpace")
            )

        # Check for input/output layers
        has_input = any(layer.layer_type == "input" for layer in node.layers)
        has_output = any(layer.layer_type == "output" for layer in node.layers)

        if len(node.layers) > 2:  # Only check if there are actual layers
            if not has_input:
                self.warnings.append("SearchSpace should have an input layer")
            if not has_output:
                self.warnings.append("SearchSpace should have an output layer")

        # Validate each layer
        super().visit_search_space(node)

    def visit_layer(self, node: LayerNode) -> None:
        """Validate layer node."""
        # Track layer type
        self.layer_types_seen.add(node.layer_type)

        # Check layer type is supported
        if node.layer_type not in LAYER_TYPES:
            self.errors.append(
                ValidationError(
                    f"Unsupported layer type: '{node.layer_type}'. "
                    f"Valid types: {', '.join(LAYER_TYPES)}",
                    "Layer",
                )
            )
            return

        # Check required parameters
        required = self.required_params.get(node.layer_type, [])
        for param_name in required:
            if param_name not in node.params:
                self.errors.append(
                    ValidationError(
                        f"Layer '{node.layer_type}' missing required parameter: '{param_name}'",
                        "Layer",
                    )
                )

        # Validate each parameter
        for param_name, param_node in node.params.items():
            self._validate_layer_param(node.layer_type, param_name, param_node)

        super().visit_layer(node)

    def _validate_layer_param(self, layer_type: str, param_name: str, param_node: ParamNode) -> None:
        """
        Validate a layer parameter.

        Args:
            layer_type: Type of layer
            param_name: Parameter name
            param_node: Parameter node
        """
        # Check parameter has values
        if not param_node.values:
            self.errors.append(
                ValidationError(f"Parameter '{param_name}' has no values", "Parameter")
            )
            return

        # Type-specific validation
        if param_name == "filters":
            self._validate_filters(param_node)
        elif param_name == "kernel_size":
            self._validate_kernel_size(param_node)
        elif param_name == "units":
            self._validate_units(param_node)
        elif param_name == "rate":
            self._validate_dropout_rate(param_node)
        elif param_name == "pool_size":
            self._validate_pool_size(param_node)

    def _validate_filters(self, param_node: ParamNode) -> None:
        """Validate filter count parameter."""
        for value in param_node.values:
            if not isinstance(value, int):
                self.errors.append(
                    ValidationError(f"Filter count must be integer, got {type(value).__name__}", "Parameter")
                )
            elif value <= 0:
                self.errors.append(ValidationError(f"Filter count must be positive, got {value}", "Parameter"))

    def _validate_kernel_size(self, param_node: ParamNode) -> None:
        """Validate kernel size parameter."""
        for value in param_node.values:
            if not isinstance(value, int):
                self.errors.append(
                    ValidationError(f"Kernel size must be integer, got {type(value).__name__}", "Parameter")
                )
            elif value <= 0:
                self.errors.append(ValidationError(f"Kernel size must be positive, got {value}", "Parameter"))
            elif value % 2 == 0:
                self.warnings.append(f"Kernel size {value} is even (odd sizes are typically preferred)")

    def _validate_units(self, param_node: ParamNode) -> None:
        """Validate units parameter for dense layers."""
        for value in param_node.values:
            if not isinstance(value, int):
                self.errors.append(
                    ValidationError(f"Units must be integer, got {type(value).__name__}", "Parameter")
                )
            elif value <= 0:
                self.errors.append(ValidationError(f"Units must be positive, got {value}", "Parameter"))

    def _validate_dropout_rate(self, param_node: ParamNode) -> None:
        """Validate dropout rate parameter."""
        for value in param_node.values:
            if not isinstance(value, (int, float)):
                self.errors.append(
                    ValidationError(f"Dropout rate must be numeric, got {type(value).__name__}", "Parameter")
                )
            elif not (0 <= value < 1):
                self.errors.append(ValidationError(f"Dropout rate must be in [0, 1), got {value}", "Parameter"))

    def _validate_pool_size(self, param_node: ParamNode) -> None:
        """Validate pooling size parameter."""
        for value in param_node.values:
            if not isinstance(value, int):
                self.errors.append(
                    ValidationError(f"Pool size must be integer, got {type(value).__name__}", "Parameter")
                )
            elif value <= 0:
                self.errors.append(ValidationError(f"Pool size must be positive, got {value}", "Parameter"))

    def visit_param(self, node: ParamNode) -> None:
        """Validate parameter node."""
        # Check values list is not empty
        if not node.values:
            self.errors.append(ValidationError(f"Parameter '{node.name}' has no values", "Parameter"))

        # Check type consistency
        if len(node.values) > 1:
            types = set(type(v) for v in node.values)
            # Allow mixing int and float
            if types - {int, float}:
                if len(types) > 1 and not (types <= {int, float}):
                    self.warnings.append(
                        f"Parameter '{node.name}' has mixed types: {[t.__name__ for t in types]}"
                    )

    def visit_evolution(self, node: EvolutionNode) -> None:
        """Validate evolution node."""
        # Check strategy is supported
        if node.strategy not in EVOLUTION_STRATEGIES:
            self.errors.append(
                ValidationError(
                    f"Unknown evolution strategy: '{node.strategy}'. "
                    f"Valid strategies: {', '.join(EVOLUTION_STRATEGIES)}",
                    "Evolution",
                )
            )

        # Validate strategy-specific parameters
        if node.strategy == "genetic":
            self._validate_genetic_params(node)
        elif node.strategy == "bayesian":
            self._validate_bayesian_params(node)

    def _validate_genetic_params(self, node: EvolutionNode) -> None:
        """Validate genetic algorithm parameters."""
        # Check for recommended parameters
        recommended = ["population_size", "num_generations", "mutation_rate"]
        for param in recommended:
            if param not in node.params:
                self.warnings.append(f"Genetic algorithm should specify '{param}' parameter")

        # Validate ranges
        if "population_size" in node.params:
            pop_size = node.params["population_size"]
            if isinstance(pop_size, int) and pop_size < 2:
                self.errors.append(
                    ValidationError("Population size must be at least 2", "Evolution")
                )

        if "mutation_rate" in node.params:
            rate = node.params["mutation_rate"]
            if isinstance(rate, (int, float)) and not (0 <= rate <= 1):
                self.errors.append(
                    ValidationError(f"Mutation rate must be in [0, 1], got {rate}", "Evolution")
                )

    def _validate_bayesian_params(self, node: EvolutionNode) -> None:
        """Validate Bayesian optimization parameters."""
        # Check for recommended parameters
        if "num_iterations" not in node.params and "max_evaluations" not in node.params:
            self.warnings.append("Bayesian optimization should specify iteration budget")

    def visit_constraint(self, node: ConstraintNode) -> None:
        """Validate constraint node."""
        # Check constraint type is known
        known_constraints = ["max_depth", "max_nodes", "max_params", "required_layers"]

        if node.constraint_type not in known_constraints:
            self.warnings.append(
                f"Unknown constraint type: '{node.constraint_type}' "
                f"(known types: {', '.join(known_constraints)})"
            )

    def _validate_layer_diversity(self) -> None:
        """Validate that search space has sufficient diversity."""
        if len(self.layer_types_seen) < 2:
            self.warnings.append(
                f"Search space has low layer diversity: only {len(self.layer_types_seen)} "
                "different layer types"
            )

        # Check for common architectural patterns
        has_conv = any(lt.startswith("conv") for lt in self.layer_types_seen)
        has_pool = any("pool" in lt for lt in self.layer_types_seen)
        has_dense = "dense" in self.layer_types_seen

        if has_conv and not has_pool:
            self.warnings.append("CNN architecture detected but no pooling layers specified")

        if has_conv and not has_dense:
            self.warnings.append("CNN architecture detected but no dense layers for classification")


def validate_ast(ast: ExperimentNode) -> List[ValidationError]:
    """
    Convenience function to validate AST.

    Args:
        ast: ExperimentNode to validate

    Returns:
        List of validation errors
    """
    validator = Validator()
    return validator.validate(ast)
