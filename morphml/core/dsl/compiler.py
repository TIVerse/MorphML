"""Compiler for MorphML DSL.

Compiles Abstract Syntax Tree into executable internal representation.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Any, Dict, List

from morphml.core.dsl.ast_nodes import (
    ConstraintNode,
    EvolutionNode,
    ExperimentNode,
    LayerNode,
    ParamNode,
    SearchSpaceNode,
)
from morphml.core.dsl.layers import LayerSpec
from morphml.core.dsl.search_space import SearchSpace
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class Compiler:
    """
    Compiles AST into internal representation.

    Transforms the abstract syntax tree into executable objects:
    - ParamNode → parameter ranges
    - LayerNode → LayerSpec
    - SearchSpaceNode → SearchSpace
    - EvolutionNode → optimizer configuration

    Example:
        >>> compiler = Compiler()
        >>> result = compiler.compile(ast)
        >>> search_space = result['search_space']
    """

    def __init__(self) -> None:
        """Initialize compiler."""
        self.symbol_table: Dict[str, Any] = {}

    def compile(self, ast: ExperimentNode) -> Dict[str, Any]:
        """
        Compile complete experiment AST.

        Args:
            ast: ExperimentNode from parser

        Returns:
            Dictionary with 'search_space', 'evolution', 'constraints'

        Raises:
            ValidationError: If compilation fails
        """
        logger.info("Starting compilation")

        # Compile search space
        search_space = self._compile_search_space(ast.search_space)

        # Compile evolution config
        evolution_config = None
        if ast.evolution:
            evolution_config = self._compile_evolution(ast.evolution)

        # Compile constraints
        constraints = [self._compile_constraint(c) for c in ast.constraints]

        logger.info(
            f"Compilation complete: {len(search_space.layers)} layers, "
            f"{len(constraints)} constraints"
        )

        return {
            "search_space": search_space,
            "evolution": evolution_config,
            "constraints": constraints,
            "objectives": ast.objectives,
        }

    def _compile_search_space(self, node: SearchSpaceNode) -> SearchSpace:
        """
        Convert SearchSpaceNode to SearchSpace object.

        Args:
            node: SearchSpaceNode from AST

        Returns:
            SearchSpace instance
        """
        # Compile layers
        layer_specs = [self._compile_layer(layer_node) for layer_node in node.layers]

        # Create search space
        search_space = SearchSpace(name=node.name or "compiled_space")

        # Add layers
        for spec in layer_specs:
            search_space.layers.append(spec)

        # Add global parameters if any
        for param_name, param_node in node.global_params.items():
            search_space.metadata[param_name] = self._compile_param_values(param_node)

        return search_space

    def _compile_layer(self, node: LayerNode) -> LayerSpec:
        """
        Convert LayerNode to LayerSpec.

        Args:
            node: LayerNode from AST

        Returns:
            LayerSpec instance
        """
        # Compile parameters
        param_ranges: Dict[str, Any] = {}

        for param_name, param_node in node.params.items():
            param_ranges[param_name] = self._compile_param_values(param_node)

        # Create LayerSpec
        return LayerSpec(
            operation=node.layer_type, param_ranges=param_ranges, metadata=node.metadata.copy()
        )

    def _compile_param_values(self, param_node: ParamNode) -> List[Any]:
        """
        Compile parameter values from ParamNode.

        Args:
            param_node: ParamNode from AST

        Returns:
            List of values (for sampling)
        """
        return param_node.values

    def _compile_evolution(self, node: EvolutionNode) -> Dict[str, Any]:
        """
        Convert EvolutionNode to optimizer configuration.

        Args:
            node: EvolutionNode from AST

        Returns:
            Configuration dictionary
        """
        config = {"optimizer_type": node.strategy}

        # Map DSL parameter names to internal names
        param_mapping = {
            "population_size": "population_size",
            "num_generations": "num_generations",
            "mutation_rate": "mutation_rate",
            "crossover_rate": "crossover_rate",
            "elite_size": "elitism",
            "elitism": "elitism",
            "selection": "selection_method",
            "selection_strategy": "selection_method",
            "tournament_size": "tournament_size",
            "max_evaluations": "max_evaluations",
            "early_stopping": "early_stopping_patience",
        }

        # Compile parameters
        for param_name, param_value in node.params.items():
            # Get internal parameter name
            internal_name = param_mapping.get(param_name, param_name)

            # Handle list values (take first if single element)
            if isinstance(param_value, list):
                if len(param_value) == 1:
                    param_value = param_value[0]

            config[internal_name] = param_value

        return config

    def _compile_constraint(self, node: ConstraintNode) -> Dict[str, Any]:
        """
        Convert ConstraintNode to constraint configuration.

        Args:
            node: ConstraintNode from AST

        Returns:
            Constraint configuration dictionary
        """
        return {"type": node.constraint_type, "params": node.params.copy()}


class CompilationContext:
    """
    Context for compilation process.

    Tracks symbols, types, and other compilation state.
    """

    def __init__(self) -> None:
        """Initialize context."""
        self.symbols: Dict[str, Any] = {}
        self.errors: List[str] = []

    def add_symbol(self, name: str, value: Any) -> None:
        """Add symbol to context."""
        self.symbols[name] = value

    def get_symbol(self, name: str) -> Any:
        """Get symbol from context."""
        return self.symbols.get(name)

    def add_error(self, message: str) -> None:
        """Record compilation error."""
        self.errors.append(message)

    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0


def compile_dsl(source: str) -> Dict[str, Any]:
    """
    Convenience function to parse and compile DSL source.

    Args:
        source: DSL source code

    Returns:
        Compiled representation

    Example:
        >>> source = '''
        ... SearchSpace(
        ...     layers=[
        ...         Layer.conv2d(filters=[32, 64], kernel_size=3),
        ...         Layer.relu()
        ...     ]
        ... )
        ... Evolution(strategy="genetic", population_size=50)
        ... '''
        >>> result = compile_dsl(source)
        >>> search_space = result['search_space']
        >>> config = result['evolution']
    """
    from morphml.core.dsl.parser import parse_dsl

    # Parse source to AST
    ast = parse_dsl(source)

    # Compile AST
    compiler = Compiler()
    return compiler.compile(ast)


def compile_to_search_space(ast: ExperimentNode) -> SearchSpace:
    """
    Compile AST directly to SearchSpace.

    Args:
        ast: ExperimentNode

    Returns:
        SearchSpace instance
    """
    compiler = Compiler()
    result = compiler.compile(ast)
    return result["search_space"]


# Type inference helpers
def infer_parameter_type(values: List[Any]) -> str:
    """
    Infer parameter type from values.

    Args:
        values: List of parameter values

    Returns:
        Type string ('categorical', 'integer', 'float', 'boolean')
    """
    if not values:
        return "categorical"

    if all(isinstance(v, bool) for v in values):
        return "boolean"
    elif all(isinstance(v, int) for v in values):
        return "integer"
    elif all(isinstance(v, (int, float)) for v in values):
        return "float"
    else:
        return "categorical"


def validate_parameter_range(param_type: str, values: List[Any]) -> bool:
    """
    Validate that parameter values match declared type.

    Args:
        param_type: Parameter type
        values: Parameter values

    Returns:
        True if valid
    """
    if param_type == "boolean":
        return all(isinstance(v, bool) for v in values)
    elif param_type == "integer":
        return all(isinstance(v, int) for v in values)
    elif param_type == "float":
        return all(isinstance(v, (int, float)) for v in values)
    elif param_type == "categorical":
        return True  # Any type allowed
    return False
