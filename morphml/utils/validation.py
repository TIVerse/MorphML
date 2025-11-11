"""Search space validation utilities.

Validate search spaces before running expensive NAS operations.

Example:
    >>> from morphml.utils.validation import validate_search_space
    >>> 
    >>> issues = validate_search_space(space)
    >>> if issues:
    ...     for issue in issues:
    ...         print(f"Warning: {issue}")
"""

from typing import List, Dict, Any, Optional
from morphml.core.dsl import SearchSpace
from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class ValidationIssue:
    """Represents a validation issue."""

    def __init__(self, severity: str, message: str, suggestion: Optional[str] = None):
        """
        Initialize validation issue.

        Args:
            severity: "error", "warning", or "info"
            message: Issue description
            suggestion: Optional fix suggestion
        """
        self.severity = severity
        self.message = message
        self.suggestion = suggestion

    def __str__(self) -> str:
        result = f"[{self.severity.upper()}] {self.message}"
        if self.suggestion:
            result += f"\n  Suggestion: {self.suggestion}"
        return result

    def __repr__(self) -> str:
        return f"ValidationIssue(severity='{self.severity}', message='{self.message}')"


class SearchSpaceValidator:
    """
    Validate search spaces for common issues.

    Example:
        >>> validator = SearchSpaceValidator(space)
        >>> validator.validate()
        >>> validator.print_report()
    """

    def __init__(self, search_space: SearchSpace):
        """
        Initialize validator.

        Args:
            search_space: Search space to validate
        """
        self.search_space = search_space
        self.issues: List[ValidationIssue] = []

    def validate(self) -> List[ValidationIssue]:
        """
        Run all validation checks.

        Returns:
            List of validation issues
        """
        self.issues.clear()

        self._check_layer_count()
        self._check_layer_types()
        self._check_search_space_size()
        self._check_flatten_usage()
        self._check_parameter_ranges()
        self._check_connectivity()

        return self.issues

    def _check_layer_count(self):
        """Check if layer count is reasonable."""
        num_layers = len(self.search_space.layers)

        if num_layers == 0:
            self.issues.append(
                ValidationIssue(
                    "error", "Search space has no layers", "Add layers using space.add_layers()"
                )
            )
        elif num_layers < 3:
            self.issues.append(
                ValidationIssue(
                    "warning",
                    f"Search space has only {num_layers} layers",
                    "Consider adding more layers for meaningful search",
                )
            )
        elif num_layers > 50:
            self.issues.append(
                ValidationIssue(
                    "warning",
                    f"Search space has {num_layers} layers (very deep)",
                    "Deep spaces may be slow to search",
                )
            )

    def _check_layer_types(self):
        """Check layer type distribution."""
        layer_types = {}
        has_input = False
        has_output = False

        for layer_spec in self.search_space.layers:
            op = layer_spec.operation
            layer_types[op] = layer_types.get(op, 0) + 1

            if op == "input":
                has_input = True
            if op in ["output", "softmax"]:
                has_output = True

        # Check for input layer
        if not has_input:
            self.issues.append(
                ValidationIssue(
                    "warning",
                    "No explicit input layer",
                    "Add Layer.input(shape=...) at the beginning",
                )
            )

        # Check for output activation
        if not has_output:
            self.issues.append(
                ValidationIssue(
                    "info",
                    "No explicit output activation",
                    "Consider adding Layer.softmax() or Layer.sigmoid() at the end",
                )
            )

        # Check for activation functions
        activations = sum(layer_types.get(act, 0) for act in ["relu", "sigmoid", "tanh", "softmax"])
        if activations == 0:
            self.issues.append(
                ValidationIssue(
                    "warning",
                    "No activation functions found",
                    "Add activation layers like Layer.relu()",
                )
            )

    def _check_search_space_size(self):
        """Estimate search space size."""
        try:
            size = 1
            for layer_spec in self.search_space.layers:
                # Count choices for each parameter
                layer_choices = 1
                for param_values in layer_spec.param_ranges.values():
                    if isinstance(param_values, list) and len(param_values) > 1:
                        layer_choices *= len(param_values)
                size *= max(layer_choices, 1)

            if size > 1e12:
                self.issues.append(
                    ValidationIssue(
                        "warning",
                        f"Extremely large search space (~{size:.2e} combinations)",
                        "Consider reducing parameter choices",
                    )
                )
            elif size > 1e9:
                self.issues.append(
                    ValidationIssue(
                        "info",
                        f"Large search space (~{size:.2e} combinations)",
                        "May require many generations to explore",
                    )
                )
            elif size < 100:
                self.issues.append(
                    ValidationIssue(
                        "info",
                        f"Small search space (~{size} combinations)",
                        "Consider adding more parameter choices for richer search",
                    )
                )

        except Exception as e:
            logger.debug(f"Could not estimate search space size: {e}")

    def _check_flatten_usage(self):
        """Check flatten layer usage."""
        has_conv = False
        has_dense = False
        has_flatten = False

        for layer_spec in self.search_space.layers:
            if layer_spec.operation in ["conv2d", "maxpool", "avgpool"]:
                has_conv = True
            elif layer_spec.operation == "dense":
                has_dense = True
            elif layer_spec.operation == "flatten":
                has_flatten = True

        # If has both conv and dense, should have flatten
        if has_conv and has_dense and not has_flatten:
            self.issues.append(
                ValidationIssue(
                    "warning",
                    "Conv layers followed by Dense without Flatten",
                    "Add Layer.flatten() between conv and dense layers",
                )
            )

    def _check_parameter_ranges(self):
        """Check parameter value ranges."""
        for layer_spec in self.search_space.layers:
            op = layer_spec.operation

            # Check conv2d filters
            if op == "conv2d":
                filters = layer_spec.param_ranges.get("filters", [])
                if isinstance(filters, list):
                    if any(f < 1 for f in filters):
                        self.issues.append(
                            ValidationIssue(
                                "error",
                                f"Conv2d has invalid filter count: {filters}",
                                "Filters must be positive integers",
                            )
                        )
                    if any(f > 1024 for f in filters):
                        self.issues.append(
                            ValidationIssue(
                                "warning",
                                f"Conv2d has very large filter count: {max(filters)}",
                                "Large filter counts increase parameters significantly",
                            )
                        )

            # Check dense units
            if op == "dense":
                units = layer_spec.param_ranges.get("units", [])
                if isinstance(units, list):
                    if any(u < 1 for u in units):
                        self.issues.append(
                            ValidationIssue(
                                "error",
                                f"Dense has invalid unit count: {units}",
                                "Units must be positive integers",
                            )
                        )

            # Check dropout rate
            if op == "dropout":
                rates = layer_spec.param_ranges.get("rate", [])
                if isinstance(rates, list):
                    if any(r < 0 or r >= 1 for r in rates):
                        self.issues.append(
                            ValidationIssue(
                                "error",
                                f"Dropout has invalid rate: {rates}",
                                "Dropout rate must be in [0, 1)",
                            )
                        )

    def _check_connectivity(self):
        """Check if sampled architectures will be connected."""
        try:
            # Sample a few architectures to check
            for _ in range(3):
                graph = self.search_space.sample()

                if not graph.is_valid():
                    self.issues.append(
                        ValidationIssue(
                            "error",
                            "Sampled architecture is not a valid DAG",
                            "Check search space definition",
                        )
                    )
                    break

        except Exception as e:
            self.issues.append(
                ValidationIssue(
                    "error",
                    f"Failed to sample from search space: {e}",
                    "Check layer definitions and parameters",
                )
            )

    def print_report(self):
        """Print validation report."""
        if not self.issues:
            print("✓ Search space validation passed!")
            return

        print(f"\nSearch Space Validation Report")
        print("=" * 70)

        errors = [i for i in self.issues if i.severity == "error"]
        warnings = [i for i in self.issues if i.severity == "warning"]
        infos = [i for i in self.issues if i.severity == "info"]

        if errors:
            print(f"\n❌ Errors ({len(errors)}):")
            for issue in errors:
                print(f"  {issue}")

        if warnings:
            print(f"\n⚠️  Warnings ({len(warnings)}):")
            for issue in warnings:
                print(f"  {issue}")

        if infos:
            print(f"\nℹ️  Info ({len(infos)}):")
            for issue in infos:
                print(f"  {issue}")

        print("=" * 70)

        if errors:
            print("\n❌ Validation failed! Fix errors before running NAS.")
        elif warnings:
            print("\n⚠️  Validation passed with warnings. Review before running NAS.")
        else:
            print("\n✓ Validation passed!")

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(i.severity == "error" for i in self.issues)

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(i.severity == "warning" for i in self.issues)


def validate_search_space(
    search_space: SearchSpace,
    print_report: bool = True,
) -> List[ValidationIssue]:
    """
    Validate search space and optionally print report.

    Args:
        search_space: Search space to validate
        print_report: Whether to print validation report

    Returns:
        List of validation issues

    Example:
        >>> issues = validate_search_space(space)
        >>> if any(i.severity == "error" for i in issues):
        ...     print("Fix errors before running NAS!")
    """
    validator = SearchSpaceValidator(search_space)
    issues = validator.validate()

    if print_report:
        validator.print_report()

    return issues


def quick_validate(search_space: SearchSpace) -> bool:
    """
    Quick validation returning True if no errors.

    Args:
        search_space: Search space to validate

    Returns:
        True if validation passed (no errors)

    Example:
        >>> if quick_validate(space):
        ...     # Run NAS
        ...     pass
    """
    validator = SearchSpaceValidator(search_space)
    issues = validator.validate()
    return not validator.has_errors()
