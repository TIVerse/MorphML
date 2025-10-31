"""Custom exceptions for MorphML."""


class MorphMLError(Exception):
    """Base exception for all MorphML errors."""

    pass


class DSLError(MorphMLError):
    """Raised when DSL parsing or compilation fails."""

    pass


class GraphError(MorphMLError):
    """Raised when graph operations fail."""

    pass


class SearchSpaceError(MorphMLError):
    """Raised when search space definition is invalid."""

    pass


class OptimizerError(MorphMLError):
    """Raised when optimizer encounters an error."""

    pass


class EvaluationError(MorphMLError):
    """Raised when architecture evaluation fails."""

    pass


class ConfigurationError(MorphMLError):
    """Raised when configuration is invalid."""

    pass


class DistributedError(MorphMLError):
    """Raised when distributed operations fail."""

    pass


class ValidationError(MorphMLError):
    """Raised when validation fails."""

    pass
