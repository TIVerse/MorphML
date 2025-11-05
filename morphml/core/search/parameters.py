"""Parameter types for search space definition.

Provides explicit parameter classes for defining hyperparameter search spaces
with different distributions and constraints.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

from morphml.exceptions import ValidationError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class Parameter(ABC):
    """
    Base class for all parameter types.

    Defines the interface for sampling and validating parameter values
    in the search space.
    """

    def __init__(self, name: str):
        """
        Initialize parameter.

        Args:
            name: Parameter name
        """
        self.name = name

    @abstractmethod
    def sample(self) -> Any:
        """
        Sample a value from parameter space.

        Returns:
            Sampled value
        """
        pass

    @abstractmethod
    def validate(self, value: Any) -> bool:
        """
        Check if value is valid for this parameter.

        Args:
            value: Value to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize parameter to dictionary."""
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name})"


class CategoricalParameter(Parameter):
    """
    Categorical parameter with discrete choices.

    Samples uniformly from a list of possible values.

    Example:
        >>> param = CategoricalParameter('activation', ['relu', 'elu', 'gelu'])
        >>> value = param.sample()
        >>> print(value)  # One of: 'relu', 'elu', 'gelu'
    """

    def __init__(self, name: str, choices: List[Any], probabilities: Optional[List[float]] = None):
        """
        Initialize categorical parameter.

        Args:
            name: Parameter name
            choices: List of possible values
            probabilities: Optional probability weights for each choice

        Raises:
            ValidationError: If choices is empty or probabilities don't match
        """
        super().__init__(name)

        if not choices:
            raise ValidationError(f"Categorical parameter '{name}' must have at least one choice")

        if probabilities:
            if len(probabilities) != len(choices):
                raise ValidationError(
                    f"Probabilities length ({len(probabilities)}) must match choices length ({len(choices)})"
                )
            if abs(sum(probabilities) - 1.0) > 1e-6:
                raise ValidationError(f"Probabilities must sum to 1.0, got {sum(probabilities)}")

        self.choices = choices
        self.probabilities = probabilities

    def sample(self) -> Any:
        """Sample uniformly or with weights from choices."""
        if self.probabilities:
            return random.choices(self.choices, weights=self.probabilities, k=1)[0]
        return random.choice(self.choices)

    def validate(self, value: Any) -> bool:
        """Check if value is in choices."""
        return value in self.choices

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "type": "categorical",
            "name": self.name,
            "choices": self.choices,
            "probabilities": self.probabilities,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"CategoricalParameter({self.name}, choices={self.choices})"


class IntegerParameter(Parameter):
    """
    Integer parameter with min/max bounds.

    Samples integers uniformly or log-uniformly from a range.

    Example:
        >>> param = IntegerParameter('filters', 32, 512, log_scale=True)
        >>> value = param.sample()
        >>> print(value)  # Integer between 32 and 512
    """

    def __init__(
        self,
        name: str,
        low: int,
        high: int,
        log_scale: bool = False,
        step: int = 1,
    ):
        """
        Initialize integer parameter.

        Args:
            name: Parameter name
            low: Minimum value (inclusive)
            high: Maximum value (inclusive)
            log_scale: If True, sample on log scale (good for powers of 2)
            step: Step size for sampling

        Raises:
            ValidationError: If low >= high
        """
        super().__init__(name)

        if low >= high:
            raise ValidationError(f"low ({low}) must be less than high ({high})")

        self.low = low
        self.high = high
        self.log_scale = log_scale
        self.step = step

    def sample(self) -> int:
        """Sample integer from range."""
        if self.log_scale:
            import math

            # Sample on log scale
            log_low = math.log2(self.low)
            log_high = math.log2(self.high)
            log_value = random.uniform(log_low, log_high)
            value = int(2**log_value)

            # Ensure within bounds
            value = max(self.low, min(self.high, value))
        else:
            # Linear sampling
            value = random.randint(self.low, self.high)

        # Apply step
        if self.step > 1:
            value = (value // self.step) * self.step

        return value

    def validate(self, value: Any) -> bool:
        """Check if value is valid integer in range."""
        if not isinstance(value, int):
            return False
        if value < self.low or value > self.high:
            return False
        if self.step > 1 and (value % self.step) != 0:
            return False
        return True

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "type": "integer",
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "log_scale": self.log_scale,
            "step": self.step,
        }

    def __repr__(self) -> str:
        """String representation."""
        scale = "log" if self.log_scale else "linear"
        return f"IntegerParameter({self.name}, [{self.low}, {self.high}], {scale})"


class FloatParameter(Parameter):
    """
    Float parameter with min/max bounds.

    Samples floats uniformly or log-uniformly from a range.

    Example:
        >>> param = FloatParameter('learning_rate', 1e-4, 1e-2, log_scale=True)
        >>> value = param.sample()
        >>> print(value)  # Float between 0.0001 and 0.01
    """

    def __init__(
        self,
        name: str,
        low: float,
        high: float,
        log_scale: bool = False,
    ):
        """
        Initialize float parameter.

        Args:
            name: Parameter name
            low: Minimum value
            high: Maximum value
            log_scale: If True, sample on log scale (good for learning rates)

        Raises:
            ValidationError: If low >= high
        """
        super().__init__(name)

        if low >= high:
            raise ValidationError(f"low ({low}) must be less than high ({high})")

        self.low = low
        self.high = high
        self.log_scale = log_scale

    def sample(self) -> float:
        """Sample float from range."""
        if self.log_scale:
            import math

            # Sample on log scale
            log_low = math.log10(self.low)
            log_high = math.log10(self.high)
            log_value = random.uniform(log_low, log_high)
            return 10**log_value
        else:
            # Linear sampling
            return random.uniform(self.low, self.high)

    def validate(self, value: Any) -> bool:
        """Check if value is valid float in range."""
        if not isinstance(value, (int, float)):
            return False
        return self.low <= value <= self.high

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "type": "float",
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "log_scale": self.log_scale,
        }

    def __repr__(self) -> str:
        """String representation."""
        scale = "log" if self.log_scale else "linear"
        return f"FloatParameter({self.name}, [{self.low}, {self.high}], {scale})"


class BooleanParameter(Parameter):
    """
    Boolean parameter.

    Samples True/False with configurable probability.

    Example:
        >>> param = BooleanParameter('use_dropout', probability=0.7)
        >>> value = param.sample()
        >>> print(value)  # True with 70% probability
    """

    def __init__(self, name: str, probability: float = 0.5):
        """
        Initialize boolean parameter.

        Args:
            name: Parameter name
            probability: Probability of sampling True (default 0.5)

        Raises:
            ValidationError: If probability not in [0, 1]
        """
        super().__init__(name)

        if not (0 <= probability <= 1):
            raise ValidationError(f"Probability must be in [0, 1], got {probability}")

        self.probability = probability

    def sample(self) -> bool:
        """Sample boolean value."""
        return random.random() < self.probability

    def validate(self, value: Any) -> bool:
        """Check if value is boolean."""
        return isinstance(value, bool)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "type": "boolean",
            "name": self.name,
            "probability": self.probability,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"BooleanParameter({self.name}, p={self.probability})"


class ConstantParameter(Parameter):
    """
    Constant parameter (always returns same value).

    Useful for fixed hyperparameters that shouldn't be searched.

    Example:
        >>> param = ConstantParameter('batch_size', 32)
        >>> value = param.sample()
        >>> print(value)  # Always 32
    """

    def __init__(self, name: str, value: Any):
        """
        Initialize constant parameter.

        Args:
            name: Parameter name
            value: Constant value
        """
        super().__init__(name)
        self.value = value

    def sample(self) -> Any:
        """Return constant value."""
        return self.value

    def validate(self, value: Any) -> bool:
        """Check if value equals constant."""
        return value == self.value

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "type": "constant",
            "name": self.name,
            "value": self.value,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"ConstantParameter({self.name}={self.value})"


# Factory function for creating parameters from specification
def create_parameter(
    name: str, spec: Union[List, tuple, dict, Any]
) -> Parameter:
    """
    Create parameter from specification.

    Args:
        name: Parameter name
        spec: Parameter specification (list of choices, tuple of (low, high), dict, or constant)

    Returns:
        Parameter instance

    Example:
        >>> p1 = create_parameter('activation', ['relu', 'elu'])
        >>> p2 = create_parameter('filters', (32, 512))
        >>> p3 = create_parameter('dropout_rate', (0.1, 0.5))
    """
    if isinstance(spec, list):
        # List of choices -> Categorical
        return CategoricalParameter(name, spec)

    elif isinstance(spec, tuple) and len(spec) == 2:
        low, high = spec
        if isinstance(low, int) and isinstance(high, int):
            # Integer range
            return IntegerParameter(name, low, high)
        else:
            # Float range
            return FloatParameter(name, float(low), float(high))

    elif isinstance(spec, dict):
        # Dictionary specification
        param_type = spec.get("type", "categorical")

        if param_type == "categorical":
            return CategoricalParameter(
                name, spec["choices"], spec.get("probabilities")
            )
        elif param_type == "integer":
            return IntegerParameter(
                name,
                spec["low"],
                spec["high"],
                spec.get("log_scale", False),
                spec.get("step", 1),
            )
        elif param_type == "float":
            return FloatParameter(
                name, spec["low"], spec["high"], spec.get("log_scale", False)
            )
        elif param_type == "boolean":
            return BooleanParameter(name, spec.get("probability", 0.5))
        elif param_type == "constant":
            return ConstantParameter(name, spec["value"])
        else:
            raise ValidationError(f"Unknown parameter type: {param_type}")

    else:
        # Single value -> Constant
        return ConstantParameter(name, spec)
