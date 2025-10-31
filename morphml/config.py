"""Configuration management for MorphML."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field

from morphml.exceptions import ConfigurationError


class OptimizerConfig(BaseModel):
    """Configuration for optimizers."""

    name: str = Field(description="Optimizer name (genetic, bayesian, darts, etc.)")
    population_size: int = Field(default=50, ge=1, description="Population size")
    num_generations: int = Field(default=100, ge=1, description="Number of generations")
    mutation_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Mutation probability")
    crossover_rate: float = Field(default=0.8, ge=0.0, le=1.0, description="Crossover probability")

    class Config:
        extra = "allow"  # Allow additional fields for optimizer-specific config


class EvaluationConfig(BaseModel):
    """Configuration for architecture evaluation."""

    num_epochs: int = Field(default=50, ge=1, description="Training epochs")
    batch_size: int = Field(default=128, ge=1, description="Batch size")
    learning_rate: float = Field(default=0.001, gt=0.0, description="Learning rate")
    device: str = Field(default="cuda", description="Device (cuda or cpu)")
    num_workers: int = Field(default=4, ge=0, description="Data loader workers")


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format",
    )
    file: Optional[str] = Field(default=None, description="Log file path")
    console: bool = Field(default=True, description="Enable console logging")


class ExperimentConfig(BaseModel):
    """Main experiment configuration."""

    name: str = Field(description="Experiment name")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    output_dir: Path = Field(default=Path("./experiments"), description="Output directory")

    optimizer: OptimizerConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    class Config:
        arbitrary_types_allowed = True


class ConfigManager:
    """
    Manage MorphML configuration.

    Supports loading from:
    - YAML files
    - Environment variables
    - Python dictionaries

    Usage:
        config = ConfigManager.from_yaml("config.yaml")
        config.get("optimizer.population_size")
        config.set("optimizer.mutation_rate", 0.2)
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration manager.

        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict or {}
        self._experiment_config: Optional[ExperimentConfig] = None

    @classmethod
    def from_yaml(cls, config_path: str) -> "ConfigManager":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML file

        Returns:
            ConfigManager instance

        Raises:
            ConfigurationError: If file cannot be loaded
        """
        try:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            return cls(config_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {config_path}: {e}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ConfigManager":
        """Create configuration from dictionary."""
        return cls(config_dict)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key.

        Args:
            key: Dot-separated key (e.g., "optimizer.population_size")
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> config.get("optimizer.mutation_rate")
            0.1
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-separated key.

        Args:
            key: Dot-separated key
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def validate(self) -> ExperimentConfig:
        """
        Validate configuration using Pydantic model.

        Returns:
            Validated ExperimentConfig

        Raises:
            ConfigurationError: If validation fails
        """
        try:
            self._experiment_config = ExperimentConfig(**self._config)
            return self._experiment_config
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return self._config.copy()

    def save(self, output_path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            output_path: Output file path
        """
        with open(output_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def merge(self, other_config: Dict[str, Any]) -> None:
        """
        Merge another configuration into this one.

        Args:
            other_config: Configuration dictionary to merge
        """
        self._deep_merge(self._config, other_config)

    @staticmethod
    def _deep_merge(base: Dict, update: Dict) -> None:
        """Deep merge update dict into base dict."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ConfigManager._deep_merge(base[key], value)
            else:
                base[key] = value


def load_config_from_env() -> Dict[str, Any]:
    """
    Load configuration from environment variables.

    Environment variables should be prefixed with MORPHML_
    Example: MORPHML_OPTIMIZER_POPULATION_SIZE=100

    Returns:
        Configuration dictionary
    """
    config: Dict[str, Any] = {}
    prefix = "MORPHML_"

    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Remove prefix and convert to lowercase with dots
            config_key = key[len(prefix) :].lower().replace("_", ".")

            # Try to convert to int/float
            try:
                if "." in value:
                    value = float(value)  # type: ignore
                else:
                    value = int(value)  # type: ignore
            except ValueError:
                pass

            # Set in config
            keys = config_key.split(".")
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

    return config


# Default configuration template
DEFAULT_CONFIG = {
    "name": "morphml_experiment",
    "seed": 42,
    "output_dir": "./experiments",
    "optimizer": {
        "name": "genetic",
        "population_size": 50,
        "num_generations": 100,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
    },
    "evaluation": {
        "num_epochs": 50,
        "batch_size": 128,
        "learning_rate": 0.001,
        "device": "cuda",
        "num_workers": 4,
    },
    "logging": {"level": "INFO", "console": True, "file": None},
}
