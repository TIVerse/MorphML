"""Logging configuration for MorphML."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging for MorphML.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        console: Enable console logging with Rich
        format_string: Custom format string (if not using Rich)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging(level="DEBUG", log_file="morphml.log")
        >>> logger.info("Starting experiment")
    """
    # Get root logger for morphml
    logger = logging.getLogger("morphml")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler with Rich
    if console:
        console_handler = RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            console=Console(stderr=True),
        )
        console_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always DEBUG to file

        # Use custom format for file
        if format_string is None:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - " "%(filename)s:%(lineno)d - %(message)s"
            )

        formatter = logging.Formatter(format_string)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(f"morphml.{name}")


# Create default logger
logger = logging.getLogger("morphml")
