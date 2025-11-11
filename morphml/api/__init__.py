"""REST API for MorphML.

Provides programmatic access to MorphML functionality via HTTP endpoints.

Example:
    # Start server
    morphml api --port 8000

    # Or programmatically
    from morphml.api import create_app
    app = create_app()
"""

from morphml.api.app import create_app
from morphml.api.models import (
    ArchitectureResponse,
    ExperimentCreate,
    ExperimentResponse,
)

__all__ = [
    "create_app",
    "ExperimentCreate",
    "ExperimentResponse",
    "ArchitectureResponse",
]
