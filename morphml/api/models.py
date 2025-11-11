"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExperimentCreate(BaseModel):
    """Request model for creating an experiment."""

    name: str = Field(..., description="Experiment name")
    search_space_config: Dict[str, Any] = Field(..., description="Search space configuration")
    optimizer_type: str = Field("genetic", description="Optimizer type")
    optimizer_config: Dict[str, Any] = Field(
        default_factory=dict, description="Optimizer configuration"
    )
    budget: int = Field(100, description="Evaluation budget")
    constraints: Optional[List[Dict[str, Any]]] = Field(None, description="Constraints")

    class Config:
        schema_extra = {
            "example": {
                "name": "cifar10-search",
                "search_space_config": {
                    "layers": [
                        {"type": "input", "shape": [3, 32, 32]},
                        {"type": "conv2d", "filters": [32, 64], "kernel_size": 3},
                        {"type": "flatten"},
                        {"type": "dense", "units": [128, 256]},
                    ]
                },
                "optimizer_type": "genetic",
                "optimizer_config": {"population_size": 20, "num_generations": 50},
                "budget": 1000,
            }
        }


class ExperimentResponse(BaseModel):
    """Response model for experiment."""

    id: str
    name: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    best_fitness: Optional[float] = None
    generations_completed: int = 0
    total_generations: int

    class Config:
        schema_extra = {
            "example": {
                "id": "exp_123abc",
                "name": "cifar10-search",
                "status": "running",
                "created_at": "2024-11-11T05:00:00Z",
                "started_at": "2024-11-11T05:01:00Z",
                "best_fitness": 0.9523,
                "generations_completed": 25,
                "total_generations": 50,
            }
        }


class ArchitectureResponse(BaseModel):
    """Response model for architecture."""

    id: str
    experiment_id: str
    fitness: float
    parameters: int
    depth: int
    nodes: int
    edges: int
    created_at: datetime
    graph_data: Optional[Dict[str, Any]] = None

    class Config:
        schema_extra = {
            "example": {
                "id": "arch_456def",
                "experiment_id": "exp_123abc",
                "fitness": 0.9523,
                "parameters": 1234567,
                "depth": 8,
                "nodes": 12,
                "edges": 11,
                "created_at": "2024-11-11T05:15:00Z",
            }
        }


class SearchSpaceCreate(BaseModel):
    """Request model for creating a search space."""

    name: str
    layers: List[Dict[str, Any]]
    constraints: Optional[List[Dict[str, Any]]] = None


class OptimizerInfo(BaseModel):
    """Information about an optimizer."""

    name: str
    type: str
    description: str
    parameters: Dict[str, Any]

    class Config:
        schema_extra = {
            "example": {
                "name": "GeneticAlgorithm",
                "type": "evolutionary",
                "description": "Genetic algorithm for neural architecture search",
                "parameters": {
                    "population_size": {"type": "int", "default": 20},
                    "num_generations": {"type": "int", "default": 50},
                    "mutation_rate": {"type": "float", "default": 0.2},
                },
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: datetime
