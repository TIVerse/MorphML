"""FastAPI application for MorphML REST API.

This module provides the main FastAPI application with all endpoints.
"""

from datetime import datetime
from typing import List, Optional
import uuid

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None

from morphml.api.models import (
    ExperimentCreate,
    ExperimentResponse,
    ArchitectureResponse,
    OptimizerInfo,
    HealthResponse,
)
from morphml.version import __version__
from morphml.logging_config import get_logger

logger = get_logger(__name__)

# In-memory storage (replace with database in production)
experiments_db = {}
architectures_db = {}


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI app
        
    Example:
        >>> app = create_app()
        >>> # Run with: uvicorn morphml.api.app:app --reload
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for the REST API. "
            "Install with: pip install 'morphml[api]' or pip install fastapi uvicorn"
        )
    
    app = FastAPI(
        title="MorphML API",
        description="REST API for Neural Architecture Search with MorphML",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Check API health status."""
        return HealthResponse(
            status="healthy",
            version=__version__,
            timestamp=datetime.now()
        )
    
    # Experiment endpoints
    @app.post(
        "/api/v1/experiments",
        response_model=ExperimentResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Experiments"]
    )
    async def create_experiment(experiment: ExperimentCreate):
        """
        Create a new NAS experiment.
        
        Args:
            experiment: Experiment configuration
            
        Returns:
            Created experiment details
        """
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        
        exp_data = ExperimentResponse(
            id=experiment_id,
            name=experiment.name,
            status="pending",
            created_at=datetime.now(),
            best_fitness=None,
            generations_completed=0,
            total_generations=experiment.optimizer_config.get("num_generations", 50)
        )
        
        experiments_db[experiment_id] = {
            "response": exp_data,
            "config": experiment.dict()
        }
        
        logger.info(f"Created experiment: {experiment_id}")
        return exp_data
    
    @app.get(
        "/api/v1/experiments",
        response_model=List[ExperimentResponse],
        tags=["Experiments"]
    )
    async def list_experiments(
        status_filter: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ):
        """
        List all experiments.
        
        Args:
            status_filter: Filter by status (pending, running, completed, failed)
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of experiments
        """
        experiments = [
            exp["response"]
            for exp in experiments_db.values()
        ]
        
        if status_filter:
            experiments = [
                exp for exp in experiments
                if exp.status == status_filter
            ]
        
        return experiments[offset:offset + limit]
    
    @app.get(
        "/api/v1/experiments/{experiment_id}",
        response_model=ExperimentResponse,
        tags=["Experiments"]
    )
    async def get_experiment(experiment_id: str):
        """
        Get experiment details.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment details
        """
        if experiment_id not in experiments_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        
        return experiments_db[experiment_id]["response"]
    
    @app.post(
        "/api/v1/experiments/{experiment_id}/start",
        response_model=ExperimentResponse,
        tags=["Experiments"]
    )
    async def start_experiment(experiment_id: str):
        """
        Start an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Updated experiment details
        """
        if experiment_id not in experiments_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        
        exp_data = experiments_db[experiment_id]["response"]
        exp_data.status = "running"
        exp_data.started_at = datetime.now()
        
        logger.info(f"Started experiment: {experiment_id}")
        return exp_data
    
    @app.post(
        "/api/v1/experiments/{experiment_id}/stop",
        response_model=ExperimentResponse,
        tags=["Experiments"]
    )
    async def stop_experiment(experiment_id: str):
        """
        Stop a running experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Updated experiment details
        """
        if experiment_id not in experiments_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        
        exp_data = experiments_db[experiment_id]["response"]
        if exp_data.status == "running":
            exp_data.status = "stopped"
            exp_data.completed_at = datetime.now()
        
        logger.info(f"Stopped experiment: {experiment_id}")
        return exp_data
    
    @app.delete(
        "/api/v1/experiments/{experiment_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["Experiments"]
    )
    async def delete_experiment(experiment_id: str):
        """
        Delete an experiment.
        
        Args:
            experiment_id: Experiment ID
        """
        if experiment_id not in experiments_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        
        del experiments_db[experiment_id]
        logger.info(f"Deleted experiment: {experiment_id}")
    
    # Architecture endpoints
    @app.get(
        "/api/v1/architectures",
        response_model=List[ArchitectureResponse],
        tags=["Architectures"]
    )
    async def list_architectures(
        experiment_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ):
        """
        List architectures.
        
        Args:
            experiment_id: Filter by experiment ID
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of architectures
        """
        architectures = list(architectures_db.values())
        
        if experiment_id:
            architectures = [
                arch for arch in architectures
                if arch.experiment_id == experiment_id
            ]
        
        return architectures[offset:offset + limit]
    
    @app.get(
        "/api/v1/architectures/{architecture_id}",
        response_model=ArchitectureResponse,
        tags=["Architectures"]
    )
    async def get_architecture(architecture_id: str):
        """
        Get architecture details.
        
        Args:
            architecture_id: Architecture ID
            
        Returns:
            Architecture details
        """
        if architecture_id not in architectures_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Architecture {architecture_id} not found"
            )
        
        return architectures_db[architecture_id]
    
    # Optimizer info endpoint
    @app.get(
        "/api/v1/optimizers",
        response_model=List[OptimizerInfo],
        tags=["Optimizers"]
    )
    async def list_optimizers():
        """
        List available optimizers.
        
        Returns:
            List of optimizer information
        """
        return [
            OptimizerInfo(
                name="GeneticAlgorithm",
                type="evolutionary",
                description="Genetic algorithm for neural architecture search",
                parameters={
                    "population_size": {"type": "int", "default": 20},
                    "num_generations": {"type": "int", "default": 50},
                    "mutation_rate": {"type": "float", "default": 0.2},
                    "crossover_rate": {"type": "float", "default": 0.8},
                }
            ),
            OptimizerInfo(
                name="RandomSearch",
                type="random",
                description="Random search baseline",
                parameters={
                    "num_samples": {"type": "int", "default": 100},
                }
            ),
            OptimizerInfo(
                name="HillClimbing",
                type="local_search",
                description="Hill climbing local search",
                parameters={
                    "max_iterations": {"type": "int", "default": 100},
                    "patience": {"type": "int", "default": 10},
                }
            ),
        ]
    
    logger.info("FastAPI application created")
    return app


# Create app instance for uvicorn
app = create_app() if FASTAPI_AVAILABLE else None
