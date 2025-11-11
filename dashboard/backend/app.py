"""MorphML Dashboard Backend - FastAPI with WebSocket support.

This module provides the complete backend for the MorphML web dashboard,
including REST API endpoints and WebSocket for real-time updates.

Run:
    uvicorn dashboard.backend.app:app --reload --port 8000
"""

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import uuid
import json

from morphml.logging_config import get_logger
from morphml.version import __version__

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MorphML Dashboard API",
    description="Real-time Neural Architecture Search Dashboard",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (replace with database in production)
experiments_db: Dict[str, Dict[str, Any]] = {}
architectures_db: Dict[str, Dict[str, Any]] = {}
experiment_progress: Dict[str, Dict[str, Any]] = {}

# WebSocket connection manager
class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, experiment_id: str):
        """Accept and store WebSocket connection."""
        await websocket.accept()
        if experiment_id not in self.active_connections:
            self.active_connections[experiment_id] = []
        self.active_connections[experiment_id].append(websocket)
        logger.info(f"WebSocket connected for experiment {experiment_id}")
    
    def disconnect(self, websocket: WebSocket, experiment_id: str):
        """Remove WebSocket connection."""
        if experiment_id in self.active_connections:
            self.active_connections[experiment_id].remove(websocket)
            logger.info(f"WebSocket disconnected for experiment {experiment_id}")
    
    async def broadcast(self, experiment_id: str, message: dict):
        """Broadcast message to all connections for an experiment."""
        if experiment_id in self.active_connections:
            for connection in self.active_connections[experiment_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to WebSocket: {e}")

manager = ConnectionManager()


# Pydantic Models
class ExperimentCreate(BaseModel):
    """Request model for creating an experiment."""
    name: str = Field(..., description="Experiment name")
    search_space: Dict[str, Any] = Field(..., description="Search space configuration")
    optimizer: str = Field("genetic", description="Optimizer type")
    budget: int = Field(100, description="Evaluation budget")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional config")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "CIFAR-10 Search",
                "search_space": {
                    "layers": [
                        {"type": "input", "shape": [3, 32, 32]},
                        {"type": "conv2d", "filters": [32, 64], "kernel_size": 3},
                        {"type": "flatten"},
                        {"type": "dense", "units": [128, 256]},
                    ]
                },
                "optimizer": "genetic",
                "budget": 1000,
                "config": {
                    "population_size": 20,
                    "num_generations": 50
                }
            }
        }


class ExperimentResponse(BaseModel):
    """Response model for experiment."""
    id: str
    name: str
    status: str  # created, running, completed, failed, stopped
    best_accuracy: Optional[float] = None
    generation: int = 0
    total_generations: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class ArchitectureResponse(BaseModel):
    """Response model for architecture."""
    id: str
    experiment_id: str
    fitness: float
    metrics: Dict[str, Any]
    parameters: int
    depth: int
    nodes: int
    created_at: str


class ProgressUpdate(BaseModel):
    """Progress update for WebSocket."""
    experiment_id: str
    generation: int
    total_generations: int
    best_fitness: float
    current_fitness: float
    diversity: float
    timestamp: str
    history: List[float]


# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/v1/experiments", response_model=ExperimentResponse, status_code=201)
async def create_experiment(experiment: ExperimentCreate):
    """
    Create a new NAS experiment.
    
    Args:
        experiment: Experiment configuration
        
    Returns:
        Created experiment details
    """
    experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
    
    total_gens = experiment.config.get("num_generations", 50)
    
    exp_data = {
        "id": experiment_id,
        "name": experiment.name,
        "status": "created",
        "best_accuracy": None,
        "generation": 0,
        "total_generations": total_gens,
        "created_at": datetime.utcnow().isoformat(),
        "started_at": None,
        "completed_at": None,
        "search_space": experiment.search_space,
        "optimizer": experiment.optimizer,
        "budget": experiment.budget,
        "config": experiment.config,
    }
    
    experiments_db[experiment_id] = exp_data
    experiment_progress[experiment_id] = {
        "history": [],
        "current_generation": 0,
    }
    
    logger.info(f"Created experiment: {experiment_id} - {experiment.name}")
    
    return ExperimentResponse(**{k: v for k, v in exp_data.items() if k in ExperimentResponse.__fields__})


@app.get("/api/v1/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List all experiments.
    
    Args:
        status: Filter by status
        limit: Maximum number of results
        offset: Offset for pagination
        
    Returns:
        List of experiments
    """
    experiments = list(experiments_db.values())
    
    if status:
        experiments = [exp for exp in experiments if exp["status"] == status]
    
    experiments = experiments[offset:offset + limit]
    
    return [
        ExperimentResponse(**{k: v for k, v in exp.items() if k in ExperimentResponse.__fields__})
        for exp in experiments
    ]


@app.get("/api/v1/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str):
    """
    Get experiment details.
    
    Args:
        experiment_id: Experiment ID
        
    Returns:
        Experiment details
    """
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    exp = experiments_db[experiment_id]
    return ExperimentResponse(**{k: v for k, v in exp.items() if k in ExperimentResponse.__fields__})


@app.get("/api/v1/experiments/{experiment_id}/details")
async def get_experiment_details(experiment_id: str):
    """
    Get detailed experiment information including architectures.
    
    Args:
        experiment_id: Experiment ID
        
    Returns:
        Detailed experiment data
    """
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    exp = experiments_db[experiment_id]
    
    # Get architectures for this experiment
    exp_architectures = [
        arch for arch in architectures_db.values()
        if arch["experiment_id"] == experiment_id
    ]
    
    # Sort by fitness
    exp_architectures.sort(key=lambda x: x["fitness"], reverse=True)
    
    return {
        **exp,
        "architectures": exp_architectures[:10],  # Top 10
        "progress": experiment_progress.get(experiment_id, {})
    }


@app.post("/api/v1/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str, background_tasks: BackgroundTasks):
    """
    Start experiment execution.
    
    Args:
        experiment_id: Experiment ID
        background_tasks: FastAPI background tasks
        
    Returns:
        Status message
    """
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    exp = experiments_db[experiment_id]
    
    if exp["status"] == "running":
        raise HTTPException(status_code=400, detail="Experiment already running")
    
    exp["status"] = "running"
    exp["started_at"] = datetime.utcnow().isoformat()
    
    # Add background task to run experiment
    background_tasks.add_task(run_experiment, experiment_id)
    
    logger.info(f"Started experiment: {experiment_id}")
    
    return {"status": "started", "experiment_id": experiment_id}


@app.post("/api/v1/experiments/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """
    Stop a running experiment.
    
    Args:
        experiment_id: Experiment ID
        
    Returns:
        Status message
    """
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    exp = experiments_db[experiment_id]
    
    if exp["status"] != "running":
        raise HTTPException(status_code=400, detail="Experiment not running")
    
    exp["status"] = "stopped"
    exp["completed_at"] = datetime.utcnow().isoformat()
    
    logger.info(f"Stopped experiment: {experiment_id}")
    
    return {"status": "stopped", "experiment_id": experiment_id}


@app.delete("/api/v1/experiments/{experiment_id}", status_code=204)
async def delete_experiment(experiment_id: str):
    """
    Delete an experiment.
    
    Args:
        experiment_id: Experiment ID
    """
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    del experiments_db[experiment_id]
    if experiment_id in experiment_progress:
        del experiment_progress[experiment_id]
    
    # Delete associated architectures
    arch_ids_to_delete = [
        arch_id for arch_id, arch in architectures_db.items()
        if arch["experiment_id"] == experiment_id
    ]
    for arch_id in arch_ids_to_delete:
        del architectures_db[arch_id]
    
    logger.info(f"Deleted experiment: {experiment_id}")


@app.get("/api/v1/architectures", response_model=List[ArchitectureResponse])
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
        architectures = [arch for arch in architectures if arch["experiment_id"] == experiment_id]
    
    # Sort by fitness
    architectures.sort(key=lambda x: x["fitness"], reverse=True)
    
    architectures = architectures[offset:offset + limit]
    
    return [ArchitectureResponse(**arch) for arch in architectures]


@app.get("/api/v1/architectures/{architecture_id}", response_model=ArchitectureResponse)
async def get_architecture(architecture_id: str):
    """
    Get architecture details.
    
    Args:
        architecture_id: Architecture ID
        
    Returns:
        Architecture details
    """
    if architecture_id not in architectures_db:
        raise HTTPException(status_code=404, detail=f"Architecture {architecture_id} not found")
    
    return ArchitectureResponse(**architectures_db[architecture_id])


# WebSocket endpoint for real-time updates
@app.websocket("/api/v1/stream/{experiment_id}")
async def websocket_endpoint(websocket: WebSocket, experiment_id: str):
    """
    WebSocket endpoint for real-time experiment updates.
    
    Args:
        websocket: WebSocket connection
        experiment_id: Experiment ID to monitor
    """
    await manager.connect(websocket, experiment_id)
    
    try:
        while True:
            # Send progress updates
            if experiment_id in experiment_progress:
                progress = experiment_progress[experiment_id]
                exp = experiments_db.get(experiment_id, {})
                
                update = {
                    "experiment_id": experiment_id,
                    "generation": progress.get("current_generation", 0),
                    "total_generations": exp.get("total_generations", 50),
                    "best_fitness": exp.get("best_accuracy", 0.0),
                    "history": progress.get("history", []),
                    "status": exp.get("status", "unknown"),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await websocket.send_json(update)
            
            await asyncio.sleep(1)  # Update every second
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, experiment_id)
        logger.info(f"WebSocket disconnected for experiment {experiment_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, experiment_id)


# Background task to simulate experiment execution
async def run_experiment(experiment_id: str):
    """
    Run experiment in background (simulated for demo).
    
    In production, this would:
    1. Load experiment config
    2. Create search space
    3. Initialize optimizer
    4. Run optimization loop
    5. Save results
    
    Args:
        experiment_id: Experiment ID
    """
    import random
    
    exp = experiments_db[experiment_id]
    total_gens = exp["total_generations"]
    
    logger.info(f"Running experiment {experiment_id} for {total_gens} generations")
    
    try:
        for gen in range(total_gens):
            # Check if stopped
            if exp["status"] != "running":
                break
            
            # Simulate generation
            await asyncio.sleep(0.5)  # Simulate work
            
            # Update progress
            best_fitness = 0.5 + (gen / total_gens) * 0.4 + random.uniform(-0.05, 0.05)
            best_fitness = min(0.99, max(0.0, best_fitness))
            
            exp["generation"] = gen + 1
            exp["best_accuracy"] = best_fitness
            
            experiment_progress[experiment_id]["current_generation"] = gen + 1
            experiment_progress[experiment_id]["history"].append(best_fitness)
            
            # Create architecture (simulated)
            if random.random() > 0.7:  # 30% chance to save architecture
                arch_id = f"arch_{uuid.uuid4().hex[:8]}"
                architectures_db[arch_id] = {
                    "id": arch_id,
                    "experiment_id": experiment_id,
                    "fitness": best_fitness,
                    "metrics": {
                        "accuracy": best_fitness,
                        "loss": 1.0 - best_fitness,
                    },
                    "parameters": random.randint(100000, 5000000),
                    "depth": random.randint(5, 20),
                    "nodes": random.randint(8, 30),
                    "created_at": datetime.utcnow().isoformat()
                }
            
            # Broadcast update
            await manager.broadcast(experiment_id, {
                "generation": gen + 1,
                "best_fitness": best_fitness,
                "status": "running"
            })
        
        # Mark as completed
        exp["status"] = "completed"
        exp["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Completed experiment {experiment_id}")
        
    except Exception as e:
        logger.error(f"Error running experiment {experiment_id}: {e}")
        exp["status"] = "failed"
        exp["completed_at"] = datetime.utcnow().isoformat()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
