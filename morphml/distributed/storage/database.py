"""PostgreSQL database backend for experiment results.

Stores structured experiment data with SQL queries for analysis.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import datetime
import hashlib
import json
from typing import Any, Dict, List, Optional

try:
    from sqlalchemy import (
        Column,
        DateTime,
        Float,
        Integer,
        String,
        Text,
        create_engine,
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import Session, sessionmaker
    
    SQLALCHEMY_AVAILABLE = True
    Base = declarative_base()
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Base = object  # type: ignore

from morphml.core.graph import ModelGraph
from morphml.exceptions import DistributedError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


if SQLALCHEMY_AVAILABLE:
    
    class Experiment(Base):  # type: ignore
        """
        Experiment table.
        
        Stores high-level experiment metadata.
        """
        
        __tablename__ = "experiments"
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        name = Column(String(255), unique=True, nullable=False, index=True)
        search_space = Column(Text)  # JSON string
        optimizer = Column(String(100))
        config = Column(Text)  # JSON string
        status = Column(String(50), default="running")  # running, completed, failed
        num_evaluations = Column(Integer, default=0)
        best_fitness = Column(Float)
        created_at = Column(DateTime, default=datetime.datetime.utcnow)
        updated_at = Column(
            DateTime,
            default=datetime.datetime.utcnow,
            onupdate=datetime.datetime.utcnow,
        )
        completed_at = Column(DateTime)
    
    class Architecture(Base):  # type: ignore
        """
        Architecture evaluation table.
        
        Stores individual architecture evaluations.
        """
        
        __tablename__ = "architectures"
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        experiment_id = Column(Integer, nullable=False, index=True)
        architecture_hash = Column(String(64), unique=True, nullable=False, index=True)
        architecture_json = Column(Text, nullable=False)
        fitness = Column(Float, nullable=False, index=True)
        metrics = Column(Text)  # JSON string
        generation = Column(Integer, default=0, index=True)
        worker_id = Column(String(100))
        duration = Column(Float)  # seconds
        evaluated_at = Column(DateTime, default=datetime.datetime.utcnow)

else:
    # Dummy classes when SQLAlchemy not available
    class Experiment:  # type: ignore
        pass
    
    class Architecture:  # type: ignore
        pass


class DatabaseManager:
    """
    Manage PostgreSQL database for experiments.
    
    Provides persistent storage for experiment results with SQL queries.
    
    Args:
        connection_string: PostgreSQL connection string
            Format: postgresql://user:password@host:port/database
        pool_size: Connection pool size (default: 20)
        echo: Echo SQL queries for debugging (default: False)
    
    Example:
        >>> db = DatabaseManager('postgresql://localhost/morphml')
        >>> exp_id = db.create_experiment('cifar10_search', {})
        >>> db.save_architecture(exp_id, graph, 0.95, metrics, 1, 'worker-1')
        >>> best = db.get_best_architectures(exp_id, top_k=10)
    """
    
    def __init__(
        self,
        connection_string: str,
        pool_size: int = 20,
        echo: bool = False,
    ):
        """Initialize database manager."""
        if not SQLALCHEMY_AVAILABLE:
            raise DistributedError(
                "SQLAlchemy not available. Install with: pip install sqlalchemy psycopg2-binary"
            )
        
        self.connection_string = connection_string
        self.engine = create_engine(
            connection_string, pool_size=pool_size, echo=echo
        )
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.Session = sessionmaker(bind=self.engine)
        
        logger.info(f"Connected to database: {connection_string.split('@')[-1]}")
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.Session()
    
    def create_experiment(
        self, name: str, config: Dict[str, Any]
    ) -> int:
        """
        Create new experiment.
        
        Args:
            name: Experiment name (must be unique)
            config: Experiment configuration
                - search_space: Search space definition
                - optimizer: Optimizer name
                - Any other config fields
        
        Returns:
            Experiment ID
        """
        session = self.get_session()
        
        try:
            exp = Experiment(
                name=name,
                search_space=json.dumps(config.get("search_space", {})),
                optimizer=config.get("optimizer", "unknown"),
                config=json.dumps(config),
                status="running",
            )
            
            session.add(exp)
            session.commit()
            
            exp_id = exp.id
            
            logger.info(f"Created experiment '{name}' with ID {exp_id}")
            
            return exp_id
        
        finally:
            session.close()
    
    def save_architecture(
        self,
        experiment_id: int,
        architecture: ModelGraph,
        fitness: float,
        metrics: Dict[str, float],
        generation: int = 0,
        worker_id: str = "unknown",
        duration: float = 0.0,
    ) -> int:
        """
        Save architecture evaluation.
        
        Args:
            experiment_id: Experiment ID
            architecture: ModelGraph
            fitness: Fitness score
            metrics: Additional metrics (accuracy, params, etc.)
            generation: Generation number
            worker_id: Worker that evaluated
            duration: Evaluation duration (seconds)
        
        Returns:
            Architecture ID
        """
        session = self.get_session()
        
        try:
            # Compute architecture hash
            arch_hash = self._hash_architecture(architecture)
            
            # Check if already evaluated (deduplication)
            existing = (
                session.query(Architecture)
                .filter_by(architecture_hash=arch_hash)
                .first()
            )
            
            if existing:
                logger.debug(f"Architecture {arch_hash[:12]} already evaluated")
                return existing.id
            
            # Save new architecture
            arch = Architecture(
                experiment_id=experiment_id,
                architecture_hash=arch_hash,
                architecture_json=architecture.to_json(),
                fitness=fitness,
                metrics=json.dumps(metrics),
                generation=generation,
                worker_id=worker_id,
                duration=duration,
            )
            
            session.add(arch)
            session.commit()
            
            # Update experiment stats
            self._update_experiment_stats(experiment_id, fitness)
            
            logger.debug(
                f"Saved architecture {arch_hash[:12]}: "
                f"fitness={fitness:.4f}, generation={generation}"
            )
            
            return arch.id
        
        finally:
            session.close()
    
    def get_best_architectures(
        self, experiment_id: int, top_k: int = 10
    ) -> List[Architecture]:
        """
        Get top-k architectures by fitness.
        
        Args:
            experiment_id: Experiment ID
            top_k: Number of architectures to return
        
        Returns:
            List of Architecture objects
        """
        session = self.get_session()
        
        try:
            architectures = (
                session.query(Architecture)
                .filter_by(experiment_id=experiment_id)
                .order_by(Architecture.fitness.desc())
                .limit(top_k)
                .all()
            )
            
            return architectures
        
        finally:
            session.close()
    
    def get_architecture_by_hash(
        self, arch_hash: str
    ) -> Optional[Architecture]:
        """
        Get architecture by hash (for deduplication).
        
        Args:
            arch_hash: Architecture hash
        
        Returns:
            Architecture object or None
        """
        session = self.get_session()
        
        try:
            return (
                session.query(Architecture)
                .filter_by(architecture_hash=arch_hash)
                .first()
            )
        
        finally:
            session.close()
    
    def get_experiment(self, experiment_id: int) -> Optional[Experiment]:
        """
        Get experiment by ID.
        
        Args:
            experiment_id: Experiment ID
        
        Returns:
            Experiment object or None
        """
        session = self.get_session()
        
        try:
            return session.query(Experiment).filter_by(id=experiment_id).first()
        
        finally:
            session.close()
    
    def get_experiment_by_name(self, name: str) -> Optional[Experiment]:
        """
        Get experiment by name.
        
        Args:
            name: Experiment name
        
        Returns:
            Experiment object or None
        """
        session = self.get_session()
        
        try:
            return session.query(Experiment).filter_by(name=name).first()
        
        finally:
            session.close()
    
    def update_experiment_status(
        self, experiment_id: int, status: str
    ) -> None:
        """
        Update experiment status.
        
        Args:
            experiment_id: Experiment ID
            status: New status ('running', 'completed', 'failed')
        """
        session = self.get_session()
        
        try:
            exp = session.query(Experiment).filter_by(id=experiment_id).first()
            
            if exp:
                exp.status = status
                
                if status == "completed":
                    exp.completed_at = datetime.datetime.utcnow()
                
                session.commit()
                
                logger.info(f"Experiment {experiment_id} status: {status}")
        
        finally:
            session.close()
    
    def get_experiment_statistics(
        self, experiment_id: int
    ) -> Dict[str, Any]:
        """
        Get experiment statistics.
        
        Args:
            experiment_id: Experiment ID
        
        Returns:
            Statistics dictionary
        """
        session = self.get_session()
        
        try:
            exp = session.query(Experiment).filter_by(id=experiment_id).first()
            
            if not exp:
                return {}
            
            architectures = (
                session.query(Architecture)
                .filter_by(experiment_id=experiment_id)
                .all()
            )
            
            if not architectures:
                return {
                    "experiment_id": experiment_id,
                    "num_evaluations": 0,
                    "status": exp.status,
                }
            
            fitnesses = [a.fitness for a in architectures]
            
            return {
                "experiment_id": experiment_id,
                "name": exp.name,
                "status": exp.status,
                "num_evaluations": len(architectures),
                "best_fitness": max(fitnesses),
                "avg_fitness": sum(fitnesses) / len(fitnesses),
                "min_fitness": min(fitnesses),
                "created_at": exp.created_at.isoformat() if exp.created_at else None,
                "updated_at": exp.updated_at.isoformat() if exp.updated_at else None,
            }
        
        finally:
            session.close()
    
    def _hash_architecture(self, architecture: ModelGraph) -> str:
        """
        Compute unique hash for architecture.
        
        Uses SHA256 of JSON representation.
        """
        arch_json = architecture.to_json()
        return hashlib.sha256(arch_json.encode()).hexdigest()
    
    def _update_experiment_stats(
        self, experiment_id: int, new_fitness: float
    ) -> None:
        """Update experiment statistics."""
        session = self.get_session()
        
        try:
            exp = session.query(Experiment).filter_by(id=experiment_id).first()
            
            if exp:
                exp.num_evaluations = (exp.num_evaluations or 0) + 1
                
                if exp.best_fitness is None or new_fitness > exp.best_fitness:
                    exp.best_fitness = new_fitness
                
                session.commit()
        
        finally:
            session.close()
    
    def list_experiments(self) -> List[Experiment]:
        """
        List all experiments.
        
        Returns:
            List of Experiment objects
        """
        session = self.get_session()
        
        try:
            return session.query(Experiment).order_by(Experiment.created_at.desc()).all()
        
        finally:
            session.close()
    
    def delete_experiment(self, experiment_id: int) -> None:
        """
        Delete experiment and all associated architectures.
        
        Args:
            experiment_id: Experiment ID
        """
        session = self.get_session()
        
        try:
            # Delete architectures
            session.query(Architecture).filter_by(
                experiment_id=experiment_id
            ).delete()
            
            # Delete experiment
            session.query(Experiment).filter_by(id=experiment_id).delete()
            
            session.commit()
            
            logger.info(f"Deleted experiment {experiment_id}")
        
        finally:
            session.close()
