"""Checkpointing for saving and resuming optimization.

Example:
    >>> from morphml.utils import Checkpoint
    >>> 
    >>> # Save
    >>> Checkpoint.save(optimizer, 'checkpoint.json')
    >>> 
    >>> # Load
    >>> optimizer = Checkpoint.load('checkpoint.json', space)
"""

import json
from pathlib import Path
from typing import Any, Dict

from morphml.core.dsl.search_space import SearchSpace
from morphml.core.search import Individual, Population
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class Checkpoint:
    """
    Checkpoint management for optimization.
    
    Saves and loads optimizer state to enable resuming
    long-running searches.
    
    Example:
        >>> # During optimization
        >>> if generation % 10 == 0:
        ...     Checkpoint.save(ga, f'checkpoint_{generation}.json')
        >>> 
        >>> # Resume later
        >>> ga = Checkpoint.load('checkpoint_50.json', search_space)
        >>> ga.optimize(evaluator)  # Continue from generation 50
    """

    @staticmethod
    def save(optimizer: Any, filepath: str) -> None:
        """
        Save optimizer state to file.
        
        Args:
            optimizer: Optimizer instance (GA, RandomSearch, etc.)
            filepath: Path to save checkpoint
        
        Example:
            >>> Checkpoint.save(ga, 'my_checkpoint.json')
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Build checkpoint data
            checkpoint = {
                "optimizer_type": type(optimizer).__name__,
                "config": getattr(optimizer, "config", {}),
            }
            
            # Save optimizer-specific state
            if hasattr(optimizer, "population"):
                # For population-based optimizers (GA, etc.)
                pop = optimizer.population
                checkpoint["population"] = {
                    "generation": pop.generation,
                    "max_size": pop.max_size,
                    "elitism": pop.elitism,
                    "individuals": [ind.to_dict() for ind in pop.individuals],
                }
            
            if hasattr(optimizer, "history"):
                checkpoint["history"] = optimizer.history
            
            if hasattr(optimizer, "best_individual") and optimizer.best_individual:
                checkpoint["best_individual"] = optimizer.best_individual.to_dict()
            
            if hasattr(optimizer, "evaluated"):
                checkpoint["evaluated"] = [ind.to_dict() for ind in optimizer.evaluated]
            
            # Write to file
            with open(filepath, "w") as f:
                json.dump(checkpoint, f, indent=2)
            
            logger.info(f"Checkpoint saved to {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    @staticmethod
    def load(
        filepath: str,
        search_space: SearchSpace,
        optimizer_class: Any = None,
    ) -> Any:
        """
        Load optimizer from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            search_space: SearchSpace instance
            optimizer_class: Optimizer class (auto-detected if None)
        
        Returns:
            Restored optimizer instance
        
        Example:
            >>> from morphml.optimizers import GeneticAlgorithm
            >>> ga = Checkpoint.load('checkpoint.json', space, GeneticAlgorithm)
        """
        try:
            with open(filepath, "r") as f:
                checkpoint = json.load(f)
            
            # Determine optimizer class
            if optimizer_class is None:
                optimizer_type = checkpoint.get("optimizer_type", "GeneticAlgorithm")
                
                # Import appropriate class
                if optimizer_type == "GeneticAlgorithm":
                    from morphml.optimizers import GeneticAlgorithm
                    optimizer_class = GeneticAlgorithm
                elif optimizer_type == "RandomSearch":
                    from morphml.optimizers import RandomSearch
                    optimizer_class = RandomSearch
                elif optimizer_type == "HillClimbing":
                    from morphml.optimizers import HillClimbing
                    optimizer_class = HillClimbing
                else:
                    raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
            # Create optimizer with saved config
            config = checkpoint.get("config", {})
            optimizer = optimizer_class(search_space=search_space, **config)
            
            # Restore population if exists
            if "population" in checkpoint:
                pop_data = checkpoint["population"]
                population = Population(
                    max_size=pop_data["max_size"], elitism=pop_data["elitism"]
                )
                
                # Restore individuals
                for ind_data in pop_data["individuals"]:
                    individual = Individual.from_dict(ind_data)
                    population.add(individual)
                
                population.generation = pop_data["generation"]
                optimizer.population = population
            
            # Restore history
            if "history" in checkpoint:
                optimizer.history = checkpoint["history"]
            
            # Restore best individual
            if "best_individual" in checkpoint:
                optimizer.best_individual = Individual.from_dict(checkpoint["best_individual"])
            
            # Restore evaluated list
            if "evaluated" in checkpoint:
                optimizer.evaluated = [
                    Individual.from_dict(ind_data) for ind_data in checkpoint["evaluated"]
                ]
            
            logger.info(f"Checkpoint loaded from {filepath}")
            return optimizer
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    @staticmethod
    def list_checkpoints(directory: str = ".") -> list:
        """
        List all checkpoint files in directory.
        
        Args:
            directory: Directory to search
        
        Returns:
            List of checkpoint file paths
        """
        dir_path = Path(directory)
        checkpoints = list(dir_path.glob("*.json"))
        return [str(cp) for cp in checkpoints]
