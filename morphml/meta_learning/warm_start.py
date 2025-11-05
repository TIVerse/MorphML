"""Warm-starting from past experiments.

Accelerates search by initializing from similar past tasks.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from morphml.core.dsl import SearchSpace
from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger
from morphml.meta_learning.architecture_similarity import (
    ArchitectureSimilarity,
    compute_task_similarity,
)
from morphml.meta_learning.experiment_database import ExperimentDatabase, TaskMetadata

logger = get_logger(__name__)


class WarmStarter:
    """
    Initialize search from past successful experiments.
    
    Strategies:
    1. **Similarity-based**: Find similar past tasks
    2. **Performance-based**: Transfer top-k architectures
    3. **Diversity-aware**: Ensure population diversity
    
    Args:
        knowledge_base: Experiment database
        config: Configuration dict
            - transfer_ratio: Fraction of population from transfer (default: 0.5)
            - min_similarity: Minimum task similarity for transfer (default: 0.6)
            - similarity_method: Task similarity method (default: 'meta_features')
            - diversity_weight: Weight for diversity (default: 0.3)
    
    Example:
        >>> kb = ExperimentDatabase(db_manager)
        >>> warm_starter = WarmStarter(kb, {'transfer_ratio': 0.5})
        >>> 
        >>> current_task = TaskMetadata(
        ...     task_id='new_task',
        ...     dataset_name='CIFAR-100',
        ...     num_samples=50000,
        ...     num_classes=100,
        ...     input_size=(3, 32, 32)
        ... )
        >>> 
        >>> population = warm_starter.generate_initial_population(
        ...     current_task, 
        ...     population_size=50,
        ...     search_space=space
        ... )
    """
    
    def __init__(
        self, knowledge_base: ExperimentDatabase, config: Optional[Dict[str, Any]] = None
    ):
        """Initialize warm-starter."""
        self.kb = knowledge_base
        self.config = config or {}
        
        self.transfer_ratio = self.config.get("transfer_ratio", 0.5)
        self.min_similarity = self.config.get("min_similarity", 0.6)
        self.similarity_method = self.config.get("similarity_method", "meta_features")
        self.diversity_weight = self.config.get("diversity_weight", 0.3)
        
        logger.info(
            f"Initialized WarmStarter (transfer_ratio={self.transfer_ratio}, "
            f"min_similarity={self.min_similarity})"
        )
    
    def generate_initial_population(
        self,
        current_task: TaskMetadata,
        population_size: int,
        search_space: SearchSpace,
    ) -> List[ModelGraph]:
        """
        Generate initial population with warm-starting.
        
        Combines:
        - Transferred architectures from similar tasks
        - Random architectures from search space
        
        Args:
            current_task: Current task metadata
            population_size: Desired population size
            search_space: Search space definition
        
        Returns:
            Initial population of architectures
        """
        logger.info(
            f"Generating initial population for task {current_task.task_id} "
            f"(size={population_size})"
        )
        
        # Find similar past tasks
        similar_tasks = self._find_similar_tasks(current_task)
        
        if not similar_tasks:
            logger.info("No similar tasks found, using pure random initialization")
            return [search_space.sample() for _ in range(population_size)]
        
        # Determine transfer count
        num_transfer = int(population_size * self.transfer_ratio)
        num_random = population_size - num_transfer
        
        logger.info(
            f"Found {len(similar_tasks)} similar tasks, "
            f"transferring {num_transfer} architectures"
        )
        
        # Get best architectures from similar tasks
        transferred = self._transfer_architectures(similar_tasks, num_transfer)
        
        # Fill rest with random
        random_archs = [search_space.sample() for _ in range(num_random)]
        
        # Combine
        population = transferred + random_archs
        
        # Ensure diversity
        if self.diversity_weight > 0:
            population = self._ensure_diversity(population, search_space)
        
        logger.info(
            f"Generated population: {len(transferred)} transferred, "
            f"{len(random_archs)} random"
        )
        
        return population
    
    def _find_similar_tasks(
        self, current_task: TaskMetadata, top_k: int = 5
    ) -> List[Tuple[TaskMetadata, float]]:
        """
        Find similar tasks from knowledge base.
        
        Args:
            current_task: Current task
            top_k: Number of similar tasks to return
        
        Returns:
            List of (task, similarity_score) tuples
        """
        all_tasks = self.kb.get_all_tasks()
        
        if not all_tasks:
            return []
        
        # Compute similarities
        similarities = []
        for task in all_tasks:
            # Skip same task
            if task.task_id == current_task.task_id:
                continue
            
            sim = compute_task_similarity(
                current_task, task, method=self.similarity_method
            )
            
            if sim >= self.min_similarity:
                similarities.append((task, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _transfer_architectures(
        self,
        similar_tasks: List[Tuple[TaskMetadata, float]],
        num_transfer: int,
    ) -> List[ModelGraph]:
        """
        Transfer best architectures from similar tasks.
        
        Architectures are weighted by task similarity.
        
        Args:
            similar_tasks: List of (task, similarity) tuples
            num_transfer: Number of architectures to transfer
        
        Returns:
            List of transferred architectures
        """
        transferred = []
        
        # Calculate weights
        total_similarity = sum(sim for _, sim in similar_tasks)
        
        for task, similarity in similar_tasks:
            # Number to transfer from this task (weighted by similarity)
            weight = similarity / total_similarity
            num_from_task = max(1, int(num_transfer * weight))
            
            # Get top architectures from this task
            top_archs = self.kb.get_top_architectures(task.task_id, top_k=20)
            
            if not top_archs:
                continue
            
            # Sample architectures
            num_sample = min(num_from_task, len(top_archs))
            sampled = random.sample(top_archs, num_sample)
            
            transferred.extend(sampled)
            
            logger.debug(
                f"Transferred {num_sample} architectures from {task.task_id} "
                f"(similarity={similarity:.3f})"
            )
            
            if len(transferred) >= num_transfer:
                break
        
        # Trim to exact size
        return transferred[:num_transfer]
    
    def _ensure_diversity(
        self, population: List[ModelGraph], search_space: SearchSpace
    ) -> List[ModelGraph]:
        """
        Ensure population diversity.
        
        Replaces very similar architectures with random ones.
        
        Args:
            population: Current population
            search_space: Search space for sampling replacements
        
        Returns:
            Diversified population
        """
        if len(population) < 2:
            return population
        
        # Compute pairwise similarities
        n = len(population)
        similarities = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = ArchitectureSimilarity.compute(
                    population[i], population[j], method="operation_distribution"
                )
                similarities[i, j] = sim
                similarities[j, i] = sim
        
        # Find very similar pairs (threshold = 0.95)
        threshold = 0.95
        replaced = set()
        
        for i in range(n):
            if i in replaced:
                continue
            
            for j in range(i + 1, n):
                if j in replaced:
                    continue
                
                if similarities[i, j] > threshold:
                    # Replace j with random
                    population[j] = search_space.sample()
                    replaced.add(j)
        
        if replaced:
            logger.info(f"Replaced {len(replaced)} similar architectures for diversity")
        
        return population
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get warm-starting statistics."""
        kb_stats = self.kb.get_statistics()
        
        return {
            "knowledge_base": kb_stats,
            "config": {
                "transfer_ratio": self.transfer_ratio,
                "min_similarity": self.min_similarity,
                "similarity_method": self.similarity_method,
            },
        }
