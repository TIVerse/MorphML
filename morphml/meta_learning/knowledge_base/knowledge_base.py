"""Knowledge base for experiment history with vector search.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger
from morphml.meta_learning.experiment_database import TaskMetadata
from morphml.meta_learning.knowledge_base.embedder import ArchitectureEmbedder
from morphml.meta_learning.knowledge_base.meta_features import MetaFeatureExtractor
from morphml.meta_learning.knowledge_base.vector_store import VectorStore

logger = get_logger(__name__)


class KnowledgeBase:
    """
    Complete knowledge base for architecture search history.
    
    Combines:
    - Architecture embedding
    - Meta-feature extraction
    - Vector-based similarity search
    - Persistent storage
    
    Args:
        embedding_dim: Dimension of architecture embeddings
        persist_path: Path for persistent storage
        embedding_method: Method for embedding ('simple' or 'gnn')
    
    Example:
        >>> kb = KnowledgeBase(embedding_dim=128)
        >>> 
        >>> # Add architecture
        >>> kb.add_architecture(
        ...     architecture=graph,
        ...     task=task_metadata,
        ...     metrics={'accuracy': 0.92, 'latency': 0.05}
        ... )
        >>> 
        >>> # Search similar
        >>> similar = kb.search_similar(query_graph, top_k=10)
        >>> 
        >>> # Get best for task
        >>> best = kb.get_best_for_task(task_metadata, top_k=5)
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        persist_path: Optional[str] = None,
        embedding_method: str = "simple",
    ):
        """Initialize knowledge base."""
        self.embedding_dim = embedding_dim
        self.persist_path = persist_path
        
        # Components
        self.embedder = ArchitectureEmbedder(
            method=embedding_method,
            embedding_dim=embedding_dim
        )
        self.meta_extractor = MetaFeatureExtractor()
        self.vector_store = VectorStore(
            embedding_dim=embedding_dim,
            persist_path=persist_path
        )
        
        # Statistics
        self.num_added = 0
        
        logger.info(
            f"Initialized KnowledgeBase "
            f"(dim={embedding_dim}, method={embedding_method})"
        )
    
    def add_architecture(
        self,
        architecture: ModelGraph,
        task: TaskMetadata,
        metrics: Dict[str, float],
        experiment_id: Optional[str] = None,
    ) -> int:
        """
        Add architecture to knowledge base.
        
        Args:
            architecture: Architecture graph
            task: Task metadata
            metrics: Performance metrics (accuracy, latency, etc.)
            experiment_id: Optional experiment identifier
        
        Returns:
            ID of added architecture
        """
        # Generate embedding
        embedding = self.embedder.embed(architecture)
        
        # Extract meta-features
        arch_features = self.meta_extractor.extract_architecture_features(architecture)
        task_features = self.meta_extractor.extract_task_features(task)
        
        # Create metadata
        metadata = {
            'experiment_id': experiment_id or f'exp_{self.num_added}',
            'task_id': task.task_id,
            'dataset': task.dataset_name,
            'num_classes': task.num_classes,
            'problem_type': task.problem_type,
            'metrics': metrics,
            'arch_features': arch_features,
            'task_features': task_features,
            'num_layers': len(architecture.layers),
            'num_parameters': architecture.count_parameters(),
        }
        
        # Add to vector store
        idx = self.vector_store.add(
            embedding=embedding,
            metadata=metadata,
            data=architecture
        )
        
        self.num_added += 1
        
        logger.debug(
            f"Added architecture {idx}: "
            f"task={task.dataset_name}, "
            f"accuracy={metrics.get('accuracy', 0):.4f}"
        )
        
        return idx
    
    def search_similar(
        self,
        query_architecture: ModelGraph,
        top_k: int = 10,
        task_filter: Optional[TaskMetadata] = None,
        metric_threshold: Optional[float] = None,
    ) -> List[Tuple[ModelGraph, float, Dict[str, Any]]]:
        """
        Search for similar architectures.
        
        Args:
            query_architecture: Query architecture
            top_k: Number of results
            task_filter: Optional task to filter by
            metric_threshold: Optional minimum accuracy threshold
        
        Returns:
            List of (architecture, similarity, metadata) tuples
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(query_architecture)
        
        # Define filter function
        def filter_fn(metadata: Dict[str, Any]) -> bool:
            # Filter by task
            if task_filter is not None:
                if metadata['dataset'] != task_filter.dataset_name:
                    return False
            
            # Filter by metric
            if metric_threshold is not None:
                accuracy = metadata['metrics'].get('accuracy', 0)
                if accuracy < metric_threshold:
                    return False
            
            return True
        
        # Search
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_fn=filter_fn if (task_filter or metric_threshold) else None
        )
        
        # Convert distance to similarity
        similar = []
        for idx, distance, metadata, architecture in results:
            # Convert L2 distance to similarity (0-1)
            similarity = 1.0 / (1.0 + distance)
            similar.append((architecture, similarity, metadata))
        
        return similar
    
    def get_best_for_task(
        self,
        task: TaskMetadata,
        top_k: int = 10,
        metric: str = 'accuracy',
    ) -> List[Tuple[ModelGraph, float]]:
        """
        Get best architectures for a specific task.
        
        Args:
            task: Task metadata
            top_k: Number of results
            metric: Metric to sort by
        
        Returns:
            List of (architecture, metric_value) tuples
        """
        # Get all architectures for this task
        all_results = []
        
        for idx in range(self.vector_store.size()):
            _, metadata, architecture = self.vector_store.get(idx)
            
            if metadata['dataset'] == task.dataset_name:
                metric_value = metadata['metrics'].get(metric, 0.0)
                all_results.append((architecture, metric_value, metadata))
        
        # Sort by metric
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return [(arch, metric_val) for arch, metric_val, _ in all_results[:top_k]]
    
    def cluster_tasks(
        self, num_clusters: int = 5
    ) -> Dict[int, List[str]]:
        """
        Cluster tasks by similarity.
        
        Args:
            num_clusters: Number of clusters
        
        Returns:
            Dict mapping cluster ID to list of task IDs
        """
        # Extract all task features
        task_features_list = []
        task_ids = []
        
        for idx in range(self.vector_store.size()):
            _, metadata, _ = self.vector_store.get(idx)
            task_id = metadata['task_id']
            
            if task_id not in task_ids:
                task_ids.append(task_id)
                task_features = metadata['task_features']
                
                # Convert to vector
                feature_vec = self.meta_extractor.feature_vector(task_features)
                task_features_list.append(feature_vec)
        
        if len(task_features_list) < num_clusters:
            logger.warning(
                f"Only {len(task_features_list)} tasks, "
                f"requested {num_clusters} clusters"
            )
            num_clusters = len(task_features_list)
        
        # Simple k-means clustering
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(task_features_list)
        
        # Group by cluster
        clusters = {}
        for task_id, label in zip(task_ids, labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(task_id)
        
        logger.info(f"Clustered {len(task_ids)} tasks into {len(clusters)} clusters")
        
        return clusters
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.
        
        Returns:
            Statistics dictionary
        """
        # Count tasks and datasets
        tasks = set()
        datasets = set()
        
        for idx in range(self.vector_store.size()):
            _, metadata, _ = self.vector_store.get(idx)
            tasks.add(metadata['task_id'])
            datasets.add(metadata['dataset'])
        
        return {
            'num_architectures': self.vector_store.size(),
            'num_tasks': len(tasks),
            'num_datasets': len(datasets),
            'embedding_dim': self.embedding_dim,
        }
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save knowledge base to disk.
        
        Args:
            path: Path to save to (uses persist_path if None)
        """
        save_path = path or self.persist_path
        
        if save_path is None:
            raise ValueError("No save path specified")
        
        self.vector_store.save(save_path)
        logger.info(f"Saved knowledge base to {save_path}")
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load knowledge base from disk.
        
        Args:
            path: Path to load from (uses persist_path if None)
        """
        load_path = path or self.persist_path
        
        if load_path is None:
            raise ValueError("No load path specified")
        
        self.vector_store.load(load_path)
        logger.info(f"Loaded knowledge base from {load_path}")
