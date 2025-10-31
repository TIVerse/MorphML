# Component 1: Warm-Starting from Past Experiments

**Duration:** Weeks 1-2  
**LOC Target:** ~3,000  
**Dependencies:** Phase 1-3 complete

---

## 🎯 Objective

Accelerate search by leveraging past experiments:
1. **Architecture Similarity** - Measure distance between architectures
2. **Transfer from Related Tasks** - Initialize from similar datasets
3. **Smart Initial Population** - Seed with promising architectures
4. **Target:** 30%+ reduction in search time

---

## 📋 Files to Create

### 1. `meta_learning/warm_start.py` (~1,500 LOC)

```python
from typing import List, Dict, Optional
import numpy as np

class WarmStarter:
    """
    Initialize search from past successful experiments.
    
    Strategies:
    1. Similarity-based: Find similar past tasks
    2. Performance-based: Transfer top-k architectures
    3. Diversity-aware: Ensure transferred population is diverse
    
    Config:
        knowledge_base: Path to experiment database
        similarity_metric: 'dataset_meta', 'task_embedding'
        transfer_ratio: Fraction of initial population from transfer (default: 0.5)
        min_similarity: Minimum similarity for transfer (default: 0.6)
    """
    
    def __init__(self, knowledge_base: ExperimentDatabase, config: Dict[str, Any]):
        self.kb = knowledge_base
        self.transfer_ratio = config.get('transfer_ratio', 0.5)
        self.min_similarity = config.get('min_similarity', 0.6)
        self.similarity_metric = config.get('similarity_metric', 'dataset_meta')
    
    def generate_initial_population(
        self,
        current_task: TaskMetadata,
        population_size: int,
        search_space: SearchSpace
    ) -> List[ModelGraph]:
        """
        Generate initial population with warm-starting.
        
        Args:
            current_task: Metadata for current task (dataset, problem type)
            population_size: Size of initial population
            search_space: Search space definition
        
        Returns:
            List of architectures (mix of transferred + random)
        """
        # Find similar past tasks
        similar_tasks = self._find_similar_tasks(current_task)
        
        if not similar_tasks:
            logger.info("No similar tasks found, using random initialization")
            return [search_space.sample() for _ in range(population_size)]
        
        # Determine how many to transfer
        num_transfer = int(population_size * self.transfer_ratio)
        num_random = population_size - num_transfer
        
        # Get best architectures from similar tasks
        transferred = self._transfer_architectures(similar_tasks, num_transfer)
        
        # Fill rest with random
        random_archs = [search_space.sample() for _ in range(num_random)]
        
        population = transferred + random_archs
        
        logger.info(
            f"Warm-started population: {num_transfer} transferred, "
            f"{num_random} random"
        )
        
        return population
    
    def _find_similar_tasks(
        self,
        current_task: TaskMetadata,
        top_k: int = 5
    ) -> List[Tuple[TaskMetadata, float]]:
        """
        Find similar tasks from knowledge base.
        
        Returns:
            List of (task, similarity_score) tuples
        """
        all_tasks = self.kb.get_all_tasks()
        
        # Compute similarities
        similarities = []
        for task in all_tasks:
            sim = self._compute_task_similarity(current_task, task)
            if sim >= self.min_similarity:
                similarities.append((task, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _compute_task_similarity(
        self,
        task1: TaskMetadata,
        task2: TaskMetadata
    ) -> float:
        """
        Compute similarity between tasks.
        
        Features:
        - Dataset size
        - Number of classes
        - Input resolution
        - Problem type (classification, detection, etc.)
        """
        if self.similarity_metric == 'dataset_meta':
            # Simple meta-feature similarity
            features1 = np.array([
                task1.num_samples / 100000,
                task1.num_classes / 100,
                task1.input_size[0] / 224,
                task1.input_size[1] / 224
            ])
            
            features2 = np.array([
                task2.num_samples / 100000,
                task2.num_classes / 100,
                task2.input_size[0] / 224,
                task2.input_size[1] / 224
            ])
            
            # Cosine similarity
            similarity = np.dot(features1, features2) / (
                np.linalg.norm(features1) * np.linalg.norm(features2) + 1e-8
            )
            
            return max(0.0, similarity)
        
        elif self.similarity_metric == 'task_embedding':
            # Use learned task embeddings
            emb1 = self._get_task_embedding(task1)
            emb2 = self._get_task_embedding(task2)
            
            return cosine_similarity(emb1, emb2)
    
    def _transfer_architectures(
        self,
        similar_tasks: List[Tuple[TaskMetadata, float]],
        num_transfer: int
    ) -> List[ModelGraph]:
        """
        Transfer best architectures from similar tasks.
        
        Weighted by task similarity.
        """
        transferred = []
        
        for task, similarity in similar_tasks:
            # Get top architectures from this task
            top_archs = self.kb.get_top_architectures(task.task_id, top_k=10)
            
            # Weight by similarity
            num_from_task = max(1, int(num_transfer * similarity / sum(s for _, s in similar_tasks)))
            
            # Sample architectures
            sampled = random.sample(top_archs, min(num_from_task, len(top_archs)))
            transferred.extend(sampled)
            
            if len(transferred) >= num_transfer:
                break
        
        return transferred[:num_transfer]


@dataclass
class TaskMetadata:
    """Metadata describing a task."""
    task_id: str
    dataset_name: str
    num_samples: int
    num_classes: int
    input_size: Tuple[int, int, int]  # (C, H, W)
    problem_type: str  # 'classification', 'detection', etc.
```

---

### 2. `meta_learning/architecture_similarity.py` (~1,000 LOC)

```python
class ArchitectureSimilarity:
    """
    Measure similarity between neural architectures.
    
    Methods:
    1. Graph Edit Distance
    2. Path Encoding Similarity
    3. Operation Distribution
    """
    
    @staticmethod
    def graph_edit_distance(graph1: ModelGraph, graph2: ModelGraph) -> float:
        """
        Compute graph edit distance (normalized).
        
        Edit operations:
        - Node insertion/deletion
        - Edge insertion/deletion
        - Node label change
        """
        import networkx as nx
        from networkx.algorithms import graph_edit_distance
        
        G1 = graph1.to_networkx()
        G2 = graph2.to_networkx()
        
        # Compute GED (expensive for large graphs)
        ged = nx.graph_edit_distance(G1, G2, timeout=10)
        
        # Normalize by graph sizes
        max_size = max(len(G1.nodes), len(G2.nodes))
        normalized_ged = ged / max_size if max_size > 0 else 0.0
        
        # Convert to similarity (0 = identical, 1 = completely different)
        similarity = 1.0 - min(normalized_ged, 1.0)
        
        return similarity
    
    @staticmethod
    def operation_distribution_similarity(
        graph1: ModelGraph,
        graph2: ModelGraph
    ) -> float:
        """
        Compare operation type distributions.
        
        Faster than GED but less precise.
        """
        ops1 = [node.operation for node in graph1.nodes.values()]
        ops2 = [node.operation for node in graph2.nodes.values()]
        
        # Count operation types
        from collections import Counter
        count1 = Counter(ops1)
        count2 = Counter(ops2)
        
        # All operation types
        all_ops = set(count1.keys()) | set(count2.keys())
        
        # Create normalized vectors
        vec1 = np.array([count1.get(op, 0) for op in all_ops])
        vec2 = np.array([count2.get(op, 0) for op in all_ops])
        
        vec1 = vec1 / (vec1.sum() + 1e-8)
        vec2 = vec2 / (vec2.sum() + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8
        )
        
        return similarity
```

---

### 3. `meta_learning/experiment_database.py` (~500 LOC)

```python
class ExperimentDatabase:
    """
    Database of past experiments for meta-learning.
    
    Stores:
    - Task metadata
    - Evaluated architectures
    - Performance metrics
    - Search trajectories
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_all_tasks(self) -> List[TaskMetadata]:
        """Get all past tasks."""
        # Query from database
        tasks = self.db.session.query(Experiment).all()
        
        return [self._experiment_to_task_metadata(exp) for exp in tasks]
    
    def get_top_architectures(
        self,
        task_id: str,
        top_k: int = 10
    ) -> List[ModelGraph]:
        """Get top-k architectures for a task."""
        archs = self.db.session.query(Architecture).filter_by(
            experiment_id=task_id
        ).order_by(Architecture.fitness.desc()).limit(top_k).all()
        
        return [ModelGraph.from_dict(a.architecture_json) for a in archs]
```

---

## 🧪 Tests

```python
def test_warm_starting():
    """Test warm-start initialization."""
    kb = ExperimentDatabase(...)
    
    # Add past experiment
    past_task = TaskMetadata(
        task_id='exp1',
        dataset_name='CIFAR-10',
        num_samples=50000,
        num_classes=10,
        input_size=(3, 32, 32),
        problem_type='classification'
    )
    
    # Current similar task
    current_task = TaskMetadata(
        task_id='exp2',
        dataset_name='CIFAR-100',
        num_samples=50000,
        num_classes=100,
        input_size=(3, 32, 32),
        problem_type='classification'
    )
    
    warm_starter = WarmStarter(kb, {})
    population = warm_starter.generate_initial_population(
        current_task,
        population_size=50,
        search_space=SearchSpace(...)
    )
    
    assert len(population) == 50
```

---

## ✅ Deliverables

- [ ] Warm-starting implementation
- [ ] Task similarity metrics
- [ ] Architecture similarity metrics
- [ ] Experiment database query interface
- [ ] Tests showing 30%+ speedup

---

**Next:** `02_performance_prediction.md`
