# Component 3: Knowledge Base & Retrieval

**Duration:** Week 5  
**LOC Target:** ~3,000  
**Dependencies:** Components 1-2

---

## 🎯 Objective

Build searchable knowledge base of experiments:
1. **Vector Database** - Store architecture embeddings
2. **Similarity Search** - Fast retrieval of similar architectures
3. **Meta-features** - Extract task characteristics
4. **Indexing** - Efficient storage and querying

---

## 📋 Files to Create

### 1. `meta_learning/knowledge_base.py` (~1,500 LOC)

```python
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional

class KnowledgeBase:
    """
    Vector database for architecture search history.
    
    Uses ChromaDB for storing and querying architecture embeddings.
    
    Features:
    - Store architecture graphs with metadata
    - Fast similarity search
    - Filter by task characteristics
    - Update with new experiments
    
    Usage:
        kb = KnowledgeBase()
        kb.add_architecture(graph, metrics, task_metadata)
        similar = kb.search_similar(query_graph, top_k=10)
    """
    
    def __init__(self, persist_directory: str = "./morphml_kb"):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        
        # Create collection
        self.collection = self.client.get_or_create_collection(
            name="architectures",
            metadata={"description": "Neural architecture search results"}
        )
        
        # Embedding model
        self.embedding_model = ArchitectureEmbedder()
    
    def add_architecture(
        self,
        architecture: ModelGraph,
        metrics: Dict[str, float],
        task: TaskMetadata,
        experiment_id: str
    ):
        """
        Add architecture to knowledge base.
        
        Args:
            architecture: Architecture graph
            metrics: Performance metrics (accuracy, latency, etc.)
            task: Task metadata
            experiment_id: Experiment identifier
        """
        # Generate embedding
        embedding = self.embedding_model.embed(architecture)
        
        # Create metadata
        metadata = {
            'experiment_id': experiment_id,
            'dataset': task.dataset_name,
            'num_classes': task.num_classes,
            'accuracy': metrics.get('accuracy', 0.0),
            'latency': metrics.get('latency', 0.0),
            'params': metrics.get('params', 0.0),
            'num_nodes': len(architecture.nodes),
            'num_edges': len(architecture.edges)
        }
        
        # Store
        arch_id = architecture.hash()
        
        self.collection.add(
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            documents=[architecture.to_json()],
            ids=[arch_id]
        )
        
        logger.debug(f"Added architecture {arch_id} to knowledge base")
    
    def search_similar(
        self,
        query_architecture: ModelGraph,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[ModelGraph, float, Dict]]:
        """
        Search for similar architectures.
        
        Args:
            query_architecture: Query architecture
            top_k: Number of results
            filters: Metadata filters (e.g., {'dataset': 'CIFAR-10'})
        
        Returns:
            List of (architecture, distance, metadata) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_model.embed(query_architecture)
        
        # Build where clause for filtering
        where = filters if filters else None
        
        # Query
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where
        )
        
        # Parse results
        similar_archs = []
        for i in range(len(results['ids'][0])):
            arch_json = results['documents'][0][i]
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]
            
            arch = ModelGraph.from_json(arch_json)
            similar_archs.append((arch, distance, metadata))
        
        return similar_archs
    
    def get_best_for_task(
        self,
        task: TaskMetadata,
        top_k: int = 10
    ) -> List[ModelGraph]:
        """Get best architectures for a specific task."""
        results = self.collection.get(
            where={'dataset': task.dataset_name},
            limit=top_k
        )
        
        # Sort by accuracy
        archs_with_acc = [
            (ModelGraph.from_json(doc), meta['accuracy'])
            for doc, meta in zip(results['documents'], results['metadatas'])
        ]
        
        archs_with_acc.sort(key=lambda x: x[1], reverse=True)
        
        return [arch for arch, _ in archs_with_acc]


class ArchitectureEmbedder:
    """
    Embed architectures into vector space.
    
    Methods:
    1. Graph2Vec - Unsupervised graph embedding
    2. GNN-based - Learned embedding
    """
    
    def __init__(self, method: str = 'gnn', embedding_dim: int = 128):
        self.method = method
        self.embedding_dim = embedding_dim
        
        if method == 'gnn':
            self.model = self._load_gnn_encoder()
    
    def embed(self, graph: ModelGraph) -> np.ndarray:
        """
        Embed architecture as vector.
        
        Returns:
            Embedding vector of shape [embedding_dim]
        """
        if self.method == 'gnn':
            return self._embed_gnn(graph)
        elif self.method == 'graph2vec':
            return self._embed_graph2vec(graph)
        else:
            return self._embed_simple(graph)
    
    def _embed_gnn(self, graph: ModelGraph) -> np.ndarray:
        """Embed using GNN encoder."""
        import torch
        from torch_geometric.data import Data
        
        # Convert to PyG
        data = self._graph_to_pyg(graph)
        
        with torch.no_grad():
            embedding = self.model.encode(data.x, data.edge_index)
        
        return embedding.cpu().numpy()
    
    def _embed_simple(self, graph: ModelGraph) -> np.ndarray:
        """
        Simple embedding based on graph statistics.
        
        Features:
        - Operation type counts
        - Depth
        - Width
        - Number of parameters (estimated)
        """
        features = []
        
        # Operation counts
        ops = [node.operation for node in graph.nodes.values()]
        op_types = ['conv2d', 'maxpool', 'avgpool', 'dense', 'relu', 'batchnorm']
        for op_type in op_types:
            features.append(ops.count(op_type))
        
        # Graph structure
        features.append(len(graph.nodes))
        features.append(len(graph.edges))
        features.append(graph.get_depth())
        features.append(graph.get_max_width())
        
        # Pad to embedding_dim
        features = np.array(features, dtype=np.float32)
        if len(features) < self.embedding_dim:
            features = np.pad(features, (0, self.embedding_dim - len(features)))
        else:
            features = features[:self.embedding_dim]
        
        return features
```

---

### 2. `meta_learning/meta_features.py` (~1,000 LOC)

```python
class MetaFeatureExtractor:
    """
    Extract meta-features from tasks and architectures.
    
    Task meta-features:
    - Dataset statistics (size, dimensionality)
    - Class distribution
    - Sample complexity
    
    Architecture meta-features:
    - Structural properties
    - Operation statistics
    - Connectivity patterns
    """
    
    @staticmethod
    def extract_task_features(task: TaskMetadata, dataset: Dataset) -> Dict[str, float]:
        """Extract task meta-features."""
        features = {}
        
        # Dataset size
        features['num_samples'] = task.num_samples
        features['num_classes'] = task.num_classes
        features['input_dim'] = np.prod(task.input_size)
        
        # Class balance
        if hasattr(dataset, 'targets'):
            targets = np.array(dataset.targets)
            class_counts = np.bincount(targets)
            features['class_imbalance'] = class_counts.std() / class_counts.mean()
        
        # Input statistics
        sample_batch = dataset[0:100]['images']
        features['pixel_mean'] = sample_batch.mean()
        features['pixel_std'] = sample_batch.std()
        
        return features
    
    @staticmethod
    def extract_architecture_features(graph: ModelGraph) -> Dict[str, float]:
        """Extract architecture meta-features."""
        features = {}
        
        # Size
        features['num_nodes'] = len(graph.nodes)
        features['num_edges'] = len(graph.edges)
        
        # Depth and width
        features['depth'] = graph.get_depth()
        features['max_width'] = graph.get_max_width()
        
        # Operation diversity
        ops = [n.operation for n in graph.nodes.values()]
        features['num_unique_ops'] = len(set(ops))
        
        # Connectivity
        degrees = [len(n.predecessors) + len(n.successors) for n in graph.nodes.values()]
        features['avg_degree'] = np.mean(degrees)
        features['max_degree'] = np.max(degrees)
        
        # Estimated parameters
        features['estimated_params'] = graph.estimate_parameters()
        
        return features
```

---

### 3. `meta_learning/indexing.py` (~500 LOC)

```python
class ExperimentIndexer:
    """
    Index experiments for fast retrieval.
    
    Indexes:
    - By dataset
    - By performance range
    - By architecture size
    - By timestamp
    """
    
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.indexes = {}
    
    def build_indexes(self):
        """Build all indexes."""
        # Dataset index
        self.indexes['dataset'] = self._build_dataset_index()
        
        # Performance index
        self.indexes['accuracy'] = self._build_performance_index()
    
    def _build_dataset_index(self) -> Dict[str, List[str]]:
        """Index architectures by dataset."""
        index = {}
        
        # Query all architectures
        all_archs = self.kb.collection.get()
        
        for arch_id, metadata in zip(all_archs['ids'], all_archs['metadatas']):
            dataset = metadata.get('dataset', 'unknown')
            
            if dataset not in index:
                index[dataset] = []
            
            index[dataset].append(arch_id)
        
        return index
```

---

## 🧪 Tests

```python
def test_knowledge_base():
    """Test knowledge base operations."""
    kb = KnowledgeBase(persist_directory="/tmp/test_kb")
    
    # Add architectures
    for i in range(10):
        graph = random_graph()
        metrics = {'accuracy': random.uniform(0.7, 0.95)}
        task = TaskMetadata(...)
        
        kb.add_architecture(graph, metrics, task, f'exp_{i}')
    
    # Search
    query = random_graph()
    similar = kb.search_similar(query, top_k=5)
    
    assert len(similar) == 5
```

---

## ✅ Deliverables

- [ ] Vector database with ChromaDB
- [ ] Architecture embedding methods
- [ ] Meta-feature extraction
- [ ] Fast similarity search
- [ ] Indexing for efficient queries
- [ ] Handle 10,000+ experiments

---

**Next:** `04_strategy_evolution.md`
