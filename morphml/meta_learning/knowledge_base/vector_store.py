"""Vector storage for fast similarity search.

Lightweight implementation without external dependencies.
Can be upgraded to FAISS when available.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)

# Try to import FAISS (optional)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class VectorStore:
    """
    Vector storage with similarity search.
    
    Uses FAISS if available, otherwise falls back to NumPy.
    
    Args:
        embedding_dim: Dimension of embeddings
        use_faiss: Whether to use FAISS (if available)
        persist_path: Path to save/load store
    
    Example:
        >>> store = VectorStore(embedding_dim=128)
        >>> store.add(embedding, metadata={'id': 'arch1'}, data='...')
        >>> results = store.search(query_embedding, top_k=10)
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        use_faiss: bool = True,
        persist_path: Optional[str] = None,
    ):
        """Initialize vector store."""
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.persist_path = persist_path
        
        # Storage
        self.embeddings: List[np.ndarray] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.data: List[Any] = []
        
        # FAISS index
        self.index = None
        if self.use_faiss:
            self.index = faiss.IndexFlatL2(embedding_dim)
            logger.info(f"Initialized VectorStore with FAISS (dim={embedding_dim})")
        else:
            logger.info(f"Initialized VectorStore with NumPy (dim={embedding_dim})")
        
        # Load if path exists
        if persist_path and Path(persist_path).exists():
            self.load(persist_path)
    
    def add(
        self,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        data: Any = None,
    ) -> int:
        """
        Add item to store.
        
        Args:
            embedding: Embedding vector
            metadata: Metadata dict
            data: Associated data (e.g., ModelGraph)
        
        Returns:
            ID of added item
        """
        # Validate embedding
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} != {self.embedding_dim}"
            )
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Add to storage
        idx = len(self.embeddings)
        self.embeddings.append(embedding)
        self.metadatas.append(metadata)
        self.data.append(data)
        
        # Add to FAISS index
        if self.use_faiss and self.index is not None:
            self.index.add(embedding.reshape(1, -1).astype(np.float32))
        
        logger.debug(f"Added item {idx} to vector store")
        
        return idx
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_fn: Optional[callable] = None,
    ) -> List[Tuple[int, float, Dict[str, Any], Any]]:
        """
        Search for similar items.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter_fn: Optional filter function on metadata
        
        Returns:
            List of (id, distance, metadata, data) tuples
        """
        if len(self.embeddings) == 0:
            return []
        
        # Normalize query
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        if self.use_faiss and self.index is not None:
            # FAISS search
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                min(top_k * 2, len(self.embeddings))  # Get extra for filtering
            )
            
            candidates = [
                (int(indices[0, i]), float(distances[0, i]))
                for i in range(len(indices[0]))
                if indices[0, i] >= 0
            ]
        else:
            # NumPy search
            embeddings_array = np.array(self.embeddings)
            
            # Compute distances (L2)
            distances = np.linalg.norm(
                embeddings_array - query_embedding, axis=1
            )
            
            # Get top candidates
            top_indices = np.argsort(distances)[:top_k * 2]
            candidates = [(int(idx), float(distances[idx])) for idx in top_indices]
        
        # Apply filter
        results = []
        for idx, distance in candidates:
            metadata = self.metadatas[idx]
            
            # Apply filter
            if filter_fn is not None and not filter_fn(metadata):
                continue
            
            results.append((idx, distance, metadata, self.data[idx]))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any], Any]:
        """
        Get item by ID.
        
        Args:
            idx: Item ID
        
        Returns:
            (embedding, metadata, data) tuple
        """
        if idx < 0 or idx >= len(self.embeddings):
            raise IndexError(f"Invalid index: {idx}")
        
        return self.embeddings[idx], self.metadatas[idx], self.data[idx]
    
    def update_metadata(self, idx: int, metadata: Dict[str, Any]) -> None:
        """Update metadata for an item."""
        if idx < 0 or idx >= len(self.embeddings):
            raise IndexError(f"Invalid index: {idx}")
        
        self.metadatas[idx] = metadata
    
    def size(self) -> int:
        """Get number of items in store."""
        return len(self.embeddings)
    
    def save(self, path: str) -> None:
        """
        Save store to disk.
        
        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.save(path / "embeddings.npy", np.array(self.embeddings))
        
        # Save metadatas
        with open(path / "metadatas.json", "w") as f:
            json.dump(self.metadatas, f)
        
        # Save data (pickle)
        with open(path / "data.pkl", "wb") as f:
            pickle.dump(self.data, f)
        
        # Save FAISS index if available
        if self.use_faiss and self.index is not None:
            faiss.write_index(self.index, str(path / "faiss.index"))
        
        logger.info(f"Saved vector store to {path}")
    
    def load(self, path: str) -> None:
        """
        Load store from disk.
        
        Args:
            path: Directory path to load from
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Store not found: {path}")
        
        # Load embeddings
        embeddings_array = np.load(path / "embeddings.npy")
        self.embeddings = [embeddings_array[i] for i in range(len(embeddings_array))]
        
        # Load metadatas
        with open(path / "metadatas.json", "r") as f:
            self.metadatas = json.load(f)
        
        # Load data
        with open(path / "data.pkl", "rb") as f:
            self.data = pickle.load(f)
        
        # Load FAISS index if available
        if self.use_faiss and (path / "faiss.index").exists():
            self.index = faiss.read_index(str(path / "faiss.index"))
        
        logger.info(f"Loaded vector store from {path} ({len(self.embeddings)} items)")
    
    def clear(self) -> None:
        """Clear all items from store."""
        self.embeddings = []
        self.metadatas = []
        self.data = []
        
        if self.use_faiss:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        logger.info("Cleared vector store")
