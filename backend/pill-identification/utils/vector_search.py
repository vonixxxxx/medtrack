"""
FAISS-based vector search for pill similarity matching
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
import json


class PillVectorSearch:
    """
    FAISS-based vector search for pill identification.
    
    Stores reference pill embeddings and metadata,
    provides similarity search functionality.
    """
    
    def __init__(
        self,
        embedding_dim: int = 2048,
        metric: str = 'cosine',
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ):
        """
        Initialize vector search index.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            metric: Distance metric ('cosine' or 'l2')
            index_path: Path to saved FAISS index
            metadata_path: Path to saved metadata JSON
        """
        self.embedding_dim = embedding_dim
        self.metric = metric
        
        # Initialize FAISS index
        if metric == 'cosine':
            # For cosine similarity, use inner product on normalized vectors
            self.index = faiss.IndexFlatIP(embedding_dim)
        elif metric == 'l2':
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        self.metadata: List[Dict] = []
        
        # Load existing index if provided
        if index_path and os.path.exists(index_path):
            self.load_index(index_path, metadata_path)
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ):
        """
        Add embeddings and metadata to the index.
        
        Args:
            embeddings: Array of embedding vectors (N x embedding_dim)
            metadata: List of metadata dictionaries for each embedding
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"Metadata length mismatch: {len(metadata)} metadata entries "
                f"for {embeddings.shape[0]} embeddings"
            )
        
        # Normalize embeddings for cosine similarity
        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.metadata.extend(metadata)
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        return_distances: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Search for similar pills.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            return_distances: Whether to return distance scores
        
        Returns:
            Tuple of (indices, distances, metadata_list)
        """
        if self.index.ntotal == 0:
            raise ValueError("Index is empty. Add embeddings first.")
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if query_embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {query_embedding.shape[1]}"
            )
        
        # Normalize for cosine similarity
        if self.metric == 'cosine':
            faiss.normalize_L2(query_embedding)
        
        # Search
        query_embedding = query_embedding.astype('float32')
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Get metadata for results
        result_metadata = [self.metadata[idx] for idx in indices[0]]
        
        # Convert distances to similarities (confidence scores)
        if self.metric == 'cosine':
            # Cosine similarity is already in [0, 1] for normalized vectors
            similarities = distances[0]
        else:
            # Convert L2 distance to similarity (inverse, normalized)
            max_dist = distances[0].max()
            if max_dist > 0:
                similarities = 1.0 - (distances[0] / max_dist)
            else:
                similarities = np.ones_like(distances[0])
        
        if return_distances:
            return indices[0], similarities, result_metadata
        else:
            return indices[0], result_metadata
    
    def save_index(self, index_path: str, metadata_path: Optional[str] = None):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata JSON
        """
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        if metadata_path is None:
            metadata_path = index_path.replace('.index', '_metadata.json')
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Saved index to {index_path}")
        print(f"Saved metadata to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: Optional[str] = None):
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        if metadata_path is None:
            metadata_path = index_path.replace('.index', '_metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded {len(self.metadata)} metadata entries")
        else:
            print(f"Warning: Metadata file not found at {metadata_path}")
            self.metadata = []
        
        print(f"Loaded index with {self.index.ntotal} vectors")
    
    def get_stats(self) -> Dict:
        """Get statistics about the index"""
        return {
            'num_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'metric': self.metric,
            'num_metadata_entries': len(self.metadata)
        }







