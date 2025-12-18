"""
Main inference module for pill identification
"""

import os
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple

from .models.embedding_model import get_model
from .utils.preprocessing import preprocess_image, decode_base64_image
from .utils.embedding import generate_embedding
from .utils.vector_search import PillVectorSearch
from .utils.model_detector import find_existing_model, validate_model


class PillIdentifier:
    """
    Main class for pill identification inference.
    
    Handles model loading, embedding generation, and similarity search.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        network: str = 'resnet18',
        embedding_dim: int = 2048,
        device: Optional[str] = None,
        metric: str = 'cosine',
        auto_detect: bool = True
    ):
        """
        Initialize pill identifier.
        
        Args:
            model_path: Path to trained model weights (.pth file)
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
            network: CNN backbone architecture
            embedding_dim: Embedding dimension
            device: Device to use ('cpu' or 'cuda')
            metric: Distance metric for search ('cosine' or 'l2')
            auto_detect: Automatically detect model and index if not provided
        """
        # Determine device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Auto-detect model if not provided
        if model_path is None and auto_detect:
            print("Auto-detecting model...")
            detected_model = find_existing_model()
            if detected_model:
                model_path = detected_model
                print(f"Found model: {model_path}")
                
                # Try to extract network from checkpoint
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    if 'network' in checkpoint:
                        network = checkpoint['network']
                        print(f"Detected network: {network}")
                    if 'embedding_dim' in checkpoint:
                        embedding_dim = checkpoint['embedding_dim']
                        print(f"Detected embedding_dim: {embedding_dim}")
                except:
                    pass
        
        # Load model
        self.model = get_model(
            network=network,
            embedding_dim=embedding_dim,
            model_path=model_path,
            device=device
        )
        self.embedding_dim = embedding_dim
        
        # Auto-detect index if not provided
        if (index_path is None or not os.path.exists(index_path)) and auto_detect:
            print("Auto-detecting index...")
            base_dir = Path(__file__).parent
            possible_index_paths = [
                base_dir / 'data' / 'pill_index.index',
                base_dir.parent / 'data' / 'pill_index.index',
                Path('./data/pill_index.index'),
            ]
            
            for possible_path in possible_index_paths:
                if possible_path.exists():
                    index_path = str(possible_path)
                    if metadata_path is None:
                        metadata_path = str(possible_path).replace('.index', '_metadata.json')
                    print(f"Found index: {index_path}")
                    break
        
        # Load vector search index
        self.vector_search = None
        if index_path and os.path.exists(index_path):
            try:
                self.vector_search = PillVectorSearch(
                    embedding_dim=embedding_dim,
                    metric=metric,
                    index_path=index_path,
                    metadata_path=metadata_path
                )
                print(f"Loaded vector search index with {self.vector_search.index.ntotal} vectors")
            except Exception as e:
                print(f"Warning: Could not load vector search index: {e}")
        else:
            print("Warning: No vector search index loaded. Similarity search will not work.")
    
    def embed(self, image_input) -> np.ndarray:
        """
        Generate embedding for an image.
        
        Args:
            image_input: Image (PIL Image, numpy array, file path, bytes, or base64 string)
        
        Returns:
            Embedding vector as numpy array
        """
        # Handle base64 string
        if isinstance(image_input, str) and (image_input.startswith('data:') or len(image_input) > 100):
            # Likely base64
            try:
                image_input = decode_base64_image(image_input)
            except:
                pass  # Try as file path
        
        # Preprocess image
        image_tensor = preprocess_image(image_input)
        
        # Generate embedding
        embedding = generate_embedding(
            model=self.model,
            image_tensor=image_tensor,
            normalize=True,
            device=self.device
        )
        
        return embedding
    
    def identify(
        self,
        image_input,
        k: int = 5,
        min_confidence: float = 0.0
    ) -> Dict:
        """
        Identify a pill from an image.
        
        Args:
            image_input: Image (PIL Image, numpy array, file path, bytes, or base64 string)
            k: Number of top matches to return
            min_confidence: Minimum confidence threshold
        
        Returns:
            Dictionary with identification results
        """
        if self.vector_search is None:
            raise ValueError("Vector search index not loaded. Cannot perform identification.")
        
        # Generate embedding
        embedding = self.embed(image_input)
        
        # Search for similar pills
        indices, similarities, metadata_list = self.vector_search.search(
            query_embedding=embedding,
            k=k,
            return_distances=True
        )
        
        # Filter by confidence threshold
        candidates = []
        for idx, similarity, metadata in zip(indices, similarities, metadata_list):
            if similarity >= min_confidence:
                candidate = {
                    'rank': len(candidates) + 1,
                    'index': int(idx),
                    'confidence': float(similarity),
                    'metadata': metadata
                }
                candidates.append(candidate)
        
        # Prepare result
        result = {
            'top_match': candidates[0] if candidates else None,
            'candidates': candidates,
            'num_results': len(candidates),
            'embedding_dim': self.embedding_dim
        }
        
        return result
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'device': self.device,
            'embedding_dim': self.embedding_dim,
            'has_index': self.vector_search is not None,
            'index_stats': self.vector_search.get_stats() if self.vector_search else None
        }

