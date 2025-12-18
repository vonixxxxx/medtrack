"""
Embedding generation utilities
"""

import torch
import numpy as np
from typing import Union
import torch.nn.functional as F

from ..models.embedding_model import EmbeddingModel, l2_norm


def generate_embedding(
    model: EmbeddingModel,
    image_tensor: torch.Tensor,
    normalize: bool = True,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Generate embedding vector for an image.
    
    Args:
        model: EmbeddingModel instance
        image_tensor: Preprocessed image tensor
        normalize: Whether to L2-normalize the embedding
        device: Device to run inference on
    
    Returns:
        Embedding vector as numpy array
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        embedding = model.get_embedding(image_tensor)
        
        if normalize:
            embedding = l2_norm(embedding)
        
        # Convert to numpy
        embedding_np = embedding.cpu().numpy()
        
        # Remove batch dimension if single image
        if embedding_np.ndim > 1 and embedding_np.shape[0] == 1:
            embedding_np = embedding_np[0]
    
    return embedding_np


def batch_generate_embeddings(
    model: EmbeddingModel,
    image_tensors: torch.Tensor,
    normalize: bool = True,
    batch_size: int = 32,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Generate embeddings for a batch of images.
    
    Args:
        model: EmbeddingModel instance
        image_tensors: Batch of preprocessed image tensors
        normalize: Whether to L2-normalize embeddings
        batch_size: Batch size for processing
        device: Device to run inference on
    
    Returns:
        Array of embedding vectors
    """
    model.eval()
    embeddings_list = []
    
    with torch.no_grad():
        for i in range(0, len(image_tensors), batch_size):
            batch = image_tensors[i:i+batch_size].to(device)
            batch_embeddings = model.get_embedding(batch)
            
            if normalize:
                batch_embeddings = l2_norm(batch_embeddings)
            
            embeddings_list.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(embeddings_list)







