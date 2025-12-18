#!/usr/bin/env python3
"""
Script to generate reference pill embeddings and build FAISS index.

This script:
1. Loads all reference pill images from a directory or CSV
2. Generates embeddings using the trained model
3. Builds a FAISS index
4. Saves index and metadata

Usage:
    python generate_reference_index.py \
        --data_dir /path/to/pill/images \
        --output_dir ./data \
        --model_path ./models/pill_model.pth \
        --metadata_csv /path/to/metadata.csv
"""

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from typing import Optional

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import based on how script is run
try:
    from inference import PillIdentifier
    from utils.preprocessing import preprocess_image
    from utils.embedding import batch_generate_embeddings
    from utils.vector_search import PillVectorSearch
except ImportError:
    # Try relative imports
    from .inference import PillIdentifier
    from .utils.preprocessing import preprocess_image
    from .utils.embedding import batch_generate_embeddings
    from .utils.vector_search import PillVectorSearch


def load_image_paths(data_dir: str, metadata_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Load image paths and metadata.
    
    Args:
        data_dir: Directory containing pill images
        metadata_csv: Optional CSV file with image paths and metadata
    
    Returns:
        DataFrame with image paths and metadata
    """
    if metadata_csv and os.path.exists(metadata_csv):
        # Load from CSV
        df = pd.read_csv(metadata_csv)
        
        # Ensure image_path column exists
        if 'image_path' not in df.columns:
            raise ValueError("CSV must contain 'image_path' column")
        
        # Make paths absolute if relative
        df['image_path'] = df['image_path'].apply(
            lambda x: os.path.join(data_dir, x) if not os.path.isabs(x) else x
        )
    else:
        # Scan directory for images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_paths = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        # Create DataFrame
        df = pd.DataFrame({
            'image_path': image_paths,
            'pill_id': [os.path.basename(p).split('.')[0] for p in image_paths]
        })
    
    # Filter to existing files
    df = df[df['image_path'].apply(os.path.exists)]
    
    print(f"Found {len(df)} images")
    return df


def generate_embeddings_batch(
    identifier: PillIdentifier,
    image_paths: list,
    batch_size: int = 32
) -> np.ndarray:
    """
    Generate embeddings for a batch of images.
    
    Args:
        identifier: PillIdentifier instance
        image_paths: List of image file paths
        batch_size: Batch size for processing
    
    Returns:
        Array of embeddings
    """
    embeddings_list = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Generating embeddings"):
        batch_paths = image_paths[i:i+batch_size]
        
        # Load and preprocess images
        image_tensors = []
        valid_indices = []
        
        for j, path in enumerate(batch_paths):
            try:
                tensor = preprocess_image(path)
                image_tensors.append(tensor)
                valid_indices.append(i + j)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
                continue
        
        if not image_tensors:
            continue
        
        # Batch process
        batch_tensor = torch.cat(image_tensors, dim=0)
        batch_embeddings = batch_generate_embeddings(
            model=identifier.model,
            image_tensors=batch_tensor,
            normalize=True,
            batch_size=batch_size,
            device=identifier.device
        )
        
        embeddings_list.append(batch_embeddings)
    
    if embeddings_list:
        return np.vstack(embeddings_list)
    else:
        return np.array([])


def main():
    parser = argparse.ArgumentParser(description='Generate reference pill embeddings and FAISS index')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing pill images')
    parser.add_argument('--metadata_csv', type=str, default=None,
                        help='CSV file with image paths and metadata (optional)')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for index and metadata')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model weights (.pth file)')
    parser.add_argument('--network', type=str, default='resnet18',
                        help='CNN backbone architecture')
    parser.add_argument('--embedding_dim', type=int, default=2048,
                        help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for embedding generation')
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'l2'],
                        help='Distance metric for FAISS index')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize identifier
    print("Initializing pill identifier...")
    identifier = PillIdentifier(
        model_path=args.model_path,
        network=args.network,
        embedding_dim=args.embedding_dim,
        device=args.device
    )
    
    # Load image paths and metadata
    print("Loading image paths...")
    df = load_image_paths(args.data_dir, args.metadata_csv)
    
    if len(df) == 0:
        print("Error: No images found!")
        return
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings_batch(
        identifier=identifier,
        image_paths=df['image_path'].tolist(),
        batch_size=args.batch_size
    )
    
    if len(embeddings) == 0:
        print("Error: No embeddings generated!")
        return
    
    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # Prepare metadata
    print("Preparing metadata...")
    metadata_list = []
    for idx, row in df.iterrows():
        meta = {
            'index': len(metadata_list),
            'image_path': row['image_path'],
            'pill_id': row.get('pill_id', f'pill_{len(metadata_list)}')
        }
        
        # Add any additional columns as metadata
        for col in df.columns:
            if col not in ['image_path', 'pill_id']:
                meta[col] = row[col] if pd.notna(row[col]) else None
        
        metadata_list.append(meta)
    
    # Build FAISS index
    print("Building FAISS index...")
    
    vector_search = PillVectorSearch(
        embedding_dim=args.embedding_dim,
        metric=args.metric
    )
    
    vector_search.add_embeddings(embeddings, metadata_list)
    
    # Save index and metadata
    index_path = os.path.join(args.output_dir, 'pill_index.index')
    metadata_path = os.path.join(args.output_dir, 'pill_metadata.json')
    
    vector_search.save_index(index_path, metadata_path)
    
    print(f"\n✓ Successfully created index with {len(embeddings)} vectors")
    print(f"  Index saved to: {index_path}")
    print(f"  Metadata saved to: {metadata_path}")
    print(f"\nTo use this index, set environment variables:")
    print(f"  export PILL_INDEX_PATH={index_path}")
    print(f"  export PILL_METADATA_PATH={metadata_path}")


if __name__ == '__main__':
    main()

