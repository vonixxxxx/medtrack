#!/usr/bin/env python3
"""
Automatically build FAISS index from prepared dataset

This script:
1. Loads the prepared dataset
2. Generates embeddings for all reference images
3. Builds FAISS index
4. Saves index and metadata
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

from inference import PillIdentifier
from utils.preprocessing import preprocess_image
from utils.embedding import batch_generate_embeddings
from utils.model_detector import find_existing_model


def main():
    parser = argparse.ArgumentParser(description='Build FAISS index from dataset')
    
    parser.add_argument('--prepared_dir', type=str, required=True,
                        help='Directory with prepared dataset')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model weights (auto-detected if not provided)')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for index')
    parser.add_argument('--network', type=str, default='resnet18',
                        help='CNN backbone architecture')
    parser.add_argument('--embedding_dim', type=int, default=2048,
                        help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for embedding generation')
    parser.add_argument('--metric', type=str, default='cosine',
                        choices=['cosine', 'l2'],
                        help='Distance metric')
    parser.add_argument('--use_reference_only', action='store_true',
                        help='Only use reference images for index')
    
    args = parser.parse_args()
    
    # Find model if not provided
    if args.model_path is None:
        print("Auto-detecting model...")
        model_path = find_existing_model()
        if model_path:
            print(f"Found model: {model_path}")
            args.model_path = model_path
        else:
            print("Error: No model found. Please provide --model_path")
            return
    
    # Load prepared dataset
    print(f"Loading dataset from {args.prepared_dir}")
    train_df = pd.read_csv(os.path.join(args.prepared_dir, 'train.csv'))
    
    if args.use_reference_only:
        # Use only reference images
        df = train_df[train_df['is_ref'] == True].copy()
        print(f"Using {len(df)} reference images")
    else:
        # Use all training images
        df = train_df.copy()
        print(f"Using {len(df)} training images")
    
    # Initialize identifier
    print("Initializing pill identifier...")
    identifier = PillIdentifier(
        model_path=args.model_path,
        network=args.network,
        embedding_dim=args.embedding_dim,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Generate embeddings
    print("Generating embeddings...")
    image_paths = df['image_path'].tolist()
    
    # Load and preprocess images in batches
    embeddings_list = []
    valid_indices = []
    
    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i+args.batch_size]
        image_tensors = []
        
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
            batch_size=args.batch_size,
            device=identifier.device
        )
        
        embeddings_list.append(batch_embeddings)
    
    if not embeddings_list:
        print("Error: No embeddings generated!")
        return
    
    embeddings = torch.cat([torch.from_numpy(e) for e in embeddings_list], dim=0).numpy()
    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # Prepare metadata
    print("Preparing metadata...")
    metadata_list = []
    for idx in valid_indices:
        row = df.iloc[idx]
        meta = {
            'index': len(metadata_list),
            'image_path': row['image_path'],
            'label': row['label'],
            'is_ref': bool(row.get('is_ref', True))
        }
        
        # Add any additional columns
        for col in df.columns:
            if col not in ['image_path', 'label', 'is_ref']:
                meta[col] = row[col] if pd.notna(row[col]) else None
        
        metadata_list.append(meta)
    
    # Build FAISS index
    print("Building FAISS index...")
    from utils.vector_search import PillVectorSearch
    
    vector_search = PillVectorSearch(
        embedding_dim=args.embedding_dim,
        metric=args.metric
    )
    
    vector_search.add_embeddings(embeddings, metadata_list)
    
    # Save index
    os.makedirs(args.output_dir, exist_ok=True)
    index_path = os.path.join(args.output_dir, 'pill_index.index')
    metadata_path = os.path.join(args.output_dir, 'pill_metadata.json')
    
    vector_search.save_index(index_path, metadata_path)
    
    print(f"\n✓ Successfully created index with {len(embeddings)} vectors")
    print(f"  Index: {index_path}")
    print(f"  Metadata: {metadata_path}")


if __name__ == '__main__':
    main()







