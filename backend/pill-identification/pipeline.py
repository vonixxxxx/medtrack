#!/usr/bin/env python3
"""
Complete pipeline for pill identification

This script orchestrates the entire pipeline:
1. Dataset download (if --auto)
2. Dataset preparation
3. Model training (if needed)
4. Index building
5. Service startup
6. Test inference (if --auto)

Usage:
    python pipeline.py --auto                    # Full automation
    python pipeline.py --data_root /path/to/data  # Manual mode
"""

import argparse
import os
import sys
import time
import requests
from pathlib import Path
import subprocess
import signal

sys.path.insert(0, str(Path(__file__).parent))

from dataset.prepare import DatasetPreparer
from dataset.download import download_dataset
from utils.model_detector import find_existing_model


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"Error: {description} failed with exit code {result.returncode}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Complete pill identification pipeline')
    
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of ePillID dataset')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for all generated files')
    parser.add_argument('--skip_preparation', action='store_true',
                        help='Skip dataset preparation')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip model training (use existing model)')
    parser.add_argument('--skip_index', action='store_true',
                        help='Skip index building')
    parser.add_argument('--start_service', action='store_true',
                        help='Start FastAPI service after pipeline')
    
    # Training arguments
    parser.add_argument('--network', type=str, default='resnet18',
                        help='CNN backbone')
    parser.add_argument('--embedding_dim', type=int, default=2048,
                        help='Embedding dimension')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prepared_dir = output_dir / 'prepared'
    models_dir = output_dir / 'models'
    data_dir = output_dir / 'data'
    
    print("="*60)
    print("Pill Identification Pipeline")
    print("="*60)
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Prepare dataset
    if not args.skip_preparation:
        print("\n[1/4] Preparing dataset...")
        preparer = DatasetPreparer(args.data_root)
        result = preparer.prepare_dataset(
            output_dir=str(prepared_dir),
            train_split=0.7,  # 70% for training
            val_split=0.15,   # 15% for validation
            test_split=0.15,   # 15% for testing
            reserve_test_classes=0.2  # Reserve 20% of classes entirely for testing
        )
        
        # Create label encoder (ONLY from train/val, NOT test)
        import pandas as pd
        import pickle
        from sklearn.preprocessing import LabelEncoder
        
        all_labels = pd.concat([
            result['train']['label'],
            result['val']['label']
        ])
        
        label_encoder = LabelEncoder()
        label_encoder.fit(all_labels)
        
        with open(prepared_dir / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print(f"✓ Dataset prepared:")
        print(f"  Training:   {len(result['train'])} images ({len(result['train']['label'].unique())} classes)")
        print(f"  Validation: {len(result['val'])} images ({len(result['val']['label'].unique())} classes)")
        print(f"  Test:       {len(result['test'])} images ({len(result['test']['label'].unique())} classes) - RESERVED")
        print(f"  ⚠️  Test set will NOT be used for training")
    else:
        print("\n[1/4] Skipping dataset preparation")
    
    # Step 2: Train model
    if not args.skip_training:
        existing_model = find_existing_model([str(models_dir)])
        if existing_model:
            print(f"\n[2/4] Found existing model: {existing_model}")
            print("  Skipping training (use --skip_training=false to force retraining)")
        else:
            print("\n[2/4] Training model...")
            cmd = [
                sys.executable, 'training/train.py',
                '--data_root', args.data_root,
                '--prepared_dir', str(prepared_dir),
                '--output_dir', str(models_dir),
                '--network', args.network,
                '--embedding_dim', str(args.embedding_dim),
                '--num_epochs', str(args.num_epochs),
                '--batch_size', str(args.batch_size)
            ]
            
            if not run_command(cmd, "Training model"):
                print("Training failed. Continuing with existing model if available...")
    else:
        print("\n[2/4] Skipping model training")
    
    # Step 3: Build index
    if not args.skip_index:
        print("\n[3/4] Building FAISS index...")
        
        # Find model
        model_path = find_existing_model([str(models_dir)])
        if not model_path:
            # Try other locations
            model_path = find_existing_model()
        
        if not model_path:
            print("Error: No model found. Cannot build index.")
            return
        
        cmd = [
            sys.executable, 'build_index.py',
            '--prepared_dir', str(prepared_dir),
            '--model_path', model_path,
            '--output_dir', str(data_dir),
            '--network', args.network,
            '--embedding_dim', str(args.embedding_dim),
            '--use_reference_only'
        ]
        
        if not run_command(cmd, "Building index"):
            print("Index building failed.")
    else:
        print("\n[3/4] Skipping index building")
    
    # Step 4: Start service (optional)
    if args.start_service:
        print("\n[4/4] Starting FastAPI service...")
        print("Service will be available at http://127.0.0.1:8005")
        print("Press Ctrl+C to stop")
        
        cmd = [
            sys.executable, '-m', 'uvicorn',
            'api.app:app',
            '--host', '127.0.0.1',
            '--port', '8005',
            '--reload'
        ]
        
        subprocess.run(cmd, cwd=Path(__file__).parent)
    else:
        print("\n[4/4] Pipeline complete!")
        print("\nTo start the service, run:")
        print("  uvicorn api.app:app --host 127.0.0.1 --port 8005 --reload")
        print("  or")
        print("  ./start_service.sh")


if __name__ == '__main__':
    main()

