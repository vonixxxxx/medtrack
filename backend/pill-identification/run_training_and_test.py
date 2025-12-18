#!/usr/bin/env python3
"""
Script to run training and test the model
"""

import os
import sys
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset.download import download_dataset
from dataset.prepare import DatasetPreparer
from utils.model_detector import find_existing_model


def main():
    print("="*60)
    print("Pill Identification - Training and Testing")
    print("="*60)
    
    # Check for dataset
    output_dir = Path("./output")
    prepared_dir = output_dir / "prepared"
    models_dir = output_dir / "models"
    data_dir = output_dir / "data"
    
    # Step 1: Check if dataset exists
    dataset_path = None
    possible_paths = [
        "./data/epillid/dataset",
        "./data/epillid",
        "../medication_dataset",
        "/tmp/epillid-repo"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Check if it has images or CSV files
            has_images = any(
                f.endswith(('.jpg', '.jpeg', '.png', '.csv'))
                for root, dirs, files in os.walk(path)
                for f in files
            )
            if has_images:
                dataset_path = path
                print(f"✓ Found dataset at: {dataset_path}")
                break
    
    if not dataset_path:
        print("\n⚠️  No dataset found. Attempting to download...")
        print("This will download the ePillID dataset from GitHub releases.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please provide dataset path.")
            return
        
        # Try to download
        downloaded = download_dataset("./data/epillid")
        if downloaded:
            dataset_path = str(downloaded)
            print(f"✓ Dataset downloaded to: {dataset_path}")
        else:
            print("✗ Failed to download dataset. Please download manually.")
            print("  Download from: https://github.com/usuyama/ePillID-benchmark/releases")
            return
    
    # Step 2: Prepare dataset
    print("\n[1/3] Preparing dataset...")
    if not prepared_dir.exists() or not (prepared_dir / "train.csv").exists():
        preparer = DatasetPreparer(dataset_path)
        result = preparer.prepare_dataset(
            output_dir=str(prepared_dir),
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            reserve_test_classes=0.2
        )
        
        # Create label encoder
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
        print(f"  Training:   {len(result['train'])} images")
        print(f"  Validation: {len(result['val'])} images")
        print(f"  Test:       {len(result['test'])} images (RESERVED)")
    else:
        print(f"✓ Using existing prepared dataset at {prepared_dir}")
    
    # Step 3: Train model
    print("\n[2/3] Training model...")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing model
    existing_model = find_existing_model([str(models_dir)])
    if existing_model:
        print(f"✓ Found existing model: {existing_model}")
        response = input("Use existing model or retrain? (use/retrain): ")
        if response.lower() == 'use':
            model_path = existing_model
            print("Using existing model.")
        else:
            model_path = None
    else:
        model_path = None
    
    if model_path is None:
        print("Starting training...")
        cmd = [
            sys.executable, 'training/train.py',
            '--data_root', dataset_path,
            '--prepared_dir', str(prepared_dir),
            '--output_dir', str(models_dir),
            '--network', 'resnet18',
            '--embedding_dim', '2048',
            '--num_epochs', '10',  # Start with 10 epochs for testing
            '--batch_size', '16',  # Smaller batch for testing
            '--loss_type', 'triplet'
        ]
        
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        if result.returncode != 0:
            print("✗ Training failed!")
            return
        
        # Find the trained model
        model_path = find_existing_model([str(models_dir)])
        if not model_path:
            print("✗ Could not find trained model!")
            return
    
    print(f"✓ Model ready: {model_path}")
    
    # Step 4: Build index
    print("\n[3/3] Building FAISS index...")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, 'build_index.py',
        '--prepared_dir', str(prepared_dir),
        '--model_path', model_path,
        '--output_dir', str(data_dir),
        '--network', 'resnet18',
        '--embedding_dim', '2048',
        '--use_reference_only'
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print("✗ Index building failed!")
        return
    
    print("✓ Index built")
    
    # Step 5: Test model
    print("\n[4/4] Testing model on reserved test set...")
    test_csv = prepared_dir / "test.csv"
    
    if not test_csv.exists():
        print("⚠️  Test set not found. Skipping evaluation.")
        return
    
    cmd = [
        sys.executable, 'evaluate_test_set.py',
        '--prepared_dir', str(prepared_dir),
        '--model_path', model_path,
        '--index_path', str(data_dir / 'pill_index.index'),
        '--metadata_path', str(data_dir / 'pill_metadata.json'),
        '--top_k', '5'
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    print("\n" + "="*60)
    print("Training and Testing Complete!")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Index: {data_dir / 'pill_index.index'}")
    print("\nTo start the inference service:")
    print("  uvicorn api.app:app --host 127.0.0.1 --port 8005")


if __name__ == '__main__':
    main()





