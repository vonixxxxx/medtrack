#!/usr/bin/env python3
"""
Quick training and testing script
Handles small datasets and provides clear output
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("="*60)
    print("Quick Training and Testing")
    print("="*60)
    
    # Check for prepared dataset
    prepared_dir = Path("./output/prepared")
    
    if not prepared_dir.exists() or not (prepared_dir / "train.csv").exists():
        print("\n❌ No prepared dataset found!")
        print("\nPlease run dataset preparation first:")
        print("  python -c \"from dataset.prepare import prepare_dataset; prepare_dataset('/path/to/data', './output/prepared')\"")
        print("\nOr use the full pipeline:")
        print("  python pipeline.py --data_root /path/to/data")
        return
    
    # Check dataset size
    import pandas as pd
    train_df = pd.read_csv(prepared_dir / "train.csv")
    val_df = pd.read_csv(prepared_dir / "val.csv")
    test_df = pd.read_csv(prepared_dir / "test.csv") if (prepared_dir / "test.csv").exists() else None
    
    print(f"\nDataset Statistics:")
    print(f"  Training images:   {len(train_df)}")
    print(f"  Validation images: {len(val_df)}")
    if test_df is not None:
        print(f"  Test images:       {len(test_df)}")
    
    if len(train_df) == 0:
        print("\n❌ No training images found!")
        print("Please prepare a dataset with training images.")
        return
    
    if len(train_df) < 10:
        print(f"\n⚠️  Warning: Very small dataset ({len(train_df)} training images)")
        print("Training may not produce good results.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Check for existing model
    from utils.model_detector import find_existing_model
    models_dir = Path("./output/models")
    existing_model = find_existing_model([str(models_dir)])
    
    if existing_model:
        print(f"\n✓ Found existing model: {existing_model}")
        use_existing = input("Use existing model? (y/n): ")
        if use_existing.lower() == 'y':
            model_path = existing_model
        else:
            model_path = None
    else:
        model_path = None
    
    # Train if needed
    if model_path is None:
        print("\n[1/2] Starting training...")
        print("This may take a while depending on dataset size and hardware.")
        
        import subprocess
        cmd = [
            sys.executable, 'training/train.py',
            '--prepared_dir', str(prepared_dir),
            '--output_dir', str(models_dir),
            '--network', 'resnet18',
            '--embedding_dim', '2048',
            '--num_epochs', '5',  # Reduced for quick testing
            '--batch_size', '8',  # Smaller batch
            '--loss_type', 'triplet'
        ]
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\n❌ Training failed!")
            return
        
        model_path = find_existing_model([str(models_dir)])
        if not model_path:
            print("\n❌ Could not find trained model!")
            return
    
    print(f"\n✓ Model ready: {model_path}")
    
    # Build index
    print("\n[2/2] Building index and testing...")
    data_dir = Path("./output/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    import subprocess
    cmd = [
        sys.executable, 'build_index.py',
        '--prepared_dir', str(prepared_dir),
        '--model_path', model_path,
        '--output_dir', str(data_dir),
        '--network', 'resnet18',
        '--embedding_dim', '2048',
        '--use_reference_only'
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\n❌ Index building failed!")
        return
    
    # Test if test set exists
    if test_df is not None and len(test_df) > 0:
        print("\n[3/3] Testing on reserved test set...")
        cmd = [
            sys.executable, 'evaluate_test_set.py',
            '--prepared_dir', str(prepared_dir),
            '--model_path', model_path,
            '--index_path', str(data_dir / 'pill_index.index'),
            '--metadata_path', str(data_dir / 'pill_metadata.json'),
            '--top_k', '5'
        ]
        
        subprocess.run(cmd)
    else:
        print("\n⚠️  No test set available for evaluation")
    
    print("\n" + "="*60)
    print("Complete!")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Index: {data_dir / 'pill_index.index'}")


if __name__ == '__main__':
    main()





