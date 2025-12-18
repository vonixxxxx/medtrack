#!/usr/bin/env python3
"""
Direct dataset preparation and training from images
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

sys.path.insert(0, str(Path(__file__).parent))

def prepare_from_images():
    """Prepare dataset directly from images"""
    img_dir = Path('data/epillid/dataset/classification_data/segmented_nih_pills_224')
    
    if not img_dir.exists():
        print(f"Error: Image directory not found: {img_dir}")
        return None
    
    print(f"Scanning {img_dir} for images...")
    images = list(img_dir.glob('*.jpg'))
    print(f"Found {len(images)} images")
    
    # Extract labels from filenames
    data = []
    for img in images:
        name = img.stem
        parts = name.split('_')
        label = parts[0] if parts else name
        is_ref = False  # All images from this directory are consumer images
        
        data.append({
            'image_path': str(img.absolute()),
            'label': label,
            'is_ref': is_ref
        })
    
    df = pd.DataFrame(data)
    print(f"Created dataset with {len(df)} images")
    print(f"  Unique labels: {df['label'].nunique()}")
    
    # Split dataset
    all_labels = df['label'].unique()
    n_test_classes = max(1, int(len(all_labels) * 0.2))
    test_classes = set(pd.Series(all_labels).sample(n=n_test_classes, random_state=42))
    
    print(f"\nReserving {len(test_classes)} classes (20%) entirely for testing")
    
    # Split by test classes
    test_mask = df['label'].isin(test_classes)
    test_df = df[test_mask].copy()
    train_val_df = df[~test_mask].copy()
    
    # Split remaining into train/val
    train_val_df = train_val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_total = len(train_val_df)
    n_val = int(n_total * 0.15)
    
    val_df = train_val_df[:n_val].copy()
    train_df = train_val_df[n_val:].copy()
    
    # Shuffle
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    output_dir = Path('output/prepared')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    
    # Create label encoder (from train/val only)
    all_labels = pd.concat([train_df['label'], val_df['label']])
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    
    with open(output_dir / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Metadata
    metadata = {
        'num_train': len(train_df),
        'num_val': len(val_df),
        'num_test': len(test_df),
        'num_classes': len(label_encoder.classes_),
        'num_test_classes': len(test_classes),
        'test_classes': sorted(list(test_classes))
    }
    
    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Dataset prepared and saved to {output_dir}")
    print(f"{'='*60}")
    print(f"  Training set:   {len(train_df):5d} images ({len(train_df['label'].unique())} classes)")
    print(f"  Validation set: {len(val_df):5d} images ({len(val_df['label'].unique())} classes)")
    print(f"  Test set:       {len(test_df):5d} images ({len(test_df['label'].unique())} classes)")
    print(f"\n  Test set is reserved and will NOT be used for training")
    print(f"{'='*60}")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'metadata': metadata
    }

if __name__ == '__main__':
    result = prepare_from_images()
    if result:
        print("\n✓ Dataset preparation complete!")
        print("\nNext steps:")
        print("  1. Train model: python3 training/train.py --prepared_dir ./output/prepared --output_dir ./output/models")
        print("  2. Build index: python3 build_index.py --prepared_dir ./output/prepared --output_dir ./output/data")
        print("  3. Test model: python3 evaluate_test_set.py --prepared_dir ./output/prepared")





