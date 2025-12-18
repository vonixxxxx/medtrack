#!/usr/bin/env python3
"""
Prepare dataset directly from images when CSV doesn't match
"""

import sys
from pathlib import Path
import pandas as pd
import os

sys.path.insert(0, str(Path(__file__).parent))

from dataset.prepare import DatasetPreparer

def main():
    # Build dataset from images directly
    img_dir = Path('data/epillid/dataset/classification_data/segmented_nih_pills_224')
    
    if not img_dir.exists():
        print(f"Error: Image directory not found: {img_dir}")
        return
    
    print(f"Scanning {img_dir} for images...")
    images = list(img_dir.glob('*.jpg'))
    print(f"Found {len(images)} images")
    
    # Extract labels from filenames
    data = []
    for img in images:
        name = img.stem
        # Format: label_0_0.jpg -> label is first part before first _
        parts = name.split('_')
        label = parts[0] if parts else name
        
        # Determine if reference (heuristic: reference images often have simpler names)
        # For now, mark all as consumer (False) since we don't have clear indication
        is_ref = False
        
        # Use absolute path
        data.append({
            'image_path': str(img.absolute()),
            'label': label,
            'is_ref': is_ref
        })
    
    # Create temporary CSV
    temp_csv = Path('output/prepared/temp_images.csv')
    temp_csv.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(data)
    df.to_csv(temp_csv, index=False)
    print(f"Created temporary CSV with {len(df)} images")
    print(f"  Unique labels: {df['label'].nunique()}")
    
    # Now use the preparer
    preparer = DatasetPreparer(str(temp_csv.parent))
    result = preparer.prepare_dataset(
        output_dir='./output/prepared',
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        reserve_test_classes=0.2
    )
    
    print(f"\n✓ Dataset prepared!")
    print(f"  Training:   {len(result['train'])} images")
    print(f"  Validation: {len(result['val'])} images")
    print(f"  Test:       {len(result['test'])} images")

if __name__ == '__main__':
    main()

