"""
Dataset preparation for ePillID benchmark
Automatically detects and organizes reference and consumer images
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict
from PIL import Image
import shutil


class DatasetPreparer:
    """
    Prepares ePillID dataset for training and inference.
    
    Automatically:
    - Detects dataset structure
    - Organizes reference and consumer images
    - Builds class→image mappings
    - Creates metadata files
    """
    
    def __init__(self, data_root: str):
        """
        Initialize dataset preparer.
        
        Args:
            data_root: Root directory of the ePillID dataset
        """
        self.data_root = Path(data_root)
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def detect_structure(self) -> Dict:
        """
        Automatically detect the dataset structure.
        
        Returns:
            Dictionary with detected structure information
        """
        structure = {
            'has_csv': False,
            'has_folders': False,
            'csv_files': [],
            'image_dirs': [],
            'structure_type': 'unknown'
        }
        
        # Look for CSV files
        csv_files = list(self.data_root.rglob('*.csv'))
        if csv_files:
            structure['has_csv'] = True
            structure['csv_files'] = [str(f) for f in csv_files]
        
        # Look for image directories
        image_dirs = []
        for root, dirs, files in os.walk(self.data_root):
            # Check if directory contains images
            has_images = any(
                any(f.lower().endswith(ext) for ext in self.image_extensions)
                for f in files
            )
            if has_images:
                image_dirs.append(root)
        
        if image_dirs:
            structure['has_folders'] = True
            structure['image_dirs'] = image_dirs
        
        # Determine structure type
        if structure['has_csv']:
            structure['structure_type'] = 'csv_based'
        elif structure['has_folders']:
            structure['structure_type'] = 'folder_based'
        
        return structure
    
    def load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load dataset information from CSV file.
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            DataFrame with image paths and metadata
        """
        df = pd.read_csv(csv_path)
        
        # Detect common column names
        path_cols = [c for c in df.columns if 'path' in c.lower() or 'image' in c.lower()]
        if not path_cols:
            raise ValueError(f"No image path column found in {csv_path}")
        
        path_col = path_cols[0]
        
        # Make paths absolute if relative
        # Check if paths exist as-is first
        sample_path = df[path_col].iloc[0] if len(df) > 0 else None
        if sample_path:
            csv_dir = Path(csv_path).parent
            
            # Check if images are in classification_data directory
            # The CSV may have wrong paths, so we'll try to find actual images
            classification_dir = csv_dir / 'classification_data' / 'segmented_nih_pills_224'
            
            if classification_dir.exists():
                # Images are in classification_data, map CSV labels to actual images
                # Create a mapping from label to actual image files
                from collections import defaultdict
                label_to_images = defaultdict(list)
                
                # Find all images and map by label prefix
                print(f"  Scanning {classification_dir} for images...")
                for img_file in classification_dir.glob('*.jpg'):
                    img_name = img_file.stem
                    # Images are named like: 51285-0092-87_BE305F72_0_0.jpg
                    # The label in CSV is: 51285-0092-87_BE305F72
                    # Extract prefix (everything before last few _)
                    # Try multiple label formats
                    parts = img_name.split('_')
                    if len(parts) >= 2:
                        # Try full label (all parts except last 2)
                        possible_label = '_'.join(parts[:-2])
                        label_to_images[possible_label].append(str(img_file))
                        # Also try shorter versions
                        if len(parts) >= 3:
                            label_to_images['_'.join(parts[:-1])].append(str(img_file))
                
                # Map CSV labels to actual image files
                # CSV labels are like: 51285-0092-87_BE305F72
                # Image names are like: 51285-0092-87_BE305F72_0_0.jpg or 51285-0092-87_0_0.jpg
                # Try to match by prefix
                def map_to_actual_image(row):
                    label = str(row.get('label', ''))
                    if not label:
                        return None
                    
                    # Try exact match first
                    if label in label_to_images:
                        return label_to_images[label][0]
                    
                    # Try prefix matching (label might be longer)
                    # Extract first part before any additional suffixes
                    label_parts = label.split('_')
                    for i in range(len(label_parts), 0, -1):
                        prefix = '_'.join(label_parts[:i])
                        if prefix in label_to_images:
                            return label_to_images[prefix][0]
                    
                    # Try matching by pilltype_id if available
                    if 'pilltype_id' in row:
                        pill_id = str(row['pilltype_id'])
                        if pill_id in label_to_images:
                            return label_to_images[pill_id][0]
                    
                    return None
                
                # Map paths
                mapped_paths = []
                valid_indices = []
                for idx, row in df.iterrows():
                    mapped = map_to_actual_image(row)
                    if mapped:
                        mapped_paths.append(mapped)
                        valid_indices.append(idx)
                
                # Update dataframe with mapped paths
                if len(mapped_paths) > 0:
                    df = df.iloc[valid_indices].copy()
                    df[path_col] = mapped_paths
                    print(f"  Mapped {len(mapped_paths)} images from {len(label_to_images)} label groups")
                else:
                    print(f"  ⚠️  Warning: Could not map any images. Using original paths.")
                    df = df.iloc[:0].copy()  # Empty dataframe
            else:
                # Standard path resolution
                possible_bases = [
                    csv_dir,
                    csv_dir.parent,
                    csv_dir / 'classification_data',
                    csv_dir / 'fcn_mix_weight',
                ]
                
                base_dir = csv_dir
                for base in possible_bases:
                    if (base / sample_path).exists():
                        base_dir = base
                        break
                
                df[path_col] = df[path_col].apply(
                    lambda x: str(base_dir / x) if not os.path.isabs(x) else x
                )
        
        # Detect reference/consumer column (must be done after path mapping)
        ref_cols = [c for c in df.columns if 'ref' in c.lower() or 'is_ref' in c.lower()]
        if ref_cols:
            df['is_ref'] = df[ref_cols[0]].astype(bool)
        else:
            # Try to infer from path - handle case where path_col might be None
            if path_col in df.columns and len(df) > 0:
                df['is_ref'] = df[path_col].astype(str).str.contains('ref', case=False, na=False)
            else:
                df['is_ref'] = False
        
        # Detect label/class column
        label_cols = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower() or 'pill' in c.lower()]
        if label_cols:
            label_col = label_cols[0]
        else:
            # Try to infer from filename
            df['label'] = df[path_col].apply(lambda x: Path(x).stem.split('_')[0])
            label_col = 'label'
        
        # Standardize column names
        result_df = pd.DataFrame({
            'image_path': df[path_col],
            'is_ref': df['is_ref'],
            'label': df[label_col] if label_col in df.columns else df.get('pilltype_id', 'unknown')
        })
        
        # Add any additional metadata columns
        for col in df.columns:
            if col not in [path_col, ref_cols[0] if ref_cols else None, label_col]:
                if col not in result_df.columns:
                    result_df[col] = df[col]
        
        # Filter to existing files
        def check_exists(path):
            try:
                return os.path.exists(str(path))
            except:
                return False
        
        result_df = result_df[result_df['image_path'].apply(check_exists)]
        
        return result_df
    
    def load_from_folders(self) -> pd.DataFrame:
        """
        Load dataset from folder structure.
        
        Assumes structure like:
        - reference/ or ref/ for reference images
        - consumer/ or cons/ for consumer images
        - Or organized by class
        
        Returns:
            DataFrame with image paths and metadata
        """
        images = []
        
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.image_extensions):
                    img_path = os.path.join(root, file)
                    
                    # Determine if reference or consumer
                    is_ref = 'ref' in root.lower() or 'reference' in root.lower()
                    
                    # Try to extract label from path or filename
                    path_parts = Path(root).parts
                    label = None
                    for part in path_parts:
                        if part not in ['ref', 'reference', 'consumer', 'cons', 'images', 'data']:
                            label = part
                            break
                    
                    if not label:
                        # Try from filename
                        label = Path(file).stem.split('_')[0]
                    
                    images.append({
                        'image_path': img_path,
                        'is_ref': is_ref,
                        'label': label,
                        'filename': file
                    })
        
        return pd.DataFrame(images)
    
    def prepare_dataset(
        self,
        output_dir: Optional[str] = None,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        use_reference_for_training: bool = True,
        reserve_test_classes: float = 0.2
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare dataset for training with proper train/val/test splits.
        
        Args:
            output_dir: Directory to save prepared data (default: data_root/prepared)
            train_split: Fraction for training set
            val_split: Fraction for validation set
            test_split: Fraction for test set (must sum to 1.0 with train_split + val_split)
            use_reference_for_training: Use reference images for training (default: True)
            reserve_test_classes: Fraction of classes to reserve entirely for testing
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        structure = self.detect_structure()
        
        if structure['structure_type'] == 'csv_based':
            # Use first CSV file found
            csv_path = structure['csv_files'][0]
            print(f"Loading dataset from CSV: {csv_path}")
            df = self.load_from_csv(csv_path)
        elif structure['structure_type'] == 'folder_based':
            print("Loading dataset from folder structure")
            df = self.load_from_folders()
        else:
            raise ValueError(f"Could not detect dataset structure in {self.data_root}")
        
        print(f"Loaded {len(df)} images")
        if len(df) > 0:
            if 'is_ref' in df.columns:
                print(f"  Reference images: {df['is_ref'].sum()}")
                print(f"  Consumer images: {(~df['is_ref']).sum()}")
            else:
                print(f"  Reference/consumer info: not available")
            if 'label' in df.columns:
                print(f"  Unique labels: {df['label'].nunique()}")
            else:
                print(f"  Labels: not available")
        else:
            print("  ⚠️  No images loaded - check dataset paths")
        
        # Reserve some classes entirely for testing (important for evaluation)
        all_labels = df['label'].unique()
        n_test_classes = max(1, int(len(all_labels) * reserve_test_classes))
        test_classes = set(pd.Series(all_labels).sample(n=n_test_classes, random_state=42))
        
        print(f"\nReserving {len(test_classes)} classes ({reserve_test_classes*100:.1f}%) entirely for testing")
        
        # Split by reference/consumer and test classes
        ref_df = df[df['is_ref']].copy()
        cons_df = df[~df['is_ref']].copy()
        
        # Separate test classes
        test_class_mask = df['label'].isin(test_classes)
        test_df = df[test_class_mask].copy()
        
        # Remaining data for train/val
        train_val_df = df[~test_class_mask].copy()
        train_val_ref = train_val_df[train_val_df['is_ref']].copy()
        train_val_cons = train_val_df[~train_val_df['is_ref']].copy()
        
        # Split remaining consumer images into train and val
        train_val_cons = train_val_cons.sample(frac=1, random_state=42).reset_index(drop=True)
        n_cons = len(train_val_cons)
        n_val = int(n_cons * (val_split / (train_split + val_split)))
        
        val_cons_df = train_val_cons[:n_val].copy()
        train_cons_df = train_val_cons[n_val:].copy()
        
        # Build training set
        if use_reference_for_training:
            # Use reference images + subset of consumer images for training
            train_df = pd.concat([train_val_ref, train_cons_df], ignore_index=True)
            val_df = val_cons_df.copy()
        else:
            # Use only consumer images
            train_df = train_cons_df.copy()
            val_df = pd.concat([train_val_ref, val_cons_df], ignore_index=True)
        
        # Shuffle training set
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Ensure test set has both reference and consumer if available
        test_ref = test_df[test_df['is_ref']].copy()
        test_cons = test_df[~test_df['is_ref']].copy()
        test_df = pd.concat([test_ref, test_cons], ignore_index=True)
        
        # Save splits
        if output_dir is None:
            output_dir = self.data_root / 'prepared'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(output_dir / 'train.csv', index=False)
        val_df.to_csv(output_dir / 'val.csv', index=False)
        test_df.to_csv(output_dir / 'test.csv', index=False)
        
        # Save metadata
        metadata = {
            'num_train': len(train_df),
            'num_val': len(val_df),
            'num_test': len(test_df),
            'num_classes': df['label'].nunique(),
            'num_test_classes': len(test_classes),
            'test_classes': sorted(list(test_classes)),
            'num_ref_images': len(ref_df),
            'num_cons_images': len(cons_df),
            'train_classes': len(train_df['label'].unique()),
            'val_classes': len(val_df['label'].unique()),
            'test_classes_count': len(test_df['label'].unique()),
            'class_distribution': df['label'].value_counts().to_dict()
        }
        
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
    
    def create_label_encoder(self, df: pd.DataFrame, output_path: str):
        """
        Create and save label encoder.
        
        Args:
            df: DataFrame with labels
            output_path: Path to save encoder
        """
        from sklearn.preprocessing import LabelEncoder
        import pickle
        
        encoder = LabelEncoder()
        encoder.fit(df['label'])
        
        with open(output_path, 'wb') as f:
            pickle.dump(encoder, f)
        
        print(f"Label encoder saved to {output_path}")
        print(f"  Number of classes: {len(encoder.classes_)}")
        
        return encoder


def prepare_dataset(
    data_root: str,
    output_dir: Optional[str] = None,
    train_split: float = 0.8,
    val_split: float = 0.1
) -> Dict:
    """
    Convenience function to prepare dataset.
    
    Args:
        data_root: Root directory of ePillID dataset
        output_dir: Output directory for prepared data
        train_split: Training split fraction
        val_split: Validation split fraction
    
    Returns:
        Dictionary with prepared datasets and metadata
    """
    preparer = DatasetPreparer(data_root)
    return preparer.prepare_dataset(output_dir, train_split, val_split)



