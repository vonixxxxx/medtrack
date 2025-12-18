#!/usr/bin/env python3
"""
Evaluate model on reserved test set

This script evaluates the trained model on the test set that was
reserved during dataset preparation and not used for training.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from inference import PillIdentifier
from utils.preprocessing import preprocess_image
from utils.embedding import generate_embedding
from utils.model_detector import find_existing_model


def evaluate_on_test_set(
    test_csv: str,
    model_path: str,
    index_path: str,
    metadata_path: str,
    top_k: int = 5
):
    """
    Evaluate model on test set.
    
    Args:
        test_csv: Path to test set CSV
        model_path: Path to trained model
        index_path: Path to FAISS index
        metadata_path: Path to metadata JSON
        top_k: Top-k accuracy to compute
    """
    print("="*60)
    print("Evaluating on Reserved Test Set")
    print("="*60)
    
    # Load test set
    test_df = pd.read_csv(test_csv)
    print(f"Test set: {len(test_df)} images")
    print(f"Test classes: {test_df['label'].nunique()}")
    
    # Initialize identifier
    print("\nLoading model and index...")
    identifier = PillIdentifier(
        model_path=model_path,
        index_path=index_path,
        metadata_path=metadata_path,
        auto_detect=False
    )
    
    # Get all unique labels in test set
    test_labels = test_df['label'].unique()
    
    # Evaluate
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    
    print("\nRunning inference on test set...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        image_path = row['image_path']
        true_label = row['label']
        
        if not os.path.exists(image_path):
            continue
        
        try:
            # Identify pill
            result = identifier.identify(image_path, k=top_k, min_confidence=0.0)
            
            if result['candidates']:
                # Get predicted labels
                predicted_labels = [
                    cand['metadata'].get('label', 'unknown')
                    for cand in result['candidates']
                ]
                
                # Top-1 accuracy
                if predicted_labels[0] == true_label:
                    correct_top1 += 1
                    per_class_correct[true_label] += 1
                
                # Top-k accuracy
                if true_label in predicted_labels:
                    correct_top5 += 1
                
                per_class_total[true_label] += 1
                total += 1
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Compute metrics
    top1_accuracy = correct_top1 / total if total > 0 else 0.0
    top5_accuracy = correct_top5 / total if total > 0 else 0.0
    
    print("\n" + "="*60)
    print("Test Set Evaluation Results")
    print("="*60)
    print(f"Total images evaluated: {total}")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f} ({correct_top1}/{total})")
    print(f"Top-{top_k} Accuracy: {top5_accuracy:.4f} ({correct_top5}/{total})")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy (classes with >= 5 images):")
    print("-"*60)
    class_accuracies = []
    for label in sorted(per_class_total.keys()):
        if per_class_total[label] >= 5:
            acc = per_class_correct[label] / per_class_total[label]
            class_accuracies.append(acc)
            print(f"  {label:30s} {acc:.4f} ({per_class_correct[label]}/{per_class_total[label]})")
    
    if class_accuracies:
        print(f"\nMean per-class accuracy: {np.mean(class_accuracies):.4f}")
    
    print("="*60)
    
    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'total_images': total,
        'per_class_accuracy': dict(per_class_correct)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on reserved test set')
    
    parser.add_argument('--prepared_dir', type=str, required=True,
                        help='Directory with prepared dataset')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model (auto-detected if not provided)')
    parser.add_argument('--index_path', type=str, default=None,
                        help='Path to FAISS index')
    parser.add_argument('--metadata_path', type=str, default=None,
                        help='Path to metadata JSON')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top-k accuracy to compute')
    
    args = parser.parse_args()
    
    # Find model if not provided
    if args.model_path is None:
        model_path = find_existing_model()
        if model_path:
            args.model_path = model_path
        else:
            print("Error: No model found. Please provide --model_path")
            return
    
    # Default paths
    if args.index_path is None:
        args.index_path = os.path.join(args.prepared_dir, '../data/pill_index.index')
    if args.metadata_path is None:
        args.metadata_path = os.path.join(args.prepared_dir, '../data/pill_metadata.json')
    
    # Check test set exists
    test_csv = os.path.join(args.prepared_dir, 'test.csv')
    if not os.path.exists(test_csv):
        print(f"Error: Test set not found at {test_csv}")
        print("Make sure dataset was prepared with test split")
        return
    
    # Evaluate
    evaluate_on_test_set(
        test_csv=test_csv,
        model_path=args.model_path,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        top_k=args.top_k
    )


if __name__ == '__main__':
    main()





