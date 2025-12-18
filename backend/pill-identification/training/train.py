#!/usr/bin/env python3
"""
Main training script for pill identification model

Automatically:
- Detects existing model or starts training
- Prepares dataset if needed
- Trains model with metric learning
- Saves checkpoints and final model
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.prepare import DatasetPreparer, prepare_dataset
from training.trainer import MetricLearningTrainer
from utils.model_detector import find_existing_model


def main():
    parser = argparse.ArgumentParser(description='Train pill identification model')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of ePillID dataset')
    parser.add_argument('--prepared_dir', type=str, default=None,
                        help='Directory with prepared dataset (if already prepared)')
    
    # Model arguments
    parser.add_argument('--network', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='CNN backbone architecture')
    parser.add_argument('--embedding_dim', type=int, default=2048,
                        help='Embedding dimension')
    parser.add_argument('--loss_type', type=str, default='triplet',
                        choices=['triplet', 'arcface'],
                        help='Loss function type')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for triplet loss')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Model detection
    parser.add_argument('--skip_if_exists', action='store_true',
                        help='Skip training if model already exists')
    
    args = parser.parse_args()
    
    # Check for existing model
    if args.skip_if_exists:
        existing_model = find_existing_model(args.output_dir)
        if existing_model:
            print(f"Found existing model: {existing_model}")
            print("Skipping training (use --skip_if_exists=false to force retraining)")
            return
    
    # Prepare dataset if needed
    if args.prepared_dir and os.path.exists(args.prepared_dir):
        print(f"Using prepared dataset from {args.prepared_dir}")
        train_df = pd.read_csv(os.path.join(args.prepared_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(args.prepared_dir, 'val.csv'))
        
        # Verify test set exists and is separate
        test_csv = os.path.join(args.prepared_dir, 'test.csv')
        if os.path.exists(test_csv):
            test_df = pd.read_csv(test_csv)
            print(f"  Test set: {len(test_df)} images (reserved, not used for training)")
        
        # Load label encoder
        import pickle
        encoder_path = os.path.join(args.prepared_dir, 'label_encoder.pkl')
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
        else:
            # Create from training data only
            all_labels = pd.concat([train_df['label'], val_df['label']])
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            label_encoder.fit(all_labels)
            with open(encoder_path, 'wb') as f:
                pickle.dump(label_encoder, f)
    else:
        print("Preparing dataset...")
        preparer = DatasetPreparer(args.data_root)
        result = preparer.prepare_dataset(
            output_dir=args.prepared_dir if args.prepared_dir else None,
            reserve_test_classes=0.2  # Reserve 20% of classes for testing
        )
        
        train_df = result['train']
        val_df = result['val']
        test_df = result['test']
        
        print(f"\n⚠️  Test set ({len(test_df)} images) is reserved and will NOT be used for training")
        
        # Create label encoder from training data only (not test)
        all_labels = pd.concat([train_df['label'], val_df['label']])
        label_encoder = preparer.create_label_encoder(
            pd.DataFrame({'label': all_labels}),
            os.path.join(preparer.data_root / 'prepared', 'label_encoder.pkl')
        )
    
    num_classes = len(label_encoder.classes_)
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    print(f"  Training images:   {len(train_df):5d}")
    print(f"  Validation images: {len(val_df):5d}")
    print(f"  Training classes:  {num_classes}")
    
    # Check if test set exists
    test_csv = os.path.join(args.prepared_dir if args.prepared_dir else preparer.data_root / 'prepared', 'test.csv')
    if os.path.exists(test_csv):
        test_df_check = pd.read_csv(test_csv)
        print(f"  Test images:       {len(test_df_check):5d} (reserved, not used)")
    print(f"{'='*60}")
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    trainer = MetricLearningTrainer(
        network=args.network,
        embedding_dim=args.embedding_dim,
        num_classes=num_classes if args.loss_type == 'arcface' else None,
        device=device,
        margin=args.margin,
        loss_type=args.loss_type
    )
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        if trainer.classification_head and 'classification_head_state_dict' in checkpoint:
            trainer.classification_head.load_state_dict(checkpoint['classification_head_state_dict'])
    
    # Create dataloaders
    train_loader, val_loader = trainer.create_dataloaders(
        train_df,
        val_df,
        label_encoder,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Train
    os.makedirs(args.output_dir, exist_ok=True)
    
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_dir=args.output_dir,
        save_best=True
    )
    
    print("\nTraining completed!")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
    print(f"\nModel saved to: {args.output_dir}")


if __name__ == '__main__':
    main()



