"""
Training pipeline for ePillID-based pill identification model
Implements modern PyTorch 2.x training with metric learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Use custom transforms to avoid torchvision compatibility issues
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.custom_transforms import (
    Compose, Resize, CenterCrop, ToTensor, Normalize,
    RandomHorizontalFlip, RandomRotation, ColorJitter
)

# Create transforms namespace
class transforms:
    Compose = Compose
    Resize = Resize
    CenterCrop = CenterCrop
    ToTensor = ToTensor
    Normalize = Normalize
    RandomHorizontalFlip = RandomHorizontalFlip
    RandomRotation = RandomRotation
    ColorJitter = ColorJitter
import pandas as pd
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import json
from typing import Dict, Optional, Tuple
import pickle
from sklearn.preprocessing import LabelEncoder

from models.embedding_model import EmbeddingModel
from utils.preprocessing import get_preprocessing_transform, IMAGENET_MEAN, IMAGENET_STD

def l2_norm(x):
    """L2 normalize"""
    import torch.nn.functional as F
    return F.normalize(x, p=2, dim=1)


class PillDataset(Dataset):
    """Dataset for pill images"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        label_encoder: LabelEncoder,
        transform=None,
        is_training: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = row['image_path']
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            img = Image.new('RGB', (224, 224))
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Get label
        label = row['label']
        label_idx = self.label_encoder.transform([label])[0]
        
        return {
            'image': img,
            'label': label_idx,
            'label_str': label,
            'image_path': img_path
        }


class TripletLoss(nn.Module):
    """Triplet loss for metric learning"""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings
            positive: Positive (same class) embeddings
            negative: Negative (different class) embeddings
        """
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class MetricLearningTrainer:
    """
    Trainer for metric learning-based pill identification.
    
    Uses:
    - CNN encoder (ResNet18/50)
    - Global Average Pooling
    - Metric learning projection head
    - L2-normalized embeddings
    - Triplet loss or ArcFace loss
    """
    
    def __init__(
        self,
        network: str = 'resnet18',
        embedding_dim: int = 2048,
        num_classes: Optional[int] = None,
        device: Optional[str] = None,
        margin: float = 1.0,
        loss_type: str = 'triplet'
    ):
        """
        Initialize trainer.
        
        Args:
            network: CNN backbone ('resnet18', 'resnet50', etc.)
            embedding_dim: Embedding dimension
            num_classes: Number of classes (for classification head)
            device: Device to use ('cpu' or 'cuda')
            margin: Margin for triplet loss
            loss_type: Loss type ('triplet' or 'arcface')
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        self.network = network
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.loss_type = loss_type
        
        # Initialize model
        self.model = EmbeddingModel(
            network=network,
            pooling='GAvP',
            embedding_dim=embedding_dim,
            pretrained=True
        ).to(device)
        
        # Classification head for ArcFace
        if loss_type == 'arcface' and num_classes:
            from ..models.margin_linear import MarginLinear
            self.classification_head = MarginLinear(
                embedding_size=embedding_dim,
                classnum=num_classes,
                s=64.0,
                m=0.5
            ).to(device)
        else:
            self.classification_head = None
        
        # Loss functions
        if loss_type == 'triplet':
            self.criterion = TripletLoss(margin=margin)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def create_dataloaders(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        label_encoder: LabelEncoder,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders"""
        
        # Transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        # Datasets
        train_dataset = PillDataset(train_df, label_encoder, train_transform, is_training=True)
        val_dataset = PillDataset(val_df, label_encoder, val_transform, is_training=False)
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.model(images)
            embeddings = l2_norm(embeddings)  # L2 normalize embeddings
            
            # Compute loss
            if self.loss_type == 'triplet':
                # Simple triplet mining: use batch as triplets
                # In practice, you'd want more sophisticated mining
                loss = self._compute_triplet_loss_simple(embeddings, labels)
            else:
                # ArcFace loss
                logits = self.classification_head(embeddings, labels, is_infer=False)
                loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def _compute_triplet_loss_simple(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Simple triplet loss computation from batch"""
        # Find positive and negative pairs
        n = embeddings.size(0)
        if n < 3:
            # Return a small dummy loss that requires grad
            dummy = embeddings.mean() * 0.0
            return dummy
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings)
        
        # Create mask for same/different labels
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Find hardest positive and negative for each anchor
        losses = []
        for i in range(n):
            anchor = embeddings[i]
            anchor_label = labels[i]
            
            # Positive: same label, different sample
            pos_mask = label_matrix[i] & (torch.arange(n, device=self.device) != i)
            if not pos_mask.any():
                continue
            
            pos_distances = distances[i][pos_mask]
            pos_indices = pos_mask.nonzero().squeeze(1)  # Get all positive indices
            if len(pos_indices) == 0:
                continue
            
            # Find the hardest positive (largest distance = most different, which is hardest)
            hardest_pos_idx_in_pos_list = pos_distances.argmax()
            if hardest_pos_idx_in_pos_list >= len(pos_indices):
                hardest_pos_idx_in_pos_list = len(pos_indices) - 1
            pos_idx = pos_indices[hardest_pos_idx_in_pos_list]
            positive = embeddings[pos_idx]
            
            # Negative: different label
            neg_mask = ~label_matrix[i]
            if not neg_mask.any():
                continue
            
            neg_distances = distances[i][neg_mask]
            neg_indices = neg_mask.nonzero().squeeze(1)  # Get all negative indices
            if len(neg_indices) == 0:
                continue
            
            # Find the hardest negative (smallest distance = most similar, which is hardest)
            hardest_neg_idx_in_neg_list = neg_distances.argmin()
            if hardest_neg_idx_in_neg_list >= len(neg_indices):
                hardest_neg_idx_in_neg_list = len(neg_indices) - 1
            neg_idx = neg_indices[hardest_neg_idx_in_neg_list]
            negative = embeddings[neg_idx]
            
            # Triplet loss
            loss = torch.relu(
                (anchor - positive).pow(2).sum() -
                (anchor - negative).pow(2).sum() +
                self.margin
            )
            losses.append(loss)
        
        if losses:
            return torch.stack(losses).mean()
        else:
            # Return a small dummy loss that requires grad
            dummy = embeddings.mean() * 0.0
            return dummy
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                embeddings = self.model(images)
                embeddings = l2_norm(embeddings)  # L2 normalize embeddings
                
                # Compute loss
                if self.loss_type == 'triplet':
                    loss = self._compute_triplet_loss_simple(embeddings, labels)
                else:
                    logits = self.classification_head(embeddings, labels, is_infer=False)
                    loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Accuracy (using nearest neighbor in embedding space)
                if self.loss_type == 'triplet':
                    # For triplet loss, we compute accuracy via nearest neighbor
                    # This is a simplified version
                    pass
                else:
                    _, predicted = logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        save_dir: Optional[str] = None,
        save_best: bool = True
    ):
        """
        Train the model.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            save_dir: Directory to save checkpoints
            save_best: Whether to save best model
        """
        # Optimizer
        optimizer = optim.Adam(
            list(self.model.parameters()) + 
            (list(self.classification_head.parameters()) if self.classification_head else []),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        best_val_loss = float('inf')
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Network: {self.network}")
        print(f"Embedding dim: {self.embedding_dim}")
        print(f"Loss type: {self.loss_type}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_accuracy = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Save checkpoint
            if save_dir:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'history': self.history
                }
                
                if self.classification_head:
                    checkpoint['classification_head_state_dict'] = self.classification_head.state_dict()
                
                # Save latest
                torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_latest.pth'))
                
                # Save best
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_best.pth'))
                    print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save final model
        if save_dir:
            final_model_path = os.path.join(save_dir, 'model_final.pth')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'embedding_dim': self.embedding_dim,
                'network': self.network,
                'history': self.history
            }, final_model_path)
            print(f"\nFinal model saved to {final_model_path}")
        
        return self.history

