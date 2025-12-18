"""
Embedding Model for Pill Identification
Based on ePillID-benchmark architecture

This module implements the CNN encoder with bilinear pooling
followed by an embedding head for generating pill embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Try to import torchvision models, with fallback
try:
    import torchvision.models as models
except (ImportError, ValueError) as e:
    print(f"Warning: torchvision.models import failed ({e})")
    print("Using custom ResNet loader instead")
    from .resnet_loader import resnet18, resnet34, resnet50
    models = None


class GlobalAveragePooling(nn.Module):
    """Global Average Pooling representation"""
    def __init__(self, dimension_reduction: Optional[int] = None):
        super(GlobalAveragePooling, self).__init__()
        self.dimension_reduction = dimension_reduction
        if dimension_reduction:
            self.fc = nn.Linear(512, dimension_reduction)
        else:
            self.fc = None
    
    def forward(self, x):
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        if self.fc:
            x = self.fc(x)
        return x


class EmbeddingModel(nn.Module):
    """
    Embedding model that takes a CNN backbone and produces embeddings.
    
    Architecture:
    - CNN backbone (ResNet18/50, etc.)
    - Global Average Pooling (or other pooling methods)
    - Dropout
    - Embedding head (optional)
    """
    
    def __init__(
        self,
        network: str = 'resnet18',
        pooling: str = 'GAvP',
        dropout_p: float = 0.5,
        embedding_dim: int = 2048,
        pretrained: bool = True,
        middle_dim: int = 1000,
        skip_emb: bool = False
    ):
        super(EmbeddingModel, self).__init__()
        
        # Load backbone CNN
        if models is None:
            # Use custom ResNet loader
            if network == 'resnet18':
                backbone = resnet18(pretrained=pretrained)
                backbone_features = 512
            elif network == 'resnet50':
                backbone = resnet50(pretrained=pretrained)
                backbone_features = 2048
            elif network == 'resnet34':
                backbone = resnet34(pretrained=pretrained)
                backbone_features = 512
            else:
                raise ValueError(f"Unsupported network: {network}")
        else:
            if network == 'resnet18':
                backbone = models.resnet18(pretrained=pretrained)
                backbone_features = 512
            elif network == 'resnet50':
                backbone = models.resnet50(pretrained=pretrained)
                backbone_features = 2048
            elif network == 'resnet34':
                backbone = models.resnet34(pretrained=pretrained)
                backbone_features = 512
            else:
                raise ValueError(f"Unsupported network: {network}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Pooling layer
        if pooling == 'GAvP':
            # Global Average Pooling with optional dimension reduction
            self.pooling = GlobalAveragePooling(dimension_reduction=backbone_features)
            pool_output_dim = backbone_features
        else:
            raise ValueError(f"Unsupported pooling: {pooling}. Only GAvP is currently supported.")
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_p)
        
        # Embedding head
        self.out_features = embedding_dim
        if skip_emb:
            self.emb = None
        else:
            self.emb = nn.Sequential(
                nn.Linear(pool_output_dim, middle_dim),
                nn.BatchNorm1d(middle_dim, affine=True),
                nn.ReLU(inplace=True),
                nn.Linear(middle_dim, embedding_dim),
                nn.Tanh()
            )
    
    def forward(self, x):
        """Forward pass through the model"""
        # Extract features from backbone
        x = self.backbone(x)
        
        # Pooling
        x = self.pooling(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Embedding head
        if self.emb is not None:
            x = self.emb(x)
        
        return x
    
    def get_embedding(self, x):
        """Get embedding for input image"""
        return self.forward(x)


def l2_norm(x):
    """L2 normalize embeddings"""
    return F.normalize(x, p=2, dim=1)


def get_model(
    network: str = 'resnet18',
    pooling: str = 'GAvP',
    embedding_dim: int = 2048,
    dropout_p: float = 0.5,
    pretrained: bool = True,
    model_path: Optional[str] = None,
    device: str = 'cpu'
) -> EmbeddingModel:
    """
    Get or load an embedding model.
    
    Args:
        network: CNN backbone architecture ('resnet18', 'resnet50', etc.)
        pooling: Pooling method ('GAvP')
        embedding_dim: Dimension of output embeddings
        dropout_p: Dropout probability
        pretrained: Whether to use pretrained weights for backbone
        model_path: Path to saved model weights (.pth file)
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        EmbeddingModel in eval mode
    """
    model = EmbeddingModel(
        network=network,
        pooling=pooling,
        dropout_p=dropout_p,
        embedding_dim=embedding_dim,
        pretrained=pretrained
    )
    
    # Load weights if provided
    if model_path:
        try:
            state_dict = torch.load(model_path, map_location=device)
            # Handle both full state dict and just the embedding model
            if 'embedding_model' in state_dict:
                model.load_state_dict(state_dict['embedding_model'])
            elif 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            else:
                model.load_state_dict(state_dict)
            print(f"Loaded model weights from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model weights from {model_path}: {e}")
            print("Using pretrained backbone only.")
    
    model = model.to(device)
    model.eval()
    
    return model



