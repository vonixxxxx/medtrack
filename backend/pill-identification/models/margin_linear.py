"""
Margin-based linear layer for ArcFace loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MarginLinear(nn.Module):
    """
    Margin-based linear layer for ArcFace loss.
    
    Implements:
    W * x / ||W|| / ||x|| * s + m * y
    where y is the one-hot encoding of the label
    """
    
    def __init__(self, embedding_size: int, classnum: int, s: float = 64.0, m: float = 0.5):
        super(MarginLinear, self).__init__()
        self.embedding_size = embedding_size
        self.classnum = classnum
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(classnum, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute cosine of margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m
        self.threshold = math.cos(math.pi - m)
    
    def forward(self, input, label, is_infer=False):
        """
        Forward pass.
        
        Args:
            input: Input embeddings (batch_size, embedding_size)
            label: Class labels (batch_size,)
            is_infer: Whether in inference mode
        """
        # Normalize weights and input
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        if is_infer:
            # For inference, just return cosine similarity
            return cosine * self.s
        
        # For training, apply margin
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        
        # One-hot encode labels
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Apply margin only to correct class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output


def l2_norm(x):
    """L2 normalize"""
    return F.normalize(x, p=2, dim=1)







