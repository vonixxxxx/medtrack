"""
Image preprocessing utilities for pill identification
Based on ePillID-benchmark preprocessing pipeline
"""

import torch
from PIL import Image
import numpy as np
from typing import Union, Tuple

# Use custom transforms to avoid torchvision issues
from .custom_transforms import Compose, Resize, ToTensor, Normalize

# Create transforms namespace
class transforms:
    Compose = Compose
    Resize = Resize
    ToTensor = ToTensor
    Normalize = Normalize


# ImageNet normalization constants (used in ePillID)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_preprocessing_transform(
    image_size: int = 224,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> transforms.Compose:
    """
    Get preprocessing transform for inference.
    
    Args:
        image_size: Target image size (square)
        mean: Normalization mean values
        std: Normalization std values
    
    Returns:
        torchvision transform
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def preprocess_image(
    image: Union[Image.Image, np.ndarray, str, bytes],
    image_size: int = 224,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> torch.Tensor:
    """
    Preprocess an image for model inference.
    
    Args:
        image: Input image (PIL Image, numpy array, file path, or bytes)
        image_size: Target image size
        mean: Normalization mean values
        std: Normalization std values
    
    Returns:
        Preprocessed tensor ready for model input
    """
    # Load image if needed
    if isinstance(image, str):
        # File path
        img = Image.open(image).convert('RGB')
    elif isinstance(image, bytes):
        # Bytes data
        from io import BytesIO
        img = Image.open(BytesIO(image)).convert('RGB')
    elif isinstance(image, np.ndarray):
        # Numpy array
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        img = Image.fromarray(image).convert('RGB')
    elif isinstance(image, Image.Image):
        # Already a PIL Image
        img = image.convert('RGB')
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Apply transforms
    transform = get_preprocessing_transform(image_size, mean, std)
    tensor = transform(img)
    
    # Add batch dimension if needed
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    return tensor


def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Decode a base64-encoded image string.
    
    Args:
        base64_string: Base64-encoded image (with or without data URL prefix)
    
    Returns:
        PIL Image
    """
    import base64
    from io import BytesIO
    
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    image_data = base64.b64decode(base64_string)
    
    # Open as image
    img = Image.open(BytesIO(image_data)).convert('RGB')
    
    return img



