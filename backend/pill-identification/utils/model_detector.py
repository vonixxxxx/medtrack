"""
Utility to detect existing trained models
"""

import os
from pathlib import Path
from typing import Optional, List
import torch


def find_existing_model(search_dirs: List[str] = None) -> Optional[str]:
    """
    Find existing trained model in project directories.
    
    Args:
        search_dirs: List of directories to search (default: common locations)
    
    Returns:
        Path to model file if found, None otherwise
    """
    if search_dirs is None:
        # Default search locations
        base_dir = Path(__file__).parent.parent.parent
        search_dirs = [
            str(base_dir / 'pill-identification' / 'models'),
            str(base_dir / 'models'),
            str(base_dir / 'ml-service' / 'pretrained-models'),
            str(base_dir / 'backend' / 'ml-service' / 'pretrained-models'),
        ]
    
    model_extensions = ['.pth', '.pt', '.pkl']
    model_keywords = ['pill', 'epillid', 'embedding', 'model', 'checkpoint']
    
    found_models = []
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                file_lower = file.lower()
                
                # Check if it's a model file
                if any(file.endswith(ext) for ext in model_extensions):
                    # Check if it contains relevant keywords
                    if any(keyword in file_lower for keyword in model_keywords):
                        file_path = os.path.join(root, file)
                        
                        # Try to load and validate
                        try:
                            checkpoint = torch.load(file_path, map_location='cpu')
                            
                            # Check if it has model state dict
                            if 'model_state_dict' in checkpoint or 'state_dict' in checkpoint:
                                found_models.append((file_path, os.path.getsize(file_path)))
                        except:
                            pass
    
    if not found_models:
        return None
    
    # Return the largest file (likely the most complete model)
    found_models.sort(key=lambda x: x[1], reverse=True)
    return found_models[0][0]


def validate_model(model_path: str, expected_network: str = None) -> bool:
    """
    Validate that a model file is compatible.
    
    Args:
        model_path: Path to model file
        expected_network: Expected network architecture (optional)
    
    Returns:
        True if model is valid, False otherwise
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check for required keys
        has_state_dict = 'model_state_dict' in checkpoint or 'state_dict' in checkpoint
        
        if not has_state_dict:
            return False
        
        # Check network if specified
        if expected_network:
            model_network = checkpoint.get('network', None)
            if model_network and model_network != expected_network:
                print(f"Warning: Model network ({model_network}) != expected ({expected_network})")
                # Still return True as it might work
        
        return True
    except Exception as e:
        print(f"Error validating model: {e}")
        return False







