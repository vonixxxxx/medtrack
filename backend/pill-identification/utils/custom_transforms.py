"""
Custom transforms to avoid torchvision compatibility issues
"""

import torch
import numpy as np
from PIL import Image
import random
import numbers


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Resize:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img):
        return img.resize(self.size, Image.BILINEAR)


class CenterCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img):
        w, h = img.size
        th, tw = self.size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return img.crop((j, i, j + tw, i + th))


class ToTensor:
    def __call__(self, img):
        img_array = np.array(img, dtype=np.float32)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=0)
        elif len(img_array.shape) == 3:
            img_array = img_array.transpose((2, 0, 1))
        img_array = img_array / 255.0
        return torch.FloatTensor(img_array)


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    
    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees
    
    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle, Image.BILINEAR, expand=False)


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, img):
        # Simplified - just return image for now
        return img





