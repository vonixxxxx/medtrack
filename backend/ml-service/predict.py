#!/usr/bin/env python3
"""
Pill Recognition ML Service
Based on Confir-Med livemodel.py
Predicts medication from pill image using PyTorch CNN model
"""

import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# Medicine classes from Confir-Med
medicines = [
    'Alaxan',
    'Bactidol',
    'Bioflu',
    'Biogesic',
    'DayZinc',
    'Fish Oil',
    'Kremil S',
    'Medicol',
    'Neozep',
]

# Define the CNN model architecture (same as Confir-Med)
class CNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2))
        )
        dummy_input = torch.zeros(1, 3, 100, 100)
        in_features = self.feature_extractor(dummy_input).view(1, -1).size(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def crop_to_square(image):
    """Crops the input PIL image to a square based on the shorter side."""
    width, height = image.size
    side_length = min(width, height)
    left = (width - side_length) // 2
    top = (height - side_length) // 2
    right = left + side_length
    bottom = top + side_length
    return image.crop((left, top, right, bottom))

def predict(image_path, model_path):
    """Predicts the medicine from an image file."""
    try:
        # Load model
        model = CNN(num_classes=9)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        # Preprocessing
        test_transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor()
        ])

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = crop_to_square(image)
        image = test_transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

        predicted_medicine = medicines[predicted.item()]
        confidence_value = confidence.item()

        return {
            'predicted_medicine': predicted_medicine,
            'confidence': f"{100 * confidence_value:.2f}%"
        }
    except Exception as e:
        return {
            'error': str(e),
            'predicted_medicine': 'Unknown',
            'confidence': '0%'
        }

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(json.dumps({
            'error': 'Usage: python predict.py <image_path> <model_path>'
        }))
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2]

    result = predict(image_path, model_path)
    print(json.dumps(result))



