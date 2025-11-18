"""
Pill Recognition Service for MedTrack
Based on Confir-Med ML model
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import sys
import json

# Medicine classes (from Confir-Med)
MEDICINES = [
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

# Define the CNN model architecture
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

# Load model
def load_model(model_path='pretrained-models/cnn2.pth'):
    model = CNN(num_classes=9)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Preprocessing
test_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

def crop_to_square(image):
    """Crops the input PIL image to a square based on the shorter side."""
    width, height = image.size
    side_length = min(width, height)
    left = (width - side_length) // 2
    top = (height - side_length) // 2
    right = left + side_length
    bottom = top + side_length
    return image.crop((left, top, right, bottom))

def predict_from_bytes(image_bytes, model):
    """Predicts the medicine from an image byte stream."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = crop_to_square(image)
        image = test_transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
        
        predicted_medicine = MEDICINES[predicted.item()]
        confidence_score = confidence.item()
        
        return {
            'predicted_medicine': predicted_medicine,
            'confidence': confidence_score,
            'confidence_percent': f"{100 * confidence_score:.2f}%"
        }
    except Exception as e:
        return {
            'error': str(e),
            'predicted_medicine': None,
            'confidence': 0.0
        }

if __name__ == '__main__':
    # Command-line interface for testing
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Image path required'}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'pretrained-models/cnn2.pth'
    
    try:
        model = load_model(model_path)
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        result = predict_from_bytes(image_bytes, model)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({'error': str(e)}))



