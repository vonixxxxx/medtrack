import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import io

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

# Define the CNN model architecture (the same as the training code)
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
        # Adjust dummy input to match image size (100x100)
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

# Initialize and load the model weights
model = CNN(num_classes=9)
model.load_state_dict(torch.load(r'pretrained-models/cnn2.pth'))
model.eval()  # Set to evaluation mode

# Preprocessing for the input image
test_transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Ensure this matches the model input size
    transforms.ToTensor()
])

def crop_to_square(image):
    """
    Crops the input PIL image to a square based on the shorter side.
    """
    width, height = image.size
    side_length = min(width, height)
    left = (width - side_length) // 2
    top = (height - side_length) // 2
    right = left + side_length
    bottom = top + side_length
    return image.crop((left, top, right, bottom))

def predict(image_bytes, model):
    """
    Predicts the medicine from an image byte stream.
    """
    image = Image.open(io.BytesIO(image_bytes))
    image = crop_to_square(image)  # Crop to square before resizing
    image = test_transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
    return medicines[predicted.item()], confidence.item()