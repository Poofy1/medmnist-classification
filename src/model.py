import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class MedMNISTModel(nn.Module):
    def __init__(self, num_classes=9):  # PathMNIST has 9 classes
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        self.efficient_net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Modify first layer to accept three channels (keeping default)
        self.efficient_net.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Replace classifier with new one for our number of classes
        num_features = self.efficient_net.classifier[1].in_features
        self.efficient_net.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        # EfficientNet expects input size of 224x224
        if x.shape[-1] != 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        return self.efficient_net(x)

def get_model(device):
    model = MedMNISTModel()
    model = model.to(device)
    return model

if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    x = torch.randn(1, 3, 28, 28).to(device)  # MedMNIST image size
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [1, 9]