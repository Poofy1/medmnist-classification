from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST

def get_data_loaders(batch_size=32):
    # Make data dir
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Define data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create datasets
    train_dataset = PathMNIST(root=str(data_dir), split='train', transform=transform, download=True)
    val_dataset = PathMNIST(root=str(data_dir), split='val', transform=transform, download=True)
    test_dataset = PathMNIST(root=str(data_dir), split='test', transform=transform, download=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Quick test
    train_loader, val_loader, test_loader = get_data_loaders()
    print("Data loaders created successfully!")