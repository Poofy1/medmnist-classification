from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import get_model
from src.preprocessing import get_data_loaders
import time
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training', leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device).long().squeeze()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating', leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device).long().squeeze()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(val_loader)
    return avg_loss, accuracy

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 2

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(batch_size)

    # Initialize model, criterion, optimizer
    model = get_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create directory for saving models
    save_dir = Path(__file__).parent / 'models'
    save_dir.mkdir(exist_ok=True)

    # Training loop
    best_val_acc = 0
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time

        # Print metrics
        print(f'Time: {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / 'best_model.pth')

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Final test evaluation
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(save_dir / 'best_model.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')

if __name__ == "__main__":
    main()