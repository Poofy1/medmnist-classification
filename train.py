from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import get_model
from src.preprocessing import get_data_loaders
from src.evaluation import MetricsTracker
import time
from tqdm import tqdm
import argparse

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MedMNIST classifier')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--skip_training', action='store_true', help='skip training and evaluate existing model')
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(args.batch_size)

    # Initialize model
    model = get_model(device)
    
    # Create directories
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    metrics_tracker = MetricsTracker(results_dir)
    
    save_dir = Path(__file__).parent / 'models'
    save_dir.mkdir(exist_ok=True)
    model_path = save_dir / 'best_model.pth'

    if not args.skip_training:
        # Initialize criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        # Training loop
        best_val_acc = 0
        for epoch in range(args.num_epochs):
            print(f'\nEpoch [{epoch+1}/{args.num_epochs}]')
            start_time = time.time()
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            epoch_time = time.time() - start_time
            
            # Update metrics tracker
            metrics_tracker.update_epoch_metrics(train_loss, val_loss, train_acc, val_acc, epoch_time)
            
            # Print metrics
            print(f'Time: {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_path)
        
        # Plot training history
        metrics_tracker.plot_history()
        print("\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")

    else:
        print("\nSkipping training, loading existing model...")
        
        # Check if model exists
        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}. Please train the model first or provide a trained model.")
        

    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    criterion = nn.CrossEntropyLoss()  # Need criterion for evaluation
    test_loss, test_acc, test_metrics = metrics_tracker.evaluate_model(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
    for metric_name, value in test_metrics.items():
        print(f'{metric_name}: {value*100:.2f}%')

if __name__ == "__main__":
    main()