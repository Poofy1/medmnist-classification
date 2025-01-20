import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple

class MetricsTracker:
    def __init__(self, save_dir):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epoch_times': []
        }
        self.current_epoch_preds = []
        self.current_epoch_targets = []
        self.save_dir = save_dir
    
    def update_epoch_metrics(self, train_loss: float, val_loss: float, 
                           train_acc: float, val_acc: float, epoch_time: float):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['epoch_times'].append(epoch_time)

    def update_batch_predictions(self, predictions: torch.Tensor, targets: torch.Tensor):
        self.current_epoch_preds.extend(predictions.cpu().numpy())
        self.current_epoch_targets.extend(targets.cpu().numpy())
    
    def compute_epoch_metrics(self) -> Dict[str, float]:
        preds = np.array(self.current_epoch_preds)
        targets = np.array(self.current_epoch_targets)
        
        # Reset for next epoch
        self.current_epoch_preds = []
        self.current_epoch_targets = []
        
        metrics = {
            'f1_score': f1_score(targets, preds, average='macro'),
            'precision': precision_score(targets, preds, average='macro'),
            'recall': recall_score(targets, preds, average='macro'),
        }
        
        return metrics

    def plot_history(self):
        # Create epoch numbers starting from 1
        epochs = list(range(1, len(self.history['train_loss']) + 1))
        
        # Loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['train_loss'], label='Train Loss')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.save_dir / 'loss.png')
        plt.close()
        
        # Accuracy plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['train_acc'], label='Train Accuracy')
        plt.plot(epochs, self.history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.savefig(self.save_dir / 'accuracy.png')
        plt.close()
        
    def plot_confusion_matrix(self, y_true, y_pred):
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Create labels for each class (0 to num_classes-1)
        classes = list(range(len(cm)))
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix.png')
        plt.close()
        
        
    def plot_auc_scores(self, auc_scores, class_names=None):
        plt.figure(figsize=(12, 6))
        x = range(len(auc_scores))
        plt.bar(x, auc_scores)
        plt.axhline(y=np.mean(auc_scores), color='r', linestyle='--', 
                    label=f'Mean AUC: {np.mean(auc_scores):.3f}')
        
        # Add value labels on top of each bar
        for i, v in enumerate(auc_scores):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.xlabel('Class')
        plt.ylabel('AUC Score')
        plt.title('ROC-AUC Score per Class')
        
        # Use class names if provided, otherwise use numbers
        if class_names:
            plt.xticks(x, class_names, rotation=45)
        else:
            plt.xticks(x)
            
        plt.legend(loc='lower right')  # Changed this line to move legend
        plt.tight_layout()
        plt.savefig(self.save_dir / 'auc_per_class.png')
        plt.close()

    def evaluate_model(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                    criterion: torch.nn.Module, device: torch.device) -> Tuple[float, float, Dict[str, float]]:
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []  # Store probabilities for AUC calculation
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).long().squeeze()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_probs.append(probabilities.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Convert to numpy arrays
        all_probs = np.vstack(all_probs)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate base metrics
        metrics = {
            'f1_score': f1_score(all_targets, all_preds, average='macro'),
            'precision': precision_score(all_targets, all_preds, average='macro'),
            'recall': recall_score(all_targets, all_preds, average='macro'),
        }
        
        # Calculate per-class AUC and plot
        plt.figure(figsize=(10, 6))
        auc_scores = []
        
        for i in range(all_probs.shape[1]):
            # One-vs-rest AUC for each class
            binary_targets = (all_targets == i).astype(int)
            auc = roc_auc_score(binary_targets, all_probs[:, i])
            auc_scores.append(auc)
        
        # Plot AUC scores
        self.plot_auc_scores(auc_scores)
        
        # Add mean AUC to metrics
        metrics['auc_mean'] = np.mean(auc_scores)
        
        # confusion matrix
        self.plot_confusion_matrix(all_targets, all_preds)
        
        # Calculate accuracy
        accuracy = 100 * np.mean(all_preds == all_targets)
        avg_loss = running_loss / len(data_loader)
        
        return avg_loss, accuracy, metrics