# MedMNIST Classification Pipeline

An end-to-end medical imaging pipeline for classifying images from the [MedMNIST](https://github.com/MedMNIST/MedMNIST) dataset. This repository demonstrates how to load, preprocess, train, and evaluate a deep learning model on PathMNIST data. The pipeline uses a pretrained EfficientNet-B0 model, fine-tuning it for the classification of 9 classes. 

## Repository Structure
```
MedMNIST_Classification 
├── data/ # Downloaded MedMNIST data 
├── models/ # Saved model checkpoints 
├── src/ 
│ ├── evaluation.py # Evaluate and visualize model performance
│ ├── preprocessing.py # Data loading and preprocessing 
│ └── model.py # EfficientNet-B0 fine-tuned
├── train.py # Main execution script
├── requirements.txt
└── README.md
```

## Requirements
- Python 3.8+
- CUDA-capable GPU with 6GB+ VRAM (recommended)
    - CUDA Toolkit installed and added to system PATH

## Installation
1. Clone the repository: `git clone https://github.com/Poofy1/medmnist-classification.git`
2. Install dependencies: `pip install -r requirements.txt`


## Usage
### Run training script: ```python train.py```

This will:
  - Download the PathMNIST dataset (if not already present in `./data`)
  - Train the model for the specified number of epochs
  - Saves the best model weights in `./models/best_model.pth`
  - Evaluate on the test dataset using the best saved weights


### Optional Parameters
- `--batch_size` (default: 32)
- `--learning_rate` (default: 0.001)
- `--num_epochs` (default: 5)
- `--skip_training` (flag to skip training and only evaluate using saved model)

Example: `python train.py --batch_size 64 --learning_rate 0.0001 --num_epochs 2`



## Model Performance Results

### Training Metrics
- Accuracy: 98.06%
- Loss: 0.0579

### Validation Metrics
- Accuracy: 97.67%
- Loss: 0.0684

### Test Metrics
- Accuracy: 89.71%
- Loss: 0.3836
- F1 Score: 86.61%
- Precision: 87.28%
- Recall: 87.42%
- AUC Mean: 99.02%

### Test Performance Visualizations
![Confusion Matrix](/results/confusion_matrix.png)
![AUC Per Class](/results/auc_per_class.png)
![Training and Validation Loss](/results/loss.png)



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
