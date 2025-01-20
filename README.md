# MedMNIST Classification Pipeline

An end-to-end medical imaging pipeline for classifying images from the [MedMNIST](https://github.com/MedMNIST/MedMNIST) dataset. This repository demonstrates how to load, preprocess, train, and evaluate a deep learning model on PathMNIST data. The pipeline uses a pretrained EfficientNet-B0 model, fine-tuning it for the classification of 9 classes. 

## Repository Structure
```
MedMNIST_Classification 
├── data/ # Downloaded MedMNIST data 
├── models/ # Saved model checkpoints 
├── src/ 
│ ├── preprocessing.py # Data loading and preprocessing 
│ └── model.py # EfficientNet-B0 fine-tuned
├── train.py # Training and evaluation script
├── requirements.txt
└── README.md
```

## Setup
(Built using Python 3.8 - results may vary with other versions.)
1. Clone the repository: `git clone https://github.com/Poofy1/medmnist-classification.git`
2. Install dependencies: `pip install -r requirements.txt`

## Usage
Run training script: ```python train.py```

This will:
  - Download the PathMNIST dataset (if not already present in `./data`).
  - Train the model for the specified number of epochs (default: 2).
  - Saves the best model weights in `./models/best_model.pth`.
  - Evaluate on the test dataset using the best saved weights.

## Results
- Training Accuracy: 92.21%
- Validation Accuracy: 97.82%
- Test Accuracy: 91.78%

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
