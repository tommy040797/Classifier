import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
from dataset import get_dataloaders
from model import BinaryResNet18

def evaluate(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = args.batch_size
    cache_images = args.cache_images

    # Paths
    csv_file = r'x:\Code\Classifier\archive\ham10000\metadata.csv'
    image_dir = r'x:\Code\Classifier\archive\ham10000'
    model_path = args.model_path
    
    # Load DataLoaders
    # We only need the test loader
    _, _, test_loader = get_dataloaders(
        csv_file=csv_file, 
        image_dir=image_dir, 
        batch_size=batch_size, 
        cache_images=cache_images
    )

    # Load Model
    # Important: Initialize with the EXACT SAME arguments as used during training
    # We can inspect the state dict or assume the user knows.
    # Our default was freeze_backbone depends on training. But architecture is the same.
    # Loading weights works regardless of freeze_backbone (which only affects requires_grad).
    model = BinaryResNet18(pretrained=False) # No need to download ImageNet weights again if we load ours
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: Model file {model_path} not found!")
        return

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Starting Evaluation on Test Set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            
            # Threshold at 0.3 for binary classification
            preds = (probs > 0.3).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Flatten lists
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print("\n" + "="*30)
    print("Evaluation Results")
    print("="*30)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("-" * 30)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Benign (0)', 'Malignant (1)']))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predict Benign', 'Predict Malignant'], yticklabels=['Actual Benign', 'Actual Malignant'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to 'confusion_matrix.png'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Binary Classifier')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--cache_images', action='store_true', help='Cache images in RAM')
    
    args = parser.parse_args()
    evaluate(args)
