import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
import numpy as np
from sklearn.metrics import f1_score, recall_score
from dataset import get_dataloaders
from model import BinaryResNet18

def train(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    cache_images = args.cache_images
    freeze_backbone = args.freeze_backbone
    monitor_metric = args.monitor # accuracy, f1, or recall

    # Data loaders
    csv_file = r'x:\Code\Classifier\archive\ham10000\metadata.csv'
    image_dir = r'x:\Code\Classifier\archive\ham10000'
    
    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_file=csv_file, 
        image_dir=image_dir, 
        batch_size=batch_size, 
        cache_images=cache_images
    )

    # Model
    print(f"Initializing Model (Freeze Backbone: {freeze_backbone})...")
    model = BinaryResNet18(pretrained=True, freeze_backbone=freeze_backbone).to(device)

    # Loss and optimizer
    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability.
    # pos_weight helps with class imbalance.
    if args.pos_weight > 1.0:
        pos_weight = torch.tensor([args.pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using BCEWithLogitsLoss with pos_weight: {args.pos_weight}")
    else:
        criterion = nn.BCEWithLogitsLoss() 
    
    # Optimizer: Only optimize parameters that have requires_grad=True, for efficiency
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Metrics tracking
    best_value = 0 # Generalized best value for monitoring
    
    # Early Stopping parameters
    patience = args.patience
    patience_counter = 0

    print(f"Starting Training (Monitoring: {monitor_metric})...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]')
        
        for i, (images, labels) in enumerate(loop):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1) # für direkten vergleich
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy
            # Use 0.0 threshold for logits (equivalent to 0.5 for sigmoid)
            predicted = (outputs > 0.0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100 * train_correct / train_total)

        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (outputs > 0.0).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_val_preds.extend(predicted.cpu().numpy()) #daten zurückholen und für f1 storen
                all_val_labels.extend(labels.cpu().numpy())

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate extra metrics
        all_val_preds = np.array(all_val_preds).flatten()
        all_val_labels = np.array(all_val_labels).flatten()
        
        val_f1 = f1_score(all_val_labels, all_val_preds)
        val_recall = recall_score(all_val_labels, all_val_preds)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] Summary:')
        print(f' - Train: Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f' - Val:   Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}, Recall: {val_recall:.4f}')

        # Determine current value for monitoring
        if monitor_metric == 'f1':
            current_value = val_f1
        elif monitor_metric == 'recall':
            current_value = val_recall
        else:
            current_value = val_acc / 100.0 # Standardize to 0-1

        # Save best model and check early stopping
        if current_value > best_value:
            best_value = current_value
            patience_counter = 0 # Reset counter
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"--> New best model saved! ({monitor_metric}: {current_value:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement in {monitor_metric}. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stop! No improvement for {patience} epochs.")
                break

    print('Finished Training')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Binary Classifier')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--cache_images', action='store_true', help='Cache images in RAM', default=True)
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze ResNet18 backbone')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (epochs)')
    parser.add_argument('--pos_weight', type=float, default=4.5, help='Weight for positive class (Malignant)') #weight 4.5 wegen bildergewichtung
    parser.add_argument('--monitor', type=str, default='f1', choices=['accuracy', 'f1', 'recall'], help='Metric to monitor for early stopping, accuracy, f1, or recall')
    
    args = parser.parse_args()
    train(args)
