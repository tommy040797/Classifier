import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import HAM10000Dataset
from torch.utils.data import DataLoader

def denormalize(tensor, mean, std):
    """Reverses the normalization on a tensor."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def visualize():
    # Paths
    csv_file = r'x:\Code\Classifier\archive\ham10000\metadata.csv'
    image_dir = r'x:\Code\Classifier\archive\ham10000'
    
    # Transforms (see here https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize(256),             # Resize shortest side to 256
        transforms.RandomCrop(224),  
        transforms.RandomHorizontalFlip(), # Show augmentation too
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load dataset (no cache needed for visualization)
    dataset = HAM10000Dataset(csv_file=csv_file, image_dir=image_dir, transform=transform, cache_images=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Get a batch
    images, labels = next(iter(dataloader))

    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("Data Loading & Preprocessing Check (Label 0: Benign, 1: Malignant)")

    for i, ax in enumerate(axes.flat):
        img_tensor = denormalize(images[i], mean, std)
        img_np = img_tensor.numpy().transpose((1, 2, 0))
        img_np = np.clip(img_np, 0, 1) # Clip to valid range
        
        ax.imshow(img_np)
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis('off')

    output_path = r'x:\Code\Classifier\preprocessing_sample.png'
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == '__main__':
    visualize()
