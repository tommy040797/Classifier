import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import torch

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, cache_images=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            cache_images (bool): If True, loads all images into RAM during initialization.
        """
        self.metadata = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.cache_images = cache_images
        self.images_cache = {}

        # Define class mapping
        # Class 0: Benign
        # Class 1: Malignant
        # Filter out Indeterminate
        self.metadata = self.metadata[self.metadata['diagnosis_1'].isin(['Benign', 'Malignant'])]
        
        # Filter metadata to include only rows where image exists
        self.metadata = self.metadata[self.metadata['isic_id'].apply(self._check_image_exists)]  
        self.metadata.reset_index(drop=True, inplace=True)

        #image caching, for fun, doesnt really do anything but i feel better
        if self.cache_images:
            print("Caching images into RAM...")
            for idx in tqdm(range(len(self.metadata))):
                img_name = os.path.join(self.image_dir, self.metadata.iloc[idx]['isic_id'] + '.jpg')
                image = Image.open(img_name).convert('RGB')
                self.images_cache[idx] = image
            print("Caching complete.")

    def _check_image_exists(self, image_id):
        img_name = os.path.join(self.image_dir, image_id + '.jpg')
        return os.path.exists(img_name)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # If idx is tensor, make it a number
        if torch.is_tensor(idx):
            idx = idx.item()
            
        if self.cache_images:
            #get Cached Image
            image = self.images_cache[idx]
        else:
            #get Image from disk
            img_name = os.path.join(self.image_dir, self.metadata.iloc[idx]['isic_id'] + '.jpg')
            image = Image.open(img_name).convert('RGB')

        #get Diagnosis
        diagnosis = self.metadata.iloc[idx]['diagnosis_1']
        
        # Binary label: Benign -> 0, Malignant -> 1
        label = 1 if diagnosis == 'Malignant' else 0

        #transform image for training or eval, depending on given parameter
        if self.transform:
            image = self.transform(image)

        return image, label

"""def get_dataloaders(csv_file, image_dir, batch_size=32, val_split=0.1, test_split=0.1, num_workers=4, cache_images=False):
    import torchvision.transforms as transforms
    from torch.utils.data import Subset

    # Define transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Training transform with augmentation
    train_transform = transforms.Compose([
        transforms.Resize(256),             # Resize shortest side to 256
        transforms.RandomCrop(224),         # Random crop for training (augmentation)
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),    # Added VerticalFlip (useful for skin lesions)
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Validation/Test transform (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),         # Center crop for evaluation
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 1. Load full dataset (no transform yet)
    full_dataset = HAM10000Dataset(csv_file, image_dir, transform=None, cache_images=cache_images)
    
    # 2. Split indices
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    # Use generator for reproducibility
    train_subset, val_subset, test_subset = random_split(
        full_dataset, 
        [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # 3. Create wrapper to apply transforms
    class TransformedSubset(Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, idx):
            x, y = self.subset[idx] 
            if self.transform:
                x = self.transform(x)
            return x, y
        
        def __len__(self):
            return len(self.subset)

    # 4. Wrap subsets
    train_data = TransformedSubset(train_subset, transform=train_transform)
    val_data = TransformedSubset(val_subset, transform=eval_transform)
    test_data = TransformedSubset(test_subset, transform=eval_transform)

    # 5. Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
"""