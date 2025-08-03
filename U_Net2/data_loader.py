# data_loader.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob

class LGGMRIDataset(Dataset):
    """Dataset for LGG MRI brain tumor segmentation"""
    
    def __init__(self, data_dir, transform=None, target_transform=None, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        
        # Load the metadata CSV
        self.metadata = pd.read_csv(os.path.join(data_dir, 'data.csv'))
        
        # Get all patient directories
        patient_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('TCGA')]
        
        # Create train/val split
        train_patients, val_patients = train_test_split(patient_dirs, test_size=0.2, random_state=42)
        
        if split == 'train':
            self.patient_dirs = train_patients
        else:
            self.patient_dirs = val_patients
        
        # Build image and mask paths
        self.images = []
        self.masks = []
        
        for patient in self.patient_dirs:
            patient_path = os.path.join(data_dir, patient)
            
            # Find all image files in the patient directory
            image_files = glob.glob(os.path.join(patient_path, '*.tif'))
            
            for img_file in image_files:
                # Determine if it's an image or mask based on filename
                if '_mask' in img_file.lower():
                    # This is a mask file
                    mask_file = img_file
                    # Find corresponding image file
                    img_file = img_file.replace('_mask', '').replace('_Mask', '')
                    
                    if os.path.exists(img_file):
                        self.images.append(img_file)
                        self.masks.append(mask_file)
                else:
                    # This is an image file, check if corresponding mask exists
                    mask_file = img_file.replace('.tif', '_mask.tif')
                    if os.path.exists(mask_file):
                        self.images.append(img_file)
                        self.masks.append(mask_file)
        
        print(f"📊 Loaded {len(self.images)} {split} samples")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Load mask
        mask_path = self.masks[idx]
        mask = Image.open(mask_path).convert('L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask

def get_lgg_transforms():
    """Get transforms for LGG MRI dataset"""
    
    # Image transforms
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Mask transforms (no normalization for binary masks)
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    return image_transform, mask_transform

def create_data_loaders(data_dir, batch_size=4, train_split=0.8):
    """Create train and validation data loaders"""
    
    # Get transforms
    image_transform, mask_transform = get_lgg_transforms()
    
    # Create datasets
    train_dataset = LGGMRIDataset(
        data_dir, 
        transform=image_transform, 
        target_transform=mask_transform,
        split='train'
    )
    
    val_dataset = LGGMRIDataset(
        data_dir, 
        transform=image_transform, 
        target_transform=mask_transform,
        split='val'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader

def visualize_sample(dataset, idx=0):
    """Visualize a sample from the dataset"""
    image, mask = dataset[idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Show original image
    ax1.imshow(image.squeeze(), cmap='gray')
    ax1.set_title('MRI Image')
    ax1.axis('off')
    
    # Show mask
    ax2.imshow(mask.squeeze(), cmap='gray')
    ax2.set_title('Tumor Mask')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test the data loader
    data_dir = "data/lgg-mri-segmentation/kaggle_3m"
    train_loader, val_loader = create_data_loaders(data_dir, batch_size=2)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Visualize a sample
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Images shape: {images.shape}, Masks shape: {masks.shape}")
        break