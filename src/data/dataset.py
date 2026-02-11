"""
Dataset classes for kidney tumor classification
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class KidneyDetectionDataset(Dataset):
    """
    Dataset for Stage 1: Binary kidney detection
    
    Args:
        image_dir (str): Directory containing images
        labels_csv (str): Path to CSV file with labels
        transform (callable, optional): Optional transform to be applied on images
    """
    
    def __init__(self, image_dir, labels_csv, transform=None):
        self.image_dir = image_dir
        self.labels = pd.read_csv(labels_csv)
        self.transform = transform
        
        # Binary labels: 0 = no kidney, 1 = kidney
        self.classes = ['no_kidney', 'kidney']
        self.num_classes = 2
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get image path and label
        img_name = self.labels.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)
        label = int(self.labels.iloc[idx, 1])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        class_counts = self.labels.iloc[:, 1].value_counts().sort_index()
        total = len(self.labels)
        weights = torch.FloatTensor([total / (self.num_classes * count) 
                                     for count in class_counts])
        return weights


class TumorClassificationDataset(Dataset):
    """
    Dataset for Stage 2: Multi-class tumor classification
    
    Args:
        image_dir (str): Directory containing images
        labels_dir (str): Directory containing YOLO format labels
        transform (callable, optional): Optional transform to be applied
    """
    
    def __init__(self, image_dir, labels_dir=None, transform=None):
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Class names
        self.classes = ['kidney', 'ccRCC', 'pRCC', 'chRCC', 'ONC']
        self.num_classes = 5
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load labels if available
        if self.labels_dir:
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_name)
            
            boxes = []
            labels = []
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        # YOLO format: class x_center y_center width height
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert to [x1, y1, x2, y2] format
                            x1 = x_center - width / 2
                            y1 = y_center - height / 2
                            x2 = x_center + width / 2
                            y2 = y_center + height / 2
                            
                            boxes.append([x1, y1, x2, y2])
                            labels.append(class_id)
            
            boxes = torch.FloatTensor(boxes) if boxes else torch.zeros((0, 4))
            labels = torch.LongTensor(labels) if labels else torch.zeros((0,), dtype=torch.long)
        else:
            boxes = None
            labels = None
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if boxes is not None:
            return image, {'boxes': boxes, 'labels': labels}
        else:
            return image
    
    def collate_fn(self, batch):
        """Custom collate function for batching"""
        images = []
        targets = []
        
        for item in batch:
            if isinstance(item, tuple):
                images.append(item[0])
                targets.append(item[1])
            else:
                images.append(item)
        
        images = torch.stack(images, 0)
        
        if targets:
            return images, targets
        else:
            return images


class KiTS19Dataset(Dataset):
    """
    KiTS19 dataset loader for CT images
    
    Args:
        data_dir (str): Root directory of KiTS19 dataset
        split (str): 'train', 'val', or 'test'
        transform (callable, optional): Optional transform
    """
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load split file
        split_file = os.path.join(data_dir, f'{split}.csv')
        if os.path.exists(split_file):
            self.samples = pd.read_csv(split_file)
        else:
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        self.num_classes = 5
        self.classes = ['kidney', 'ccRCC', 'pRCC', 'chRCC', 'ONC']
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Get sample info
        case_id = self.samples.iloc[idx]['case_id']
        slice_idx = self.samples.iloc[idx]['slice_idx']
        tumor_type = self.samples.iloc[idx]['tumor_type']
        
        # Load image
        img_path = os.path.join(self.data_dir, 'images', 
                               f'case_{case_id:05d}_slice_{slice_idx:03d}.png')
        image = Image.open(img_path).convert('RGB')
        
        # Convert tumor type to class index
        if tumor_type in self.classes:
            label = self.classes.index(tumor_type)
        else:
            label = 0  # Default to kidney
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label, case_id


def get_train_val_split(data_csv, val_ratio=0.2, random_seed=42):
    """
    Split dataset into train and validation sets with stratification
    
    Args:
        data_csv (str): Path to data CSV file
        val_ratio (float): Validation set ratio
        random_seed (int): Random seed for reproducibility
    
    Returns:
        train_df, val_df: Training and validation dataframes
    """
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(data_csv)
    
    # Stratified split based on tumor type
    train_df, val_df = train_test_split(
        df, 
        test_size=val_ratio,
        stratify=df['tumor_type'],
        random_state=random_seed
    )
    
    return train_df, val_df


if __name__ == '__main__':
    # Example usage
    print("Dataset module loaded successfully")
    
    # Test KidneyDetectionDataset
    print("\nTesting KidneyDetectionDataset...")
    # dataset = KidneyDetectionDataset(
    #     image_dir='data/images',
    #     labels_csv='data/labels.csv'
    # )
    # print(f"Dataset size: {len(dataset)}")
    # print(f"Number of classes: {dataset.num_classes}")
    # print(f"Classes: {dataset.classes}")
