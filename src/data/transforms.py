"""
Data augmentation and transforms for kidney tumor classification
"""

import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=640, method='pytorch'):
    """
    Get training data transforms
    
    Args:
        image_size (int): Target image size
        method (str): 'pytorch' or 'albumentations'
    
    Returns:
        Transform composition
    """
    if method == 'pytorch':
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
    
    elif method == 'albumentations':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=0,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            ], p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


def get_val_transforms(image_size=640, method='pytorch'):
    """
    Get validation/test data transforms
    
    Args:
        image_size (int): Target image size
        method (str): 'pytorch' or 'albumentations'
    
    Returns:
        Transform composition
    """
    if method == 'pytorch':
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
    
    elif method == 'albumentations':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


def get_yolo_transforms(image_size=640):
    """
    Get transforms for YOLO training
    
    Args:
        image_size (int): Target image size
    
    Returns:
        Albumentations composition for YOLO
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
        ], p=0.5),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))


class Denormalize:
    """Denormalize image for visualization"""
    
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        """
        Args:
            tensor: Normalized tensor image
        
        Returns:
            Denormalized tensor
        """
        import torch
        
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        
        return tensor


if __name__ == '__main__':
    import torch
    from PIL import Image
    import numpy as np
    
    # Test transforms
    print("Testing data transforms...\n")
    
    # Create dummy image
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    )
    
    # PyTorch transforms
    print("PyTorch transforms:")
    train_transform = get_train_transforms(method='pytorch')
    val_transform = get_val_transforms(method='pytorch')
    
    train_tensor = train_transform(dummy_image)
    val_tensor = val_transform(dummy_image)
    
    print(f"  Train output shape: {train_tensor.shape}")
    print(f"  Val output shape: {val_tensor.shape}")
    
    # Albumentations transforms
    print("\nAlbumentations transforms:")
    train_transform_alb = get_train_transforms(method='albumentations')
    val_transform_alb = get_val_transforms(method='albumentations')
    
    dummy_array = np.array(dummy_image)
    train_result = train_transform_alb(image=dummy_array)
    val_result = val_transform_alb(image=dummy_array)
    
    print(f"  Train output shape: {train_result['image'].shape}")
    print(f"  Val output shape: {val_result['image'].shape}")
    
    print("\n✓ All transforms working correctly")
