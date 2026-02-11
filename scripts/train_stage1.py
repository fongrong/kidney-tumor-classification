"""
Train Stage 1: Kidney Detection with EfficientNet
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import json

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.dataset import KidneyDetectionDataset


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100 * correct / total
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def main(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = KidneyDetectionDataset(
        image_dir=os.path.join(args.data_dir, 'train/images'),
        labels_csv=os.path.join(args.data_dir, 'train/labels.csv'),
        transform=train_transform
    )
    
    val_dataset = KidneyDetectionDataset(
        image_dir=os.path.join(args.data_dir, 'val/images'),
        labels_csv=os.path.join(args.data_dir, 'val/labels.csv'),
        transform=val_transform
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f'Train dataset: {len(train_dataset)} images')
    print(f'Val dataset: {len(val_dataset)} images')
    
    # Model
    print(f'Loading model: {args.model}')
    model = EfficientNet.from_pretrained(args.model, num_classes=2)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Training
    best_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f'\nTraining for {args.epochs} epochs...\n')
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        print('-' * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f'✓ Best model saved (Val Acc: {val_acc:.2f}%)')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break
        
        print()
    
    # Save final model
    torch.save(model.state_dict(), 
              os.path.join(args.output_dir, 'final_model.pth'))
    
    # Save training history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'\n✓ Training completed!')
    print(f'Best validation accuracy: {best_acc:.2f}%')
    print(f'Models saved to: {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Stage 1 Model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Data directory')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='efficientnet-b7',
                       help='EfficientNet model variant')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage1',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
