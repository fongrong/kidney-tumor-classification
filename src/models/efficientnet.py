"""
EfficientNet model for kidney detection
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-based classifier for kidney detection
    
    Args:
        model_name (str): EfficientNet variant ('efficientnet-b0' to 'efficientnet-b7')
        num_classes (int): Number of output classes
        pretrained (bool): Use ImageNet pre-trained weights
    """
    
    def __init__(self, model_name='efficientnet-b7', num_classes=2, pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained EfficientNet
        if pretrained:
            self.model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        else:
            self.model = EfficientNet.from_name(model_name, num_classes=num_classes)
        
        # Get model info
        self.in_channels = self.model._conv_stem.in_channels
        self.out_channels = self.model._fc.in_features
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    
    def extract_features(self, x):
        """Extract features before classification layer"""
        return self.model.extract_features(x)
    
    def get_num_parameters(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self):
        """Freeze all layers except the final classifier"""
        for name, param in self.model.named_parameters():
            if '_fc' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    @staticmethod
    def get_model_info(model_name='efficientnet-b7'):
        """
        Get model architecture information
        
        Returns:
            dict: Model specifications
        """
        specs = {
            'efficientnet-b0': {'params': 5.3, 'flops': 0.39, 'input': 224},
            'efficientnet-b1': {'params': 7.8, 'flops': 0.70, 'input': 240},
            'efficientnet-b2': {'params': 9.2, 'flops': 1.0, 'input': 260},
            'efficientnet-b3': {'params': 12.0, 'flops': 1.8, 'input': 300},
            'efficientnet-b4': {'params': 19.3, 'flops': 4.2, 'input': 380},
            'efficientnet-b5': {'params': 30.4, 'flops': 9.9, 'input': 456},
            'efficientnet-b6': {'params': 43.0, 'flops': 19.0, 'input': 528},
            'efficientnet-b7': {'params': 66.3, 'flops': 37.0, 'input': 600},
        }
        return specs.get(model_name, {})


def create_efficientnet(model_name='efficientnet-b7', num_classes=2, 
                       pretrained=True, device='cuda'):
    """
    Create and initialize EfficientNet model
    
    Args:
        model_name (str): Model variant
        num_classes (int): Number of classes
        pretrained (bool): Use pre-trained weights
        device (str): Device to load model on
    
    Returns:
        model: Initialized EfficientNet model
    """
    model = EfficientNetClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained
    )
    
    model = model.to(device)
    
    print(f"Created {model_name}")
    print(f"Total parameters: {model.get_num_parameters():,}")
    print(f"Trainable parameters: {model.get_trainable_parameters():,}")
    
    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing EfficientNet model creation...\n")
    
    # Create model
    model = create_efficientnet(
        model_name='efficientnet-b7',
        num_classes=2,
        pretrained=False,
        device='cpu'
    )
    
    # Test forward pass
    x = torch.randn(1, 3, 600, 600)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Get model info
    info = EfficientNetClassifier.get_model_info('efficientnet-b7')
    print(f"\nModel info: {info}")
