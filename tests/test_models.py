"""
Tests for model modules
"""

import pytest
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_efficientnet_creation():
    """Test EfficientNet model creation"""
    from src.models.efficientnet import EfficientNetClassifier
    
    model = EfficientNetClassifier(
        model_name='efficientnet-b0',
        num_classes=2,
        pretrained=False
    )
    
    assert model.num_classes == 2
    assert model.get_num_parameters() > 0


def test_efficientnet_forward():
    """Test forward pass"""
    from src.models.efficientnet import EfficientNetClassifier
    
    model = EfficientNetClassifier(
        model_name='efficientnet-b0',
        num_classes=2,
        pretrained=False
    )
    
    # Test input
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    assert output.shape == (2, 2), "Output shape should be (batch_size, num_classes)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
