"""
Tests for preprocessing module
"""

import pytest
import numpy as np
from PIL import Image
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing.jet_enhancement import CTPreprocessor


def test_jet_preprocessor():
    """Test JET colormap preprocessing"""
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    
    # Initialize preprocessor
    preprocessor = CTPreprocessor(method='jet', target_size=(640, 640))
    
    # Preprocess
    result = preprocessor.preprocess(dummy_image)
    
    # Assertions
    assert result.shape == (640, 640, 3), "Output should be 640x640x3"
    assert result.dtype == np.uint8, "Output should be uint8"
    assert result.max() <= 255, "Values should be <= 255"
    assert result.min() >= 0, "Values should be >= 0"


def test_gray_preprocessor():
    """Test grayscale preprocessing"""
    dummy_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    
    preprocessor = CTPreprocessor(method='gray', target_size=(640, 640))
    result = preprocessor.preprocess(dummy_image)
    
    assert result.shape == (640, 640, 3)


def test_hsv_preprocessor():
    """Test HSV preprocessing"""
    dummy_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    
    preprocessor = CTPreprocessor(method='hsv', target_size=(640, 640))
    result = preprocessor.preprocess(dummy_image)
    
    assert result.shape == (640, 640, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
