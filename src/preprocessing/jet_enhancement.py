"""
Image preprocessing methods for kidney tumor CT images
"""

import cv2
import numpy as np
from PIL import Image


class CTPreprocessor:
    """
    CT image preprocessing with multiple enhancement methods
    
    Supported methods:
    - gray: Standard grayscale
    - jet: JET colormap enhancement
    - hsv: HSV color space transformation
    """
    
    def __init__(self, method='jet', target_size=(640, 640)):
        """
        Args:
            method (str): Preprocessing method ('gray', 'jet', 'hsv')
            target_size (tuple): Target image size (height, width)
        """
        self.method = method.lower()
        self.target_size = target_size
        
        if self.method not in ['gray', 'jet', 'hsv']:
            raise ValueError(f"Unknown method: {method}. "
                           f"Choose from 'gray', 'jet', or 'hsv'")
    
    def preprocess(self, image):
        """
        Preprocess a single image
        
        Args:
            image: Input image (PIL Image, numpy array, or path)
        
        Returns:
            Preprocessed image as numpy array
        """
        # Convert to numpy array if needed
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('L'))
        
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize
        image = cv2.resize(image, self.target_size, 
                          interpolation=cv2.INTER_LINEAR)
        
        # Apply preprocessing method
        if self.method == 'gray':
            result = self._preprocess_gray(image)
        elif self.method == 'jet':
            result = self._preprocess_jet(image)
        elif self.method == 'hsv':
            result = self._preprocess_hsv(image)
        
        return result
    
    def _preprocess_gray(self, image):
        """Keep as grayscale, convert to RGB"""
        # Convert to 3-channel for consistency
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return result
    
    def _preprocess_jet(self, image):
        """
        Apply JET colormap enhancement
        
        Maps grayscale intensity to rainbow color spectrum:
        - Blue: Low intensity
        - Cyan/Green: Medium intensity  
        - Yellow/Red: High intensity
        """
        # Apply JET colormap
        jet_image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        
        # Convert BGR to RGB
        result = cv2.cvtColor(jet_image, cv2.COLOR_BGR2RGB)
        
        return result
    
    def _preprocess_hsv(self, image):
        """
        Transform to HSV color space
        
        Separates:
        - Hue: Color type
        - Saturation: Color intensity
        - Value: Brightness
        """
        # Convert grayscale to RGB first
        gray_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Convert to HSV
        hsv = cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2HSV)
        
        # Enhance saturation channel using histogram equalization
        hsv[:, :, 1] = cv2.equalizeHist(hsv[:, :, 1])
        
        # Convert back to RGB for visualization
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return result
    
    def batch_preprocess(self, input_dir, output_dir, file_extension='.jpg'):
        """
        Preprocess all images in a directory
        
        Args:
            input_dir (str): Input directory path
            output_dir (str): Output directory path
            file_extension (str): Image file extension
        """
        import os
        from pathlib import Path
        from tqdm import tqdm
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(input_dir) 
                      if f.endswith(file_extension)]
        
        print(f"Processing {len(image_files)} images with method: {self.method}")
        
        # Process each image
        for img_file in tqdm(image_files):
            input_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, img_file)
            
            # Preprocess
            processed = self.preprocess(input_path)
            
            # Save
            cv2.imwrite(output_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
        
        print(f"Completed! Processed images saved to: {output_dir}")


class DICOMPreprocessor:
    """
    Preprocess DICOM CT images
    """
    
    def __init__(self, window_center=40, window_width=400):
        """
        Args:
            window_center (int): Windowing center (HU)
            window_width (int): Windowing width (HU)
        """
        self.window_center = window_center
        self.window_width = window_width
    
    def load_dicom(self, dicom_path):
        """Load DICOM file"""
        try:
            import pydicom
            dcm = pydicom.dcmread(dicom_path)
            image = dcm.pixel_array
            
            # Apply modality LUT
            if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                image = image * dcm.RescaleSlope + dcm.RescaleIntercept
            
            return image
        except ImportError:
            raise ImportError("pydicom is required for DICOM processing. "
                            "Install it with: pip install pydicom")
    
    def apply_windowing(self, image):
        """
        Apply CT windowing
        
        Args:
            image: CT image in Hounsfield Units
        
        Returns:
            Windowed image scaled to 0-255
        """
        min_value = self.window_center - self.window_width / 2
        max_value = self.window_center + self.window_width / 2
        
        # Clip values
        image = np.clip(image, min_value, max_value)
        
        # Scale to 0-255
        image = ((image - min_value) / (max_value - min_value) * 255.0)
        image = image.astype(np.uint8)
        
        return image
    
    def preprocess_dicom(self, dicom_path, output_path, method='jet'):
        """
        Complete DICOM preprocessing pipeline
        
        Args:
            dicom_path (str): Path to DICOM file
            output_path (str): Path to save preprocessed image
            method (str): Enhancement method
        """
        # Load DICOM
        image = self.load_dicom(dicom_path)
        
        # Apply windowing
        windowed = self.apply_windowing(image)
        
        # Apply enhancement
        preprocessor = CTPreprocessor(method=method)
        enhanced = preprocessor.preprocess(windowed)
        
        # Save
        cv2.imwrite(output_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
        
        return enhanced


def normalize_hounsfield(image, min_hu=-1000, max_hu=400):
    """
    Normalize Hounsfield Units to 0-1 range
    
    Args:
        image: CT image in HU
        min_hu: Minimum HU value
        max_hu: Maximum HU value
    
    Returns:
        Normalized image
    """
    image = np.clip(image, min_hu, max_hu)
    image = (image - min_hu) / (max_hu - min_hu)
    return image


def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        image: Input grayscale image
        clip_limit: Contrast limiting threshold
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        Enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                            tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess CT images')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory or file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory or file')
    parser.add_argument('--method', type=str, default='jet',
                       choices=['gray', 'jet', 'hsv'],
                       help='Preprocessing method')
    parser.add_argument('--size', type=int, default=640,
                       help='Target image size')
    parser.add_argument('--batch', action='store_true',
                       help='Batch process directory')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = CTPreprocessor(
        method=args.method,
        target_size=(args.size, args.size)
    )
    
    # Process
    if args.batch:
        preprocessor.batch_preprocess(args.input, args.output)
    else:
        result = preprocessor.preprocess(args.input)
        cv2.imwrite(args.output, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"Preprocessed image saved to: {args.output}")
