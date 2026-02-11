"""
Data preprocessing script
"""

import os
import argparse
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing.jet_enhancement import CTPreprocessor


def main(args):
    """Main preprocessing function"""
    print(f"Preprocessing CT images...")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Method: {args.method}")
    print(f"Size: {args.size}x{args.size}")
    print()
    
    # Initialize preprocessor
    preprocessor = CTPreprocessor(
        method=args.method,
        target_size=(args.size, args.size)
    )
    
    # Batch process
    preprocessor.batch_preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_extension=args.ext
    )
    
    print("\n✓ Preprocessing completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess CT images')
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing CT images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for preprocessed images')
    parser.add_argument('--method', type=str, default='jet',
                       choices=['gray', 'jet', 'hsv'],
                       help='Preprocessing method')
    parser.add_argument('--size', type=int, default=640,
                       help='Target image size')
    parser.add_argument('--ext', type=str, default='.jpg',
                       help='Image file extension')
    
    args = parser.parse_args()
    main(args)
