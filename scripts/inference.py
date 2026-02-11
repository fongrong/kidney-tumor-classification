"""
Inference script for kidney tumor classification
"""

import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.efficientnet import create_efficientnet
from src.preprocessing.jet_enhancement import CTPreprocessor
from src.data.transforms import get_val_transforms


def load_stage1_model(weights_path, device='cuda'):
    """Load Stage 1 model"""
    model = create_efficientnet(
        model_name='efficientnet-b7',
        num_classes=2,
        pretrained=False,
        device=device
    )
    
    checkpoint = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def predict_kidney(image_path, model, preprocessor, transform, device='cuda'):
    """
    Predict if image contains kidney
    
    Returns:
        (has_kidney, confidence)
    """
    # Preprocess
    image = preprocessor.preprocess(image_path)
    image = Image.fromarray(image)
    
    # Transform
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probabilities, dim=1)
        confidence = probabilities[0, predicted].item()
    
    has_kidney = predicted.item() == 1
    
    return has_kidney, confidence


def predict_tumor(image_path, yolo_weights, device='cuda'):
    """
    Predict tumor subtypes using YOLOv7
    
    Returns:
        List of detections with boxes, classes, and confidences
    """
    # This is a placeholder - actual implementation would use YOLOv7
    # You would need to integrate with the YOLOv7 detect.py script
    
    print(f"Running YOLOv7 detection on {image_path}...")
    print(f"Using weights: {yolo_weights}")
    
    # Placeholder return
    detections = []
    
    return detections


def visualize_results(image_path, detections, output_path):
    """Visualize detection results"""
    image = cv2.imread(image_path)
    
    # Class names
    class_names = ['kidney', 'ccRCC', 'pRCC', 'chRCC', 'ONC']
    colors = [
        (0, 255, 0),    # Green for kidney
        (255, 0, 0),    # Red for ccRCC
        (0, 0, 255),    # Blue for pRCC
        (255, 255, 0),  # Yellow for chRCC
        (255, 0, 255)   # Magenta for ONC
    ]
    
    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = det['box']
        class_id = det['class']
        confidence = det['confidence']
        
        color = colors[class_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_names[class_id]}: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(output_path, image)
    print(f"✓ Results saved to: {output_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize preprocessor and transforms
    preprocessor = CTPreprocessor(method='jet')
    transform = get_val_transforms(image_size=640)
    
    # Load Stage 1 model
    print("Loading Stage 1 model...")
    stage1_model = load_stage1_model(args.stage1_weights, device)
    
    # Process single image or directory
    if os.path.isfile(args.image):
        image_paths = [args.image]
    else:
        image_paths = [os.path.join(args.image, f) 
                      for f in os.listdir(args.image)
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\nProcessing {len(image_paths)} images...\n")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process each image
    results = []
    
    for img_path in image_paths:
        print(f"Processing: {img_path}")
        
        # Stage 1: Kidney detection
        has_kidney, confidence = predict_kidney(
            img_path, stage1_model, preprocessor, transform, device
        )
        
        print(f"  Kidney detected: {has_kidney} (confidence: {confidence:.4f})")
        
        if has_kidney and args.stage2_weights:
            # Stage 2: Tumor classification
            detections = predict_tumor(img_path, args.stage2_weights, device)
            print(f"  Found {len(detections)} tumor(s)")
            
            # Visualize
            output_path = os.path.join(
                args.output, 
                os.path.basename(img_path).replace('.', '_result.')
            )
            visualize_results(img_path, detections, output_path)
        
        results.append({
            'image': img_path,
            'has_kidney': has_kidney,
            'confidence': confidence,
            'detections': detections if has_kidney and args.stage2_weights else []
        })
        
        print()
    
    # Save results
    import json
    results_file = os.path.join(args.output, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Inference completed!")
    print(f"✓ Results saved to: {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kidney Tumor Classification Inference')
    
    parser.add_argument('--image', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--stage1_weights', type=str, required=True,
                       help='Stage 1 model weights')
    parser.add_argument('--stage2_weights', type=str, default=None,
                       help='Stage 2 (YOLOv7) weights')
    parser.add_argument('--output', type=str, default='results/',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    main(args)
