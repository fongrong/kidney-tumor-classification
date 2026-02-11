"""
Evaluation metrics for kidney tumor classification
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, average_precision_score
)


def calculate_metrics(y_true, y_pred, num_classes=5, class_names=None):
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        class_names: List of class names
    
    Returns:
        dict: Dictionary of metrics
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Overall metrics
    accuracy = 100 * accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Weighted averages
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    metrics = {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'class_accuracy': class_acc,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm
    }
    
    # Classification report
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    metrics['classification_report'] = report
    
    return metrics


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for bounding boxes
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
    
    Returns:
        float: IoU value
    """
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # IoU
    iou = intersection / union if union > 0 else 0
    
    return iou


def calculate_map(predictions, ground_truths, iou_threshold=0.5, num_classes=5):
    """
    Calculate mean Average Precision (mAP) for object detection
    
    Args:
        predictions: List of predicted boxes and scores
        ground_truths: List of ground truth boxes and labels
        iou_threshold: IoU threshold for positive detection
        num_classes: Number of classes
    
    Returns:
        dict: mAP metrics
    """
    aps = []
    
    for class_id in range(num_classes):
        # Get predictions and ground truths for this class
        class_preds = [p for p in predictions if p['class_id'] == class_id]
        class_gts = [gt for gt in ground_truths if gt['class_id'] == class_id]
        
        if len(class_gts) == 0:
            continue
        
        # Sort predictions by confidence
        class_preds = sorted(class_preds, key=lambda x: x['confidence'], reverse=True)
        
        # Calculate precision and recall
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        
        gt_detected = set()
        
        for i, pred in enumerate(class_preds):
            max_iou = 0
            max_gt_idx = -1
            
            for gt_idx, gt in enumerate(class_gts):
                if gt_idx in gt_detected:
                    continue
                
                iou = calculate_iou(pred['box'], gt['box'])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            if max_iou >= iou_threshold and max_gt_idx != -1:
                tp[i] = 1
                gt_detected.add(max_gt_idx)
            else:
                fp[i] = 1
        
        # Cumulative sums
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Precision and recall
        recalls = tp_cumsum / len(class_gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Average Precision (area under PR curve)
        ap = np.trapz(precisions, recalls)
        aps.append(ap)
    
    # Mean Average Precision
    mAP = np.mean(aps) if aps else 0
    
    return {
        'mAP': mAP,
        'per_class_AP': aps
    }


def sensitivity_specificity(y_true, y_pred, positive_class=1):
    """
    Calculate sensitivity and specificity
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        positive_class: Label for positive class
    
    Returns:
        tuple: (sensitivity, specificity)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # For binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        # For multi-class, calculate for positive class
        tp = cm[positive_class, positive_class]
        fn = cm[positive_class, :].sum() - tp
        fp = cm[:, positive_class].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return sensitivity, specificity


def ppv_npv(y_true, y_pred, positive_class=1):
    """
    Calculate Positive Predictive Value (PPV) and Negative Predictive Value (NPV)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        positive_class: Label for positive class
    
    Returns:
        tuple: (ppv, npv)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    else:
        tp = cm[positive_class, positive_class]
        fn = cm[positive_class, :].sum() - tp
        fp = cm[:, positive_class].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return ppv, npv


if __name__ == '__main__':
    # Test metrics
    print("Testing metrics calculation...\n")
    
    # Generate dummy data
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4] * 10)
    y_pred = np.array([0, 1, 2, 3, 3, 0, 1, 1, 3, 4] * 10)  # Some errors
    
    class_names = ['kidney', 'ccRCC', 'pRCC', 'chRCC', 'ONC']
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, num_classes=5, class_names=class_names)
    
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1-Score (weighted): {metrics['f1_weighted']:.4f}")
    
    print("\nPer-class metrics:")
    for i, name in enumerate(class_names):
        print(f"  {name}:")
        print(f"    Precision: {metrics['precision_per_class'][i]:.4f}")
        print(f"    Recall: {metrics['recall_per_class'][i]:.4f}")
        print(f"    F1-Score: {metrics['f1_per_class'][i]:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Test IoU
    box1 = [10, 10, 50, 50]
    box2 = [30, 30, 70, 70]
    iou = calculate_iou(box1, box2)
    print(f"\nIoU between boxes: {iou:.4f}")
    
    print("\n✓ All metrics calculated successfully")
