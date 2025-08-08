#!/usr/bin/env python3
"""
Demo script for Two-Stage Nuclei Analysis Pipeline

This script demonstrates the complete pipeline by:
1. Loading a sample from the PanNuke dataset 
2. Running it through both segmentation and classification stages
3. Comparing results with ground truth
4. Creating comprehensive visualizations
"""

import sys
import os
import platform

# Add parent directory to path to access root-level modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from two_stage_pipeline import TwoStageNucleiPipeline, find_best_models
from pannuke_dataset import PanNukeDataset
from nuclei_extraction import extract_nuclei_from_prediction, visualize_extracted_nuclei
import albumentations as A

# Fix for Windows multiprocessing
if platform.system() == 'Windows':
    torch.multiprocessing.set_start_method('spawn', force=True)


def load_sample_from_dataset(dataset_path: str = "data", 
                           fold: int = 1, 
                           sample_idx: int = 0,
                           image_type: str = 'images') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a sample from the PanNuke dataset for demonstration.
    
    Args:
        dataset_path: Path to PanNuke dataset directory
        fold: Dataset fold (1, 2, or 3)
        sample_idx: Index of sample to load
        image_type: Type of images to load ('images' or 'masks')
        
    Returns:
        Tuple of (image, instance_mask, type_mask)
    """
    print(f"📊 Loading sample {sample_idx} from PanNuke fold {fold}...")
    
    try:
        # Try to use the PanNuke dataset loader
        # Map fold to part name
        fold_part_map = {1: "Part 1", 2: "Part 2", 3: "Part 3"}
        parts_to_load = [fold_part_map.get(fold, "Part 1")]
        
        dataset = PanNukeDataset(
            root=dataset_path,
            parts=parts_to_load,
            augmentations=None,  # No transforms for demo
            validate_dataset=False  # Skip validation for faster loading
        )
        
        # Get a sample
        if sample_idx >= len(dataset):
            sample_idx = len(dataset) - 1
            print(f"   Adjusted sample index to {sample_idx} (max available)")
        
        sample = dataset[sample_idx]
        image = sample['image']
        mask = sample['mask']  # Instance segmentation
        
        # Convert tensors to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.numpy()
            if image.shape[0] == 3:  # CHW -> HWC
                image = np.transpose(image, (1, 2, 0))
        
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        # Handle normalized data from PanNuke dataset
        # PanNuke applies Z-score normalization which can result in values outside 0-1
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Denormalize from Z-score to 0-255 range for visualization
            # Clip extreme values and normalize to 0-1, then to 0-255
            image_clipped = np.clip(image, -3, 3)  # Clip to reasonable Z-score range
            image_min, image_max = image_clipped.min(), image_clipped.max()
            if image_max > image_min:
                image = ((image_clipped - image_min) / (image_max - image_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image_clipped, dtype=np.uint8)
        elif image.max() <= 1.0:
            # Standard 0-1 normalized image
            image = (image * 255).astype(np.uint8)
        
        print(f"   ✅ Loaded sample: {image.shape}, mask: {mask.shape}")
        return image, mask, mask  # Using same mask for both instance and type
        
    except Exception as e:
        print(f"   ❌ Could not load from PanNuke dataset: {e}")
        print(f"   Please ensure the dataset is available at: {dataset_path}")
        print(f"   Expected structure: {dataset_path}/Part {fold}/Images/ and {dataset_path}/Part {fold}/Masks/")
        raise RuntimeError(f"Failed to load dataset from {dataset_path}. Error: {e}")




def visualize_pipeline_results(image: np.ndarray,
                             gt_instance_mask: np.ndarray,
                             gt_type_mask: np.ndarray,
                             pipeline_results: Dict,
                             save_path: str = None,
                             show_plot: bool = True) -> plt.Figure:
    """Create comprehensive visualization comparing ground truth with pipeline results."""
    
    # Extract pipeline results
    seg_prediction = pipeline_results['segmentation']['prediction_mask'].cpu().numpy()
    extracted_nuclei = pipeline_results['extracted_nuclei']
    nucleus_classifications = pipeline_results['classifications']
    
    # Create classification mask from pipeline results using centroid and area (since masks are empty)
    pipeline_type_mask = np.zeros_like(seg_prediction, dtype=np.uint8)
    for nucleus_info in extracted_nuclei:
        nucleus_id = nucleus_info['nucleus_id']
        
        # Find classification for this nucleus
        if nucleus_id in nucleus_classifications:
            classification = nucleus_classifications[nucleus_id]
            predicted_class = classification['predicted_class'] + 1  # Convert 0-4 to 1-5
            
            # Create circular mask from centroid and area since the provided masks are empty
            centroid = nucleus_info.get('centroid', (128, 128))
            area = nucleus_info.get('area', 100)
            
            # Create circular approximation at centroid
            cy, cx = int(centroid[0]), int(centroid[1])
            radius = max(3, int(np.sqrt(area / np.pi)))  # Minimum radius of 3
            
            # Ensure centroid is within image bounds
            cy = max(radius, min(seg_prediction.shape[0] - radius, cy))
            cx = max(radius, min(seg_prediction.shape[1] - radius, cx))
            
            y, x = np.ogrid[:seg_prediction.shape[0], :seg_prediction.shape[1]]
            circle_mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            
            pipeline_type_mask[circle_mask] = predicted_class
    
    # Class names and colors
    class_names = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
    class_colors = np.array([
        [0, 0, 0],        # Background - black
        [255, 0, 0],      # Neoplastic - red
        [0, 255, 0],      # Inflammatory - green  
        [0, 0, 255],      # Connective - blue
        [255, 255, 0],    # Dead - yellow
        [255, 0, 255],    # Epithelial - magenta
    ], dtype=np.uint8)
    
    # Create colored masks
    gt_colored = class_colors[gt_type_mask]
    seg_colored = class_colors[seg_prediction]
    pipeline_colored = class_colors[pipeline_type_mask]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Two-Stage Pipeline Results vs Ground Truth', fontsize=16, fontweight='bold')
    
    # Row 1: Original data and ground truth
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_colored)
    axes[0, 1].set_title('Ground Truth\n(Type Classification)', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Show instance mask as overlay
    instance_overlay = image.copy()
    instance_contours = cv2.findContours((gt_instance_mask > 0).astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(instance_overlay, instance_contours, -1, (255, 255, 255), 2)
    axes[0, 2].imshow(instance_overlay)
    axes[0, 2].set_title('Ground Truth\n(Instance Segmentation)', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Class distribution (ground truth)
    gt_unique, gt_counts = np.unique(gt_type_mask, return_counts=True)
    gt_dist = np.zeros(len(class_names))
    for cls, count in zip(gt_unique, gt_counts):
        if cls < len(class_names):
            gt_dist[cls] = count
    
    # Remove background for cleaner visualization
    gt_dist_no_bg = gt_dist[1:]
    class_names_no_bg = class_names[1:]
    colors_no_bg = class_colors[1:] / 255.0
    
    bars_gt = axes[0, 3].bar(range(len(class_names_no_bg)), gt_dist_no_bg, 
                           color=colors_no_bg, alpha=0.8)
    axes[0, 3].set_title('Ground Truth\nClass Distribution', fontweight='bold')
    axes[0, 3].set_xlabel('Classes')
    axes[0, 3].set_ylabel('Pixel Count')
    axes[0, 3].set_xticks(range(len(class_names_no_bg)))
    axes[0, 3].set_xticklabels([name[:4] for name in class_names_no_bg], rotation=45)
    axes[0, 3].grid(True, alpha=0.3)
    
    # Row 2: Pipeline results
    axes[1, 0].imshow(seg_colored)
    axes[1, 0].set_title('Stage 1: Segmentation\nPrediction', fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pipeline_colored)
    axes[1, 1].set_title('Stage 2: Classification\nFinal Result', fontweight='bold')
    axes[1, 1].axis('off')
    
    # Show extracted nuclei with classifications
    nuclei_overlay = image.copy()
    for nucleus_info in extracted_nuclei:
        contour = nucleus_info['contour']
        nucleus_id = nucleus_info['nucleus_id']
        
        # Get classification
        if nucleus_id in nucleus_classifications:
            classification = nucleus_classifications[nucleus_id]
            predicted_class = classification['predicted_class']
            confidence = classification['confidence']
            class_name = class_names[predicted_class + 1]  # Convert 0-4 to 1-5
            color = tuple(map(int, class_colors[predicted_class + 1]))
            
            # Ensure contour is in correct format for OpenCV
            if isinstance(contour, (list, tuple)) and len(contour) > 0:
                # Convert to numpy array if needed
                contour_np = np.array(contour, dtype=np.int32)
                if contour_np.ndim == 2 and contour_np.shape[1] == 2:
                    # Reshape to (n_points, 1, 2) format expected by OpenCV
                    contour_np = contour_np.reshape(-1, 1, 2)
                elif contour_np.ndim == 3:
                    # Already in correct format
                    pass
                else:
                    # Skip invalid contours
                    continue
                    
                # Draw contour and label
                cv2.drawContours(nuclei_overlay, [contour_np], -1, color, 2)
                
                # Add label with confidence
                M = cv2.moments(contour_np)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(nuclei_overlay, f'{class_name[:4]}\n{confidence:.2f}', 
                               (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                # If contour is invalid or empty, skip this nucleus
                continue
    
    axes[1, 2].imshow(nuclei_overlay)
    axes[1, 2].set_title('Extracted Nuclei with\nClassifications', fontweight='bold')
    axes[1, 2].axis('off')
    
    # Pipeline class distribution
    pipeline_unique, pipeline_counts = np.unique(pipeline_type_mask, return_counts=True)
    pipeline_dist = np.zeros(len(class_names))
    for cls, count in zip(pipeline_unique, pipeline_counts):
        if cls < len(class_names):
            pipeline_dist[cls] = count
    
    pipeline_dist_no_bg = pipeline_dist[1:]
    
    bars_pipeline = axes[1, 3].bar(range(len(class_names_no_bg)), pipeline_dist_no_bg, 
                                 color=colors_no_bg, alpha=0.8)
    axes[1, 3].set_title('Pipeline Result\nClass Distribution', fontweight='bold')
    axes[1, 3].set_xlabel('Classes')
    axes[1, 3].set_ylabel('Pixel Count')
    axes[1, 3].set_xticks(range(len(class_names_no_bg)))
    axes[1, 3].set_xticklabels([name[:4] for name in class_names_no_bg], rotation=45)
    axes[1, 3].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for ax, bars in [(axes[0, 3], bars_gt), (axes[1, 3], bars_pipeline)]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + max(height*0.01, 1),
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Pipeline comparison saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def calculate_pipeline_metrics(gt_type_mask: np.ndarray, 
                             pipeline_type_mask: np.ndarray) -> Dict[str, float]:
    """Calculate performance metrics comparing pipeline results with ground truth."""
    
    # Flatten masks
    gt_flat = gt_type_mask.flatten()
    pred_flat = pipeline_type_mask.flatten()
    
    # Calculate overall accuracy (excluding background)
    non_bg_mask = gt_flat > 0
    if non_bg_mask.sum() > 0:
        accuracy = (gt_flat[non_bg_mask] == pred_flat[non_bg_mask]).mean()
    else:
        accuracy = 0.0
    
    # Calculate per-class metrics
    class_names = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
    class_metrics = {}
    
    for i, class_name in enumerate(class_names):
        if i == 0:  # Skip background
            continue
            
        # True positives, false positives, false negatives
        tp = ((gt_flat == i) & (pred_flat == i)).sum()
        fp = ((gt_flat != i) & (pred_flat == i)).sum()
        fn = ((gt_flat == i) & (pred_flat != i)).sum()
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': (gt_flat == i).sum()
        }
    
    # Calculate weighted F1
    total_support = sum([metrics['support'] for metrics in class_metrics.values()])
    if total_support > 0:
        weighted_f1 = sum([metrics['f1'] * metrics['support'] 
                          for metrics in class_metrics.values()]) / total_support
    else:
        weighted_f1 = 0.0
    
    return {
        'overall_accuracy': accuracy,
        'weighted_f1': weighted_f1,
        'class_metrics': class_metrics
    }


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description='Two-Stage Pipeline Demonstration')
    
    # Data arguments
    parser.add_argument('--dataset_path', type=str, default='Dataset',
                       help='Path to PanNuke dataset directory')
    parser.add_argument('--fold', type=int, default=1, choices=[1, 2, 3],
                       help='Dataset fold to use')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to demonstrate')
    
    # Model arguments (with auto-detection)
    parser.add_argument('--segmentation_model', type=str, 
                       default='lightning_logs/segmentation/version_1/checkpoints/advanced-epoch=112-val_dice=0.656.ckpt',
                       help='Path to segmentation model (auto-detected if not provided)')
    parser.add_argument('--classifier_model', type=str, 
                       default='lightning_logs/classifier/classifier_efficientnet_b3_20250727_002713/version_0/checkpoints/classifier-epoch=19-val_f1=0.806.ckpt',
                       help='Path to classifier model (auto-detected if not provided)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='demo_results',
                       help='Directory to save demonstration results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--no_display', action='store_true',
                       help='Do not display plots (save only)')
    
    args = parser.parse_args()
    
    print("🧬 TWO-STAGE NUCLEI PIPELINE DEMONSTRATION")
    print("="*60)
    print(f"📊 Dataset: {args.dataset_path} (fold {args.fold})")
    print(f"🎯 Sample: {args.sample_idx}")
    print(f"⚡ Device: {args.device}")
    print("="*60)
    
    # Auto-detect models if not provided
    if args.segmentation_model is None or args.classifier_model is None:
        print("🔍 Auto-detecting best available models...")
        best_models = find_best_models()
        
        if args.segmentation_model is None:
            args.segmentation_model = best_models.get('segmentation')
        if args.classifier_model is None:
            args.classifier_model = best_models.get('classifier')
        
        if args.segmentation_model:
            print(f"   ✅ Segmentation: {args.segmentation_model}")
        if args.classifier_model:
            print(f"   ✅ Classifier: {args.classifier_model}")
    
    # Check models exist
    if not args.segmentation_model or not Path(args.segmentation_model).exists():
        raise FileNotFoundError(f"Segmentation model not found: {args.segmentation_model}")
    if not args.classifier_model or not Path(args.classifier_model).exists():
        raise FileNotFoundError(f"Classifier model not found: {args.classifier_model}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load sample from dataset
    image, gt_instance_mask, gt_type_mask = load_sample_from_dataset(
        args.dataset_path, args.fold, args.sample_idx
    )
    
    # Initialize pipeline
    print("\n🚀 Initializing Two-Stage Pipeline...")
    pipeline = TwoStageNucleiPipeline(
        segmentation_model_path=args.segmentation_model,
        classifier_model_path=args.classifier_model,
        device=args.device
    )
    
    # Save original image
    original_path = output_dir / f'sample_{args.sample_idx}_original.png'
    # Ensure image is in proper format for saving
    if image.dtype == np.uint8:
        # Convert uint8 (0-255) to float (0-1) for plt.imsave
        image_to_save = image.astype(np.float32) / 255.0
    else:
        # Already float, ensure it's in 0-1 range
        image_to_save = np.clip(image.astype(np.float32), 0, 1)
    
    plt.imsave(original_path, image_to_save)
    
    # Run pipeline analysis
    print("\n🔬 Running Two-Stage Analysis...")
    pipeline_results = pipeline.analyze_image_array(
        image=image,
        save_results=True,
        output_dir=str(output_dir),
        visualize=False  # We'll create our own visualization
    )
    
    # Create classification mask from pipeline results
    seg_prediction = pipeline_results['segmentation']['prediction_mask'].cpu().numpy()
    extracted_nuclei = pipeline_results['extracted_nuclei']
    nucleus_classifications = pipeline_results['classifications']
    
    pipeline_type_mask = np.zeros_like(seg_prediction, dtype=np.uint8)
    
    # Create visualization mask from classification results using centroid and area information
    for nucleus_info in extracted_nuclei:
        nucleus_id = nucleus_info['nucleus_id']
        
        if nucleus_id in nucleus_classifications:
            classification = nucleus_classifications[nucleus_id]
            predicted_class = classification['predicted_class'] + 1  # Convert 0-4 to 1-5
            
            # Create circular mask from centroid and area since the provided masks are empty
            centroid = nucleus_info.get('centroid', (128, 128))
            area = nucleus_info.get('area', 100)
            
            # Create circular approximation at centroid
            cy, cx = int(centroid[0]), int(centroid[1])
            radius = max(3, int(np.sqrt(area / np.pi)))  # Minimum radius of 3
            
            # Ensure centroid is within image bounds
            cy = max(radius, min(seg_prediction.shape[0] - radius, cy))
            cx = max(radius, min(seg_prediction.shape[1] - radius, cx))
            
            y, x = np.ogrid[:seg_prediction.shape[0], :seg_prediction.shape[1]]
            circle_mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            
            pipeline_type_mask[circle_mask] = predicted_class
    
    print(f"✅ Applied classification masks for {len(nucleus_classifications)} nuclei")
    
    # Calculate performance metrics
    print("\n📊 Calculating Performance Metrics...")
    metrics = calculate_pipeline_metrics(gt_type_mask, pipeline_type_mask)
    
    # Print results
    print("\n" + "="*50)
    print("📈 PIPELINE PERFORMANCE RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Weighted F1 Score: {metrics['weighted_f1']:.4f}")
    
    print("\nPer-Class Results:")
    for class_name, class_metrics in metrics['class_metrics'].items():
        print(f"  {class_name:12}: F1={class_metrics['f1']:.3f}, "
              f"Precision={class_metrics['precision']:.3f}, "
              f"Recall={class_metrics['recall']:.3f}, "
              f"Support={class_metrics['support']}")
    
    print(f"\nDetected Nuclei: {len(extracted_nuclei)}")
    print(f"Classification Results:")
    class_names = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
    class_counts = {name: 0 for name in class_names}
    
    for nucleus_id, classification in nucleus_classifications.items():
        class_name = class_names[classification['predicted_class']]
        class_counts[class_name] += 1
    
    for class_name, count in class_counts.items():
        print(f"  {class_name:12}: {count} nuclei")
    
    # Create comprehensive visualization
    print("\n🎨 Creating Visualization...")
    viz_path = output_dir / f'pipeline_demo_sample_{args.sample_idx}.png'
    
    fig = visualize_pipeline_results(
        image=image,
        gt_instance_mask=gt_instance_mask,
        gt_type_mask=gt_type_mask,
        pipeline_results=pipeline_results,
        save_path=str(viz_path),
        show_plot=not args.no_display
    )
    
    # Save detailed results
    results_path = output_dir / f'demo_results_sample_{args.sample_idx}.json'
    detailed_results = {
        'sample_info': {
            'dataset_path': args.dataset_path,
            'fold': args.fold,
            'sample_idx': args.sample_idx,
            'image_shape': image.shape,
        },
        'models': {
            'segmentation_model': args.segmentation_model,
            'classifier_model': args.classifier_model,
        },
        'pipeline_results': {
            'nuclei_detected': len(extracted_nuclei),
            'classification_counts': class_counts,
        },
        'performance_metrics': metrics,
        'individual_classifications': [
            {
                'nucleus_id': nucleus_id,
                'predicted_class': classification['predicted_class'],
                'predicted_class_name': class_names[classification['predicted_class']],
                'confidence': classification['confidence'],
                'area': nucleus_info['area'],
                'centroid': nucleus_info['centroid']
            }
            for nucleus_info in extracted_nuclei
            for nucleus_id, classification in nucleus_classifications.items()
            if nucleus_info['nucleus_id'] == nucleus_id
        ]
    }
    
    with open(results_path, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"💾 Detailed results saved to: {results_path}")
    print("="*50)
    print("✅ Pipeline demonstration complete!")
    
    return detailed_results


if __name__ == "__main__":
    main() 