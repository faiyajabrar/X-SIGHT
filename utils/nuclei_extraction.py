"""
Nuclei Instance Extraction for Two-Stage Classification Pipeline

This module provides functions to extract individual nuclei instances from 
segmentation masks using connected components analysis. Each nucleus is cropped
with context padding and prepared for fine-grained classification.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from skimage import measure, morphology
from skimage.segmentation import clear_border
import warnings
warnings.filterwarnings('ignore')


def extract_nuclei_instances(
    image: np.ndarray,
    segmentation_mask: np.ndarray,
    min_area: int = 50,
    max_area: int = 5000,
    context_padding: int = 32,
    target_size: int = 224
) -> List[Dict]:
    """
    Extract individual nuclei instances from segmentation mask using connected components.
    
    Args:
        image: Original RGB image [H, W, 3]
        segmentation_mask: Segmentation mask with class labels [H, W]
        min_area: Minimum nucleus area in pixels
        max_area: Maximum nucleus area in pixels  
        context_padding: Padding around nucleus for context
        target_size: Target size for resized patches (224x224 for ResNet50)
        
    Returns:
        List of dictionaries containing:
        - 'patch': Cropped and resized nucleus patch [target_size, target_size, 3]
        - 'mask_patch': Corresponding mask patch [target_size, target_size]
        - 'class_id': Nucleus class (1-5, excluding background=0)
        - 'bbox': Bounding box (y1, x1, y2, x2)
        - 'area': Nucleus area in pixels
        - 'centroid': Nucleus centroid (y, x)
        - 'instance_id': Unique instance identifier
    """
    nuclei_instances = []
    instance_id = 0
    
    # Process each nucleus class (1-5, excluding background=0)
    for class_id in range(1, 6):
        # Create binary mask for current class
        class_mask = (segmentation_mask == class_id).astype(np.uint8)
        
        if class_mask.sum() == 0:
            continue
            
        # Apply morphological operations to clean up the mask
        # Remove small holes and smooth boundaries
        class_mask = morphology.remove_small_holes(class_mask.astype(bool), area_threshold=64)
        class_mask = morphology.remove_small_objects(class_mask, min_size=min_area)
        class_mask = class_mask.astype(np.uint8)
        
        # Find connected components
        labeled_mask = measure.label(class_mask, connectivity=2)
        regions = measure.regionprops(labeled_mask)
        
        for region in regions:
            # Filter by area
            if region.area < min_area or region.area > max_area:
                continue
                
            # Get bounding box with padding
            y1, x1, y2, x2 = region.bbox
            
            # Add context padding
            h, w = image.shape[:2]
            y1_pad = max(0, y1 - context_padding)
            x1_pad = max(0, x1 - context_padding)
            y2_pad = min(h, y2 + context_padding)
            x2_pad = min(w, x2 + context_padding)
            
            # Extract image and mask patches
            image_patch = image[y1_pad:y2_pad, x1_pad:x2_pad]
            mask_patch = segmentation_mask[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Skip if patch is too small
            if image_patch.shape[0] < 10 or image_patch.shape[1] < 10:
                continue
                
            # Resize to target size while maintaining aspect ratio
            patch_h, patch_w = image_patch.shape[:2]
            
            # Calculate scaling to fit target size
            scale = min(target_size / patch_h, target_size / patch_w)
            new_h = int(patch_h * scale)
            new_w = int(patch_w * scale)
            
            # Resize image and mask
            image_resized = cv2.resize(image_patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask_patch, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Pad to target size with zeros
            final_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            final_mask = np.zeros((target_size, target_size), dtype=np.uint8)
            
            # Center the resized patch
            start_y = (target_size - new_h) // 2
            start_x = (target_size - new_w) // 2
            
            final_image[start_y:start_y+new_h, start_x:start_x+new_w] = image_resized
            final_mask[start_y:start_y+new_h, start_x:start_x+new_w] = mask_resized
            
            # Store nucleus instance
            nucleus_info = {
                'patch': final_image,
                'mask_patch': final_mask,
                'class_id': class_id,
                'bbox': (y1_pad, x1_pad, y2_pad, x2_pad),
                'area': region.area,
                'centroid': region.centroid,
                'instance_id': instance_id,
                'original_size': (patch_h, patch_w),
                'scale_factor': scale
            }
            
            nuclei_instances.append(nucleus_info)
            instance_id += 1
    
    return nuclei_instances


def extract_nuclei_from_prediction(
    image: torch.Tensor,
    prediction_logits: torch.Tensor,
    min_area: int = 50,
    max_area: int = 5000,
    context_padding: int = 32,
    target_size: int = 224
) -> List[Dict]:
    """
    Extract nuclei instances from model prediction logits.
    
    Args:
        image: Input image tensor [C, H, W] (normalized)
        prediction_logits: Model prediction logits [num_classes, H, W]
        min_area: Minimum nucleus area in pixels
        max_area: Maximum nucleus area in pixels
        context_padding: Padding around nucleus for context
        target_size: Target size for resized patches
        
    Returns:
        List of nucleus instances
    """
    # Convert prediction to class map
    prediction_mask = torch.argmax(prediction_logits, dim=0).cpu().numpy()
    
    # Convert normalized image back to displayable format
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))  # C,H,W -> H,W,C
        
        # Improved denormalization for Z-score normalized data
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            # Handle Z-score normalization (can have values outside 0-1)
            # Clip extreme Z-scores and normalize to 0-255 range
            img_clipped = np.clip(img_np, -3, 3)  # Clip to reasonable Z-score range
            img_min, img_max = img_clipped.min(), img_clipped.max()
            if img_max > img_min:
                img_np = ((img_clipped - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_np = np.zeros_like(img_clipped, dtype=np.uint8)
        elif img_np.max() <= 1.0:
            # Standard 0-1 normalized image
            img_np = (img_np * 255).astype(np.uint8)
        else:
            # Already in 0-255 range
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    else:
        img_np = image
    
    return extract_nuclei_instances(
        image=img_np,
        segmentation_mask=prediction_mask,
        min_area=min_area,
        max_area=max_area,
        context_padding=context_padding,
        target_size=target_size
    )


def visualize_extracted_nuclei(
    nuclei_instances: List[Dict],
    max_display: int = 20,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize extracted nuclei instances.
    
    Args:
        nuclei_instances: List of extracted nuclei
        max_display: Maximum number of nuclei to display
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    if len(nuclei_instances) == 0:
        print("No nuclei instances to visualize")
        return
    
    # Limit display count
    display_count = min(len(nuclei_instances), max_display)
    instances_to_show = nuclei_instances[:display_count]
    
    # Calculate grid size
    cols = min(5, display_count)
    rows = (display_count + cols - 1) // cols
    
    # Class names and colors
    class_names = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1) if cols > 1 else [axes]
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, nucleus in enumerate(instances_to_show):
        row = i // cols
        col = i % cols
        
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Display nucleus patch
        ax.imshow(nucleus['patch'])
        
        class_name = class_names[nucleus['class_id']]
        title = f"ID:{nucleus['instance_id']} {class_name}\nArea:{nucleus['area']}"
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(display_count, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.suptitle(f'Extracted Nuclei Instances ({display_count}/{len(nuclei_instances)})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved nuclei visualization to: {save_path}")
    
    plt.show()


def nucleus_statistics(nuclei_instances: List[Dict]) -> Dict:
    """
    Compute statistics about extracted nuclei instances.
    
    Args:
        nuclei_instances: List of extracted nuclei
        
    Returns:
        Dictionary with statistics
    """
    if len(nuclei_instances) == 0:
        return {"total_count": 0}
    
    # Class distribution
    class_counts = {}
    class_names = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
    
    for nucleus in nuclei_instances:
        class_id = nucleus['class_id']
        class_name = class_names[class_id]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Area statistics
    areas = [nucleus['area'] for nucleus in nuclei_instances]
    
    stats = {
        "total_count": len(nuclei_instances),
        "class_distribution": class_counts,
        "area_stats": {
            "mean": np.mean(areas),
            "std": np.std(areas),
            "min": np.min(areas),
            "max": np.max(areas),
            "median": np.median(areas)
        }
    }
    
    return stats


def save_nuclei_dataset(
    nuclei_instances: List[Dict],
    save_dir: str,
    dataset_name: str = "extracted_nuclei"
) -> str:
    """
    Save extracted nuclei instances to disk for training using memory-efficient storage.
    Images are saved as separate files, metadata stored in pickle.
    
    Args:
        nuclei_instances: List of extracted nuclei
        save_dir: Directory to save dataset
        dataset_name: Name of the dataset
        
    Returns:
        Path to saved dataset metadata file
    """
    import os
    import pickle
    from pathlib import Path
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    # Create images directory
    images_dir = save_path / f"{dataset_name}_images"
    images_dir.mkdir(exist_ok=True)
    
    print(f"💾 Saving {len(nuclei_instances)} nuclei instances with memory-efficient storage...")
    
    # Prepare metadata (without large image arrays)
    metadata_instances = []
    
    for i, nucleus in enumerate(nuclei_instances):
        if i % 10000 == 0:
            print(f"   Processed {i}/{len(nuclei_instances)} nuclei...")
        
        # Save image as separate file
        image_filename = f"nucleus_{i:06d}.npy"
        image_path = images_dir / image_filename
        
        # Save image data (stored in 'patch' key)
        np.save(image_path, nucleus['patch'])
        
        # Create metadata entry (without image data)
        metadata = {k: v for k, v in nucleus.items() if k not in ['patch', 'mask_patch']}
        metadata['image_path'] = str(image_path)
        metadata['image_shape'] = nucleus['patch'].shape
        
        metadata_instances.append(metadata)
    
    # Save metadata pickle file (much smaller)
    dataset_file = save_path / f"{dataset_name}_metadata.pkl"
    
    with open(dataset_file, 'wb') as f:
        pickle.dump(metadata_instances, f)
    
    print(f"✅ Saved {len(nuclei_instances)} nuclei instances to: {dataset_file}")
    print(f"   Images directory: {images_dir}")
    print(f"   Metadata file size: ~{os.path.getsize(dataset_file) / 1024 / 1024:.1f} MB")
    
    # Save summary statistics
    stats = nucleus_statistics(nuclei_instances)
    stats_file = save_path / f"{dataset_name}_stats.json"
    
    import json
    with open(stats_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        stats_json = json.dumps(stats, default=convert_numpy, indent=2)
        f.write(stats_json)
    
    print(f"Saved dataset statistics to: {stats_file}")
    
    return str(dataset_file)


def load_nuclei_metadata(dataset_path: str) -> List[Dict]:
    """
    Load nuclei metadata from disk (for memory-efficient loading).
    
    Args:
        dataset_path: Path to the metadata pickle file
        
    Returns:
        List of nuclei metadata (without image data)
    """
    import pickle
    
    with open(dataset_path, 'rb') as f:
        metadata_instances = pickle.load(f)
    
    print(f"✅ Loaded {len(metadata_instances)} nuclei metadata entries")
    return metadata_instances


def load_nucleus_image(image_path: str) -> np.ndarray:
    """
    Load a single nucleus image from disk.
    
    Args:
        image_path: Path to the nucleus image file
        
    Returns:
        Nucleus image array
    """
    return np.load(image_path)


def prepare_classifier_dataset(
    dataset_root: str = 'Dataset',
    output_dir: str = 'nuclei_dataset',
    min_area: int = 50,
    max_area: int = 5000,
    context_padding: int = 32,
    max_samples: int = None,
    visualize_samples: int = 20
) -> str:
    """
    Complete pipeline to prepare nuclei classification dataset from PanNuke ground truth.
    
    Args:
        dataset_root: Root directory of PanNuke dataset
        output_dir: Directory to save extracted nuclei dataset
        min_area: Minimum nucleus area in pixels
        max_area: Maximum nucleus area in pixels
        context_padding: Context padding around nuclei
        max_samples: Maximum number of samples to process (None for all)
        visualize_samples: Number of extracted nuclei to visualize
        
    Returns:
        Path to saved nuclei dataset file
    """
    from pathlib import Path
    from tqdm import tqdm
    import json
    
    print("🧬 PREPARING NUCLEI CLASSIFICATION DATASET")
    print("="*60)
    print(f"Dataset root: {dataset_root}")
    print(f"Output directory: {output_dir}")
    print(f"Area range: {min_area} - {max_area} pixels")
    print(f"Context padding: {context_padding} pixels")
    print(f"Using: GROUND TRUTH annotations")
    print("="*60)
    
    # Import here to avoid circular imports
    try:
        from utils.pannuke_dataset import PanNukeDataset
    except ImportError:
        # Handle direct execution from utils directory
        from pannuke_dataset import PanNukeDataset
    
    # Load dataset
    print("📂 Loading PanNuke dataset...")
    dataset = PanNukeDataset(
        root=dataset_root,
        augmentations=None,  # No augmentations for extraction
        validate_dataset=False,
        base_size=256
    )
    
    total_samples = len(dataset)
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)
    
    print(f"📊 Dataset loaded: {len(dataset)} total samples")
    print(f"🔄 Processing {total_samples} samples...")
    
    # Extract nuclei instances
    all_nuclei_instances = []
    sample_count = 0
    nuclei_count = 0
    
    for idx in tqdm(range(total_samples), desc="Extracting nuclei from ground truth"):
        try:
            # Get sample from dataset
            sample = dataset[idx]
            image_tensor = sample['image']  # [C, H, W]
            ground_truth_mask = sample['mask']  # [H, W]
            
            # Convert image tensor back to numpy for extraction
            image_np = image_tensor.cpu().numpy()
            if image_np.ndim == 3 and image_np.shape[0] == 3:
                image_np = np.transpose(image_np, (1, 2, 0))  # C,H,W -> H,W,C
            
            # Denormalize from Z-score normalization (approximate)
            img_min, img_max = image_np.min(), image_np.max()
            if img_max > img_min:
                image_np = (image_np - img_min) / (img_max - img_min)
            image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
            
            # Use ground truth mask for extraction
            gt_mask_np = ground_truth_mask.cpu().numpy()
            
            # Extract nuclei instances from ground truth
            nuclei_instances = extract_nuclei_instances(
                image=image_np,
                segmentation_mask=gt_mask_np,
                min_area=min_area,
                max_area=max_area,
                context_padding=context_padding,
                target_size=224
            )
            
            # Add sample index to each nucleus
            for nucleus in nuclei_instances:
                nucleus['sample_idx'] = idx
                nucleus['global_instance_id'] = nuclei_count + nucleus['instance_id']
            
            all_nuclei_instances.extend(nuclei_instances)
            nuclei_count += len(nuclei_instances)
            sample_count += 1
            
            # Progress update
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{total_samples} samples, "
                      f"extracted {nuclei_count} nuclei instances")
            
        except Exception as e:
            print(f"⚠️  Error processing sample {idx}: {e}")
            continue
    
    print(f"\n✅ Extraction complete!")
    print(f"📊 Processed {sample_count} samples")
    print(f"🧬 Extracted {len(all_nuclei_instances)} nuclei instances")
    
    if len(all_nuclei_instances) == 0:
        print("❌ No nuclei instances extracted!")
        return None
    
    # Generate statistics
    stats = nucleus_statistics(all_nuclei_instances)
    print(f"\n📈 Nuclei Statistics:")
    print(f"  Total count: {stats['total_count']}")
    print(f"  Class distribution: {stats['class_distribution']}")
    print(f"  Area stats: mean={stats['area_stats']['mean']:.1f}, "
          f"std={stats['area_stats']['std']:.1f}")
    
    # Save nuclei dataset
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    dataset_file = save_nuclei_dataset(
        nuclei_instances=all_nuclei_instances,
        save_dir=str(output_path),
        dataset_name="extracted_nuclei"
    )
    
    # Save extraction metadata
    metadata = {
        'dataset_root': dataset_root,
        'total_samples_processed': sample_count,
        'total_nuclei_extracted': len(all_nuclei_instances),
        'extraction_method': 'ground_truth_annotations',
        'extraction_parameters': {
            'min_area': min_area,
            'max_area': max_area,
            'context_padding': context_padding,
            'target_size': 224
        },
        'statistics': stats
    }
    
    metadata_file = output_path / 'extraction_metadata.json'
    with open(metadata_file, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(metadata, f, default=convert_numpy, indent=2)
    
    print(f"💾 Extraction metadata saved to: {metadata_file}")
    
    # Visualize some extracted nuclei
    if visualize_samples > 0 and len(all_nuclei_instances) > 0:
        print(f"\n🎨 Visualizing {min(visualize_samples, len(all_nuclei_instances))} extracted nuclei...")
        
        visualization_path = output_path / 'nuclei_samples.png'
        visualize_extracted_nuclei(
            nuclei_instances=all_nuclei_instances,
            max_display=visualize_samples,
            save_path=str(visualization_path)
        )
    
    print(f"\n🎉 Nuclei dataset preparation complete!")
    print(f"📁 Dataset saved to: {dataset_file}")
    print(f"📊 Ready for classifier training with {len(all_nuclei_instances)} nucleus instances")
    
    return dataset_file


# Command-line interface for standalone usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare nuclei classification dataset from PanNuke')
    
    # Dataset arguments
    parser.add_argument('--dataset_root', type=str, default='Dataset',
                       help='Root directory of PanNuke dataset')
    parser.add_argument('--output_dir', type=str, default='nuclei_dataset',
                       help='Directory to save extracted nuclei dataset')
    
    # Processing arguments
    parser.add_argument('--min_area', type=int, default=50,
                       help='Minimum nucleus area in pixels')
    parser.add_argument('--max_area', type=int, default=5000,
                       help='Maximum nucleus area in pixels')
    parser.add_argument('--context_padding', type=int, default=32,
                       help='Context padding around nuclei')
    
    # Optional arguments
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (None for all)')
    parser.add_argument('--visualize_samples', type=int, default=20,
                       help='Number of extracted nuclei to visualize')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    from pathlib import Path
    if not Path(args.dataset_root).exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    
    # Run dataset preparation
    prepare_classifier_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        min_area=args.min_area,
        max_area=args.max_area,
        context_padding=args.context_padding,
        max_samples=args.max_samples,
        visualize_samples=args.visualize_samples
    ) 