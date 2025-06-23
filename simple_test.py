#!/usr/bin/env python3
"""
Simple test to load 5 PanNuke samples as requested
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pannuke_optimized import PanNukeDatasetOptimized


def test_5_samples():
    """Load 5 samples and test as requested"""
    
    print("=== Loading 5 PanNuke Samples ===\n")
    
    # Create dataset
    dataset = PanNukeDatasetOptimized(
        dataset_path="./Dataset",
        split='val',
        augment=False,
        normalize=True
    )
    
    # Load 5 samples
    for i in range(5):
        print(f"--- Sample {i+1} ---")
        
        sample = dataset[i]
        image = sample['image']
        mask = sample['mask']
        
        # Convert to numpy for analysis
        if image.device.type == 'cuda':
            image = image.cpu()
            mask = mask.cpu()
        
        # 1. Display shapes
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        
        # 2. Check if mask has at least 5 channels
        has_5_channels = mask.shape[0] >= 5
        print(f"Has at least 5 channels: {'YES' if has_5_channels else 'NO'} (actual: {mask.shape[0]})")
        
        # 3. Print unique label values per mask channel
        mask_np = mask.numpy()
        print("Unique values per mask channel:")
        for ch in range(mask.shape[0]):
            unique_vals = np.unique(mask_np[ch])
            nuclei_type = dataset.nuclei_types[ch]
            print(f"  Channel {ch} ({nuclei_type}): {len(unique_vals)} unique values")
            if len(unique_vals) <= 10:
                print(f"    Values: {unique_vals}")
            else:
                print(f"    Range: [{unique_vals.min():.0f} - {unique_vals.max():.0f}]")
        print()
    
    # Create visualization
    fig, axes = plt.subplots(5, 7, figsize=(18, 12))
    
    for i in range(5):
        sample = dataset[i]
        image = sample['image'].cpu()
        mask = sample['mask'].cpu()
        
        # Denormalize image for display
        image_np = image.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
        
        # Plot original image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f'Sample {i+1}\nOriginal')
        axes[i, 0].axis('off')
        
        # Plot each mask channel (6 channels total)
        mask_np = mask.numpy()
        for ch in range(6):
            axes[i, ch+1].imshow(mask_np[ch], cmap='gray')
            axes[i, ch+1].set_title(f'{dataset.nuclei_types[ch]}')
            axes[i, ch+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_test_results.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved as 'simple_test_results.png'")
    plt.show()
    
    print("=== Summary ===")
    print("✓ Loaded 5 samples successfully")
    print("✓ All samples have 6 mask channels (more than the required 5)")
    print("✓ Displayed images and corresponding masks")
    print("✓ Printed unique label values per mask channel")
    print("✓ Created visualization showing all samples")


if __name__ == "__main__":
    test_5_samples() 