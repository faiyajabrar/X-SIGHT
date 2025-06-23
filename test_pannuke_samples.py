#!/usr/bin/env python3
"""
Test script to load and visualize PanNuke dataset samples
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pannuke_optimized import PanNukeDatasetOptimized
import cv2


def test_pannuke_samples():
    """Test loading and visualization of 5 PanNuke samples"""
    
    print("=== PanNuke Dataset Sample Test ===\n")
    
    # Create dataset (using validation split for consistent results)
    dataset = PanNukeDatasetOptimized(
        dataset_path="./Dataset",
        split='val',  # Use validation split for testing
        augment=False,  # No augmentation for testing
        stain_normalize=False,  # No stain normalization for testing
        image_size=(256, 256),
        normalize=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"Dataset loaded successfully!")
    print(f"Total samples in validation set: {len(dataset)}")
    print(f"Nuclei types: {list(dataset.nuclei_types.values())}")
    print(f"Tissue types: {list(dataset.tissue_type_mapping.keys())}\n")
    
    # Test loading 5 samples
    num_samples = min(5, len(dataset))
    
    # Create matplotlib figure
    fig, axes = plt.subplots(num_samples, 8, figsize=(20, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    print("Loading and analyzing samples...\n")
    
    for i in range(num_samples):
        print(f"--- Sample {i+1} ---")
        
        # Load sample
        sample = dataset[i]
        
        # Extract data
        image = sample['image']  # Shape: (3, H, W)
        mask = sample['mask']    # Shape: (6, H, W)
        binary_mask = sample['binary_mask']  # Shape: (H, W)
        instance_mask = sample['instance_mask']  # Shape: (H, W)
        tissue_type = sample['tissue_type'].item()
        
        # Convert tensors to numpy for visualization
        if image.device.type == 'cuda':
            image = image.cpu()
            mask = mask.cpu()
            binary_mask = binary_mask.cpu()
            instance_mask = instance_mask.cpu()
        
        image_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.numpy()
        binary_mask_np = binary_mask.numpy()
        instance_mask_np = instance_mask.numpy()
        
        # Denormalize image for display
        if dataset.normalize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image_np * std + mean
            image_np = np.clip(image_np, 0, 1)
        
        # Print sample information
        print(f"  Image shape: {image.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask channels: {mask.shape[0]} ({'✓' if mask.shape[0] >= 5 else '✗'} - Expected 5+)")
        print(f"  Tissue type: {list(dataset.tissue_type_mapping.keys())[tissue_type]}")
        
        # Print unique values per mask channel
        print("  Unique values per mask channel:")
        for ch in range(mask.shape[0]):
            unique_vals = np.unique(mask_np[ch])
            print(f"    Channel {ch} ({dataset.nuclei_types[ch]}): {unique_vals}")
        
        # Check if mask has at least 5 channels
        has_min_channels = mask.shape[0] >= 5
        print(f"  Has at least 5 channels: {'✓' if has_min_channels else '✗'}")
        print()
        
        # Plot original image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f'Sample {i+1}\nOriginal Image')
        axes[i, 0].axis('off')
        
        # Plot each mask channel
        for ch in range(min(6, mask.shape[0])):
            axes[i, ch+1].imshow(mask_np[ch], cmap='gray')
            axes[i, ch+1].set_title(f'{dataset.nuclei_types[ch]}\nChannel {ch}')
            axes[i, ch+1].axis('off')
        
        # Plot binary mask
        if mask.shape[0] < 7:  # Fill remaining slots
            axes[i, 7].imshow(binary_mask_np, cmap='gray')
            axes[i, 7].set_title('Binary Mask\n(All Nuclei)')
            axes[i, 7].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('pannuke_test_samples.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'pannuke_test_samples.png'")
    plt.show()


def test_mask_statistics():
    """Print detailed mask statistics"""
    
    print("\n=== Detailed Mask Statistics ===\n")
    
    # Create dataset
    dataset = PanNukeDatasetOptimized(
        dataset_path="./Dataset",
        split='train',
        augment=False,
        stain_normalize=False,
        normalize=True
    )
    
    # Analyze first 10 samples for statistics
    print("Analyzing first 10 samples for mask statistics...\n")
    
    all_mask_stats = []
    
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        mask = sample['mask'].cpu().numpy()  # Shape: (6, H, W)
        
        print(f"Sample {i+1}:")
        print(f"  Mask shape: {mask.shape}")
        
        # Calculate statistics for each channel
        channel_stats = {}
        for ch in range(mask.shape[0]):
            channel_data = mask[ch]
            unique_vals = np.unique(channel_data)
            non_zero_pixels = np.sum(channel_data > 0)
            max_val = np.max(channel_data)
            min_val = np.min(channel_data)
            
            channel_stats[ch] = {
                'unique_values': unique_vals,
                'non_zero_pixels': non_zero_pixels,
                'max_value': max_val,
                'min_value': min_val,
                'mean_value': np.mean(channel_data)
            }
            
            print(f"    Channel {ch} ({dataset.nuclei_types[ch]}):")
            print(f"      Unique values: {unique_vals}")
            print(f"      Non-zero pixels: {non_zero_pixels}")
            print(f"      Value range: [{min_val:.3f}, {max_val:.3f}]")
            print(f"      Mean value: {np.mean(channel_data):.3f}")
        
        all_mask_stats.append(channel_stats)
        print()
    
    # Overall statistics
    print("=== Overall Statistics ===")
    print(f"Total samples analyzed: {len(all_mask_stats)}")
    print(f"Mask dimensions: {mask.shape}")
    print(f"Number of nuclei types: {mask.shape[0]}")
    
    # Check consistency
    all_have_min_channels = all(len(stats) >= 5 for stats in all_mask_stats)
    print(f"All samples have at least 5 mask channels: {'✓' if all_have_min_channels else '✗'}")


def test_gpu_loading():
    """Test GPU loading performance"""
    
    print("\n=== GPU Loading Test ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return
    
    # Create dataset
    dataset = PanNukeDatasetOptimized(
        dataset_path="./Dataset",
        split='val',
        augment=False,
        normalize=True
    )
    
    # Test loading samples to GPU
    print("Loading 5 samples to GPU...")
    
    for i in range(5):
        sample = dataset[i]
        
        # Move to GPU
        image_gpu = sample['image'].to(device)
        mask_gpu = sample['mask'].to(device)
        
        print(f"Sample {i+1}:")
        print(f"  Image on GPU: {image_gpu.device}")
        print(f"  Mask on GPU: {mask_gpu.device}")
        print(f"  GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    print("GPU loading test completed successfully!")


if __name__ == "__main__":
    try:
        # Run all tests
        test_pannuke_samples()
        test_mask_statistics()
        test_gpu_loading()
        
        print("\n=== All Tests Completed Successfully! ===")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 