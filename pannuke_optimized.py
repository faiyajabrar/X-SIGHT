#!/usr/bin/env python3
"""
Memory-Optimized PanNuke Dataset DataLoader for PyTorch
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import random
import os
from typing import Tuple, Optional, Dict


class PanNukeDatasetOptimized(Dataset):
    """Memory-optimized PanNuke Dataset using memory mapping"""
    
    def __init__(
        self,
        dataset_path: str,
        split: str = 'train',
        augment: bool = True,
        stain_normalize: bool = False,  # Disabled by default for memory optimization
        image_size: Tuple[int, int] = (256, 256),
        normalize: bool = True,
        device: str = 'auto'
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Setup memory-mapped access
        self._setup_data_access()
        
        # Nuclei type mapping
        self.nuclei_types = {
            0: 'Background', 1: 'Neoplastic', 2: 'Inflammatory', 
            3: 'Connective', 4: 'Dead', 5: 'Epithelial'
        }
        
        # Tissue type mapping (create from unique values in dataset)
        unique_types = np.unique(self.types)
        self.tissue_type_mapping = {tissue: idx for idx, tissue in enumerate(unique_types)}
        print(f"Tissue types found: {list(self.tissue_type_mapping.keys())}")
        
        print(f"Loaded {len(self.indices)} samples for {split} split")
    
    def _setup_data_access(self):
        """Setup memory-mapped access to data files"""
        
        # File paths
        images_path = os.path.join(self.dataset_path, 'Images', 'images.npy')
        masks_path = os.path.join(self.dataset_path, 'Masks', 'masks.npy')
        types_path = os.path.join(self.dataset_path, 'Images', 'types.npy')
        
        # Load types (small file, can load entirely)
        self.types = np.load(types_path)
        
        # Use memory mapping for large files
        print("Setting up memory-mapped arrays...")
        self.images_mmap = np.load(images_path, mmap_mode='r')
        self.masks_mmap = np.load(masks_path, mmap_mode='r')
        
        print(f"Data shapes:")
        print(f"  Images: {self.images_mmap.shape}")
        print(f"  Masks: {self.masks_mmap.shape}")
        print(f"  Types: {self.types.shape}")
        
        # Split data indices
        total_samples = len(self.types)
        
        if self.split == 'train':
            start_idx, end_idx = 0, int(0.7 * total_samples)
        elif self.split == 'val':
            start_idx, end_idx = int(0.7 * total_samples), int(0.9 * total_samples)
        elif self.split == 'test':
            start_idx, end_idx = int(0.9 * total_samples), total_samples
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        self.indices = list(range(start_idx, end_idx))
        self.types_split = self.types[start_idx:end_idx]
    
    def _apply_augmentations(self, image, mask):
        """Apply simple augmentations"""
        # Random horizontal flip
        if random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
        
        # Random 90-degree rotation
        if random.random() > 0.5:
            k = random.randint(1, 3)
            image = np.rot90(image, k, axes=(0, 1))
            mask = np.rot90(mask, k, axes=(0, 1))
        
        return image, mask
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset"""
        
        # Get global index
        global_idx = self.indices[idx]
        
        # Load data (memory-mapped, efficient)
        image = self.images_mmap[global_idx].copy()  # Copy to avoid memory mapping issues
        mask = self.masks_mmap[global_idx].copy()
        tissue_type = self.types_split[idx]
        
        # Apply augmentations if training
        if self.augment and self.split == 'train':
            image, mask = self._apply_augmentations(image, mask)
        
        # Resize if necessary
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            mask_resized = np.zeros((self.image_size[0], self.image_size[1], mask.shape[2]))
            for c in range(mask.shape[2]):
                mask_resized[:, :, c] = cv2.resize(
                    mask[:, :, c], 
                    (self.image_size[1], self.image_size[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            mask = mask_resized
        
        # Convert to tensors (ensure contiguous arrays)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        mask_tensor = torch.from_numpy(mask.transpose(2, 0, 1).copy()).float()
        
        # Normalize image
        if self.normalize:
            image_tensor = image_tensor / 255.0
        
        # Create binary mask (exclude background)
        binary_mask = (mask_tensor[1:].sum(dim=0) > 0).float()
        
        # Create instance mask
        instance_mask = torch.argmax(mask_tensor, dim=0).float()
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'binary_mask': binary_mask,
            'instance_mask': instance_mask,
            'tissue_type': torch.tensor(self.tissue_type_mapping[str(tissue_type)], dtype=torch.long),
            'idx': torch.tensor(global_idx, dtype=torch.long)
        }


def create_optimized_dataloaders(
    dataset_path: str,
    batch_size: int = 4,
    num_workers: int = 1,
    image_size: Tuple[int, int] = (256, 256),
    augment: bool = True,
    pin_memory: bool = True
):
    """Create optimized dataloaders"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Creating optimized dataloaders for device: {device}")
    
    # Create datasets
    train_dataset = PanNukeDatasetOptimized(
        dataset_path=dataset_path,
        split='train',
        augment=augment,
        image_size=image_size,
        device=device
    )
    
    val_dataset = PanNukeDatasetOptimized(
        dataset_path=dataset_path,
        split='val',
        augment=False,
        image_size=image_size,
        device=device
    )
    
    test_dataset = PanNukeDatasetOptimized(
        dataset_path=dataset_path,
        split='test',
        augment=False,
        image_size=image_size,
        device=device
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


# Test the optimized data loader
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing optimized data loader on device: {device}")
    
    dataset_path = "./Dataset"
    
    try:
        train_loader, val_loader, test_loader = create_optimized_dataloaders(
            dataset_path=dataset_path,
            batch_size=2,  # Very small batch for testing
            num_workers=0,  # No multiprocessing for testing
            image_size=(256, 256),
            augment=True
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test loading batches
        print("\nTesting batch loading...")
        for batch_idx, batch in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Image shape: {batch['image'].shape}")
            print(f"  Mask shape: {batch['mask'].shape}")
            print(f"  Tissue types: {batch['tissue_type']}")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                batch['image'] = batch['image'].to(device)
                batch['mask'] = batch['mask'].to(device)
                print(f"  Successfully moved to GPU: {batch['image'].device}")
            
            if batch_idx >= 1:  # Test just 2 batches
                break
        
        print("Optimized data loader test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 