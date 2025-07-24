"""
Nuclei Classification Dataset for Two-Stage Pipeline

This module provides PyTorch dataset classes for training the nuclei classifier
on individual nucleus patches extracted from segmentation masks.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import cv2
import json


class NucleiClassificationDataset(Dataset):
    """
    PyTorch Dataset for nucleus classification using individual nucleus patches.
    
    This dataset loads pre-extracted nucleus instances and applies augmentations
    for training the ResNet50 classifier.
    """
    
    def __init__(
        self,
        nuclei_instances: List[Dict],
        augmentations: Optional[A.Compose] = None,
        normalize: bool = True,
        target_size: int = 224
    ):
        """
        Initialize nuclei classification dataset.
        
        Args:
            nuclei_instances: List of extracted nuclei instances from nuclei_extraction.py
            augmentations: Albumentations augmentation pipeline
            normalize: Whether to apply ImageNet normalization
            target_size: Target image size (should be 224 for ResNet50)
        """
        self.nuclei_instances = nuclei_instances
        self.augmentations = augmentations
        self.normalize = normalize
        self.target_size = target_size
        
        # ImageNet normalization parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Base transforms (always applied)
        base_transforms = []
        if target_size != 224:  # Resize if needed
            base_transforms.append(A.Resize(target_size, target_size))
        
        if normalize:
            base_transforms.append(A.Normalize(mean=self.mean, std=self.std))
            
        base_transforms.append(ToTensorV2())
        self.base_transforms = A.Compose(base_transforms)
        
        print(f"[NucleiDataset] Loaded {len(self.nuclei_instances)} nucleus instances")
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print class distribution in the dataset."""
        class_counts = {}
        # Only nucleus classes (no background for classifier)
        class_names = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
        
        for nucleus in self.nuclei_instances:
            class_id = nucleus['class_id']  # Original 1-5 from extraction
            if 1 <= class_id <= 5:  # Valid nucleus classes only
                class_name = class_names[class_id - 1]  # Map to 0-4 for indexing
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("[NucleiDataset] Class distribution:")
        for class_name, count in class_counts.items():
            percentage = count / len(self.nuclei_instances) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.nuclei_instances)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single nucleus instance.
        
        Returns:
            Dictionary containing:
            - 'image': Nucleus patch tensor [C, H, W]
            - 'label': Class label (0-4, mapped from original 1-5)
            - 'instance_id': Original instance ID
        """
        nucleus = self.nuclei_instances[idx]
        
        # Get image patch - handle both old and new formats
        if 'image_path' in nucleus:
            # New memory-efficient format: load image from disk
            image = np.load(nucleus['image_path'])
        elif 'patch' in nucleus:
            # Legacy format: image data stored in memory
            image = nucleus['patch'].copy()
        elif 'image' in nucleus:
            # Alternative format: image data in 'image' key
            image = nucleus['image'].copy()
        else:
            raise KeyError(f"No image data found in nucleus instance. Keys: {list(nucleus.keys())}")
        
        # Ensure image is in correct format (H, W, C) and uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Map class ID from 1-5 to 0-4 (excluding background)
        label = nucleus['class_id'] - 1  # 1->0, 2->1, 3->2, 4->3, 5->4
        
        # Apply augmentations if provided
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        
        # Apply base transforms (normalization + to tensor)
        transformed = self.base_transforms(image=image)
        image = transformed['image']
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'instance_id': nucleus.get('instance_id', idx)  # Fallback to idx if no instance_id
        }


def get_classification_augmentations(training: bool = True) -> A.Compose:
    """
    Get augmentation pipeline for nucleus classification.
    
    MEDICAL IMAGING OPTIMIZED: Conservative augmentations that preserve
    clinically relevant features while providing sufficient regularization.
    
    Args:
        training: Whether to return training or validation augmentations
        
    Returns:
        Albumentations composition
    """
    if training:
        # Medical imaging augmentations - conservative but effective
        return A.Compose([
            # Geometric transformations (preserving medical structure)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),  # 90-degree rotations are safe for nuclei
            A.Rotate(limit=20, p=0.3),  # Reduced from 45 to 20 degrees
            A.ShiftScaleRotate(
                shift_limit=0.05,  # Reduced shift
                scale_limit=0.1,   # Reduced scale variation
                rotate_limit=10,   # Reduced rotation
                p=0.3
            ),
            
            # Conservative color augmentations for H&E staining
            A.ColorJitter(
                brightness=0.1,    # Reduced from 0.2
                contrast=0.15,     # Reduced from 0.2
                saturation=0.1,    # Reduced from 0.2
                hue=0.02,          # Significantly reduced from 0.1
                p=0.3              # Reduced probability
            ),
            
            # Medical imaging specific: Simulate staining variations
            A.HueSaturationValue(
                hue_shift_limit=5,     # Reduced from 10
                sat_shift_limit=10,    # Reduced from 15
                val_shift_limit=10,    # Kept same
                p=0.25                 # Reduced probability
            ),
            
            # Light noise (medical images are typically clean)
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.15),  # Reduced noise and probability
            
            # Very light blur (preserve cellular detail)
            A.GaussianBlur(blur_limit=2, p=0.1),  # Reduced blur and probability
            
            # REMOVED: GridDistortion and ElasticTransform
            # These can distort nuclear morphology which is diagnostically important
        ])
    else:
        # Validation - no augmentations for consistent evaluation
        return None


def create_nuclei_dataloaders(
    nuclei_instances: List[Dict],
    train_ratio: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    """
    Create train and validation dataloaders for nucleus classification.
    
    CRITICAL: Uses stratified splitting by original image to prevent data leakage.
    Nuclei from the same original image are kept in the same split.
    
    Args:
        nuclei_instances: List of extracted nuclei instances
        train_ratio: Ratio of data to use for training
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_loader, val_loader, train_sample_indices, val_sample_indices)
    """
    # CRITICAL FIX: Stratified splitting by original image to prevent data leakage
    from sklearn.model_selection import train_test_split
    
    # Group nuclei by original sample/image
    sample_groups = {}
    for i, nucleus in enumerate(nuclei_instances):
        sample_idx = nucleus.get('sample_idx', nucleus.get('global_instance_id', i // 10))
        if sample_idx not in sample_groups:
            sample_groups[sample_idx] = []
        sample_groups[sample_idx].append(i)
    
    # Get unique sample indices and their nucleus counts  
    sample_indices = list(sample_groups.keys())
    sample_class_labels = []
    
    # For stratification, use the majority class in each sample
    for sample_idx in sample_indices:
        nucleus_indices = sample_groups[sample_idx]
        classes = [nuclei_instances[i]['class_id'] for i in nucleus_indices]
        majority_class = max(set(classes), key=classes.count)
        sample_class_labels.append(majority_class)
    
    # Stratified split by sample, not by individual nuclei
    try:
        train_samples, val_samples = train_test_split(
            sample_indices,
            test_size=1-train_ratio,
            random_state=random_seed,
            stratify=sample_class_labels
        )
    except ValueError:
        # Fallback to random split if stratification fails
        print("Warning: Stratified split failed, using random split")
        np.random.seed(random_seed)
        shuffled_samples = np.random.permutation(sample_indices)
        split_point = int(len(shuffled_samples) * train_ratio)
        train_samples = shuffled_samples[:split_point]
        val_samples = shuffled_samples[split_point:]
    
    # Collect all nuclei indices for train/val samples
    train_indices = []
    val_indices = []
    
    for sample_idx in train_samples:
        train_indices.extend(sample_groups[sample_idx])
    
    for sample_idx in val_samples:
        val_indices.extend(sample_groups[sample_idx])
    
    # Create instance subsets
    train_instances = [nuclei_instances[i] for i in train_indices]
    val_instances = [nuclei_instances[i] for i in val_indices]
    
    print(f"[DataLoader] Stratified split: {len(train_samples)} train samples -> {len(train_instances)} nuclei")
    print(f"[DataLoader] Stratified split: {len(val_samples)} val samples -> {len(val_instances)} nuclei")
    
    print(f"[DataLoader] Train instances: {len(train_instances)}")
    print(f"[DataLoader] Validation instances: {len(val_instances)}")
    
    # Create datasets
    train_dataset = NucleiClassificationDataset(
        nuclei_instances=train_instances,
        augmentations=get_classification_augmentations(training=True),
        normalize=True
    )
    
    val_dataset = NucleiClassificationDataset(
        nuclei_instances=val_instances,
        augmentations=get_classification_augmentations(training=False),
        normalize=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # For stable batch norm in training
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive between epochs
    )
    
    # Convert to lists if they're numpy arrays
    train_sample_list = train_samples.tolist() if hasattr(train_samples, 'tolist') else list(train_samples)
    val_sample_list = val_samples.tolist() if hasattr(val_samples, 'tolist') else list(val_samples)
    
    return train_loader, val_loader, train_sample_list, val_sample_list


def load_nuclei_dataset(dataset_path: str) -> List[Dict]:
    """
    Load pre-extracted nuclei instances from disk (memory-efficient format).
    
    Args:
        dataset_path: Path to saved nuclei dataset metadata (.pkl file)
        
    Returns:
        List of nuclei metadata instances (images loaded on-demand)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Handle both old format (with images) and new format (metadata only)
    with open(dataset_path, 'rb') as f:
        nuclei_instances = pickle.load(f)
    
    # Check if this is the new memory-efficient format
    if nuclei_instances and 'image_path' in nuclei_instances[0]:
        print(f"[DataLoader] Loaded {len(nuclei_instances)} nuclei metadata entries (memory-efficient format)")
        print(f"               Images will be loaded on-demand during training")
    else:
        print(f"[DataLoader] Loaded {len(nuclei_instances)} nuclei instances (legacy format)")
    
    return nuclei_instances


def save_dataset_split(
    train_instances: List[Dict],
    val_instances: List[Dict],
    save_dir: str,
    dataset_name: str = "nuclei_classification"
) -> None:
    """
    Save train/validation split for reproducible experiments.
    
    Args:
        train_instances: Training nuclei instances
        val_instances: Validation nuclei instances
        save_dir: Directory to save split
        dataset_name: Name for the saved files
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    # Save split indices
    train_ids = [nucleus['instance_id'] for nucleus in train_instances]
    val_ids = [nucleus['instance_id'] for nucleus in val_instances]
    
    split_info = {
        'train_ids': train_ids,
        'val_ids': val_ids,
        'train_count': len(train_instances),
        'val_count': len(val_instances)
    }
    
    split_file = save_path / f"{dataset_name}_split.json"
    with open(split_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"[DataLoader] Saved dataset split info to: {split_file}")


class NucleiDataModule:
    """
    Data module for nucleus classification with perfect resume capability.
    """
    
    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        num_workers: int = 4,
        random_seed: int = 42,
        train_indices: Optional[List[int]] = None,
        val_indices: Optional[List[int]] = None
    ):
        """
        Initialize nuclei data module with perfect resume capability.
        
        Args:
            dataset_path: Path to nuclei dataset file
            batch_size: Batch size for dataloaders
            train_ratio: Ratio of data for training
            num_workers: Number of dataloader workers
            random_seed: Random seed for splits
            train_indices: Pre-defined training sample indices for resume (original image IDs)
            val_indices: Pre-defined validation sample indices for resume (original image IDs)
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers
        self.random_seed = random_seed
        
        # Resume capability - nuclei indices (from saved dataset split)
        self.train_indices = train_indices
        self.val_indices = val_indices
        
        # Sample indices for proper splitting (actual sample IDs from original images)
        self.train_sample_indices = None
        self.val_sample_indices = None
        
        self.nuclei_instances = None
        self.train_loader = None
        self.val_loader = None
    
    def setup(self) -> None:
        """Load dataset and create dataloaders with resume capability."""
        # Load nuclei instances
        self.nuclei_instances = load_nuclei_dataset(self.dataset_path)
        
        # Use saved indices if available (for resume), otherwise create new split
        if self.train_indices is not None and self.val_indices is not None:
            print(f"[NucleiDataModule] Using saved split for resume")
            
            # Treat saved indices as sample indices (original image IDs)
            # Convert sample indices to nuclei indices
            sample_groups = {}
            for i, nucleus in enumerate(self.nuclei_instances):
                sample_idx = nucleus.get('sample_idx', nucleus.get('global_instance_id', i // 10))
                if sample_idx not in sample_groups:
                    sample_groups[sample_idx] = []
                sample_groups[sample_idx].append(i)
            
            # Get nuclei indices for the saved sample indices
            train_nuclei_indices = []
            val_nuclei_indices = []
            
            for sample_idx in self.train_indices:
                if sample_idx in sample_groups:
                    train_nuclei_indices.extend(sample_groups[sample_idx])
            
            for sample_idx in self.val_indices:
                if sample_idx in sample_groups:
                    val_nuclei_indices.extend(sample_groups[sample_idx])
            
            # Create instances subsets using nuclei indices
            train_instances = [self.nuclei_instances[i] for i in train_nuclei_indices]
            val_instances = [self.nuclei_instances[i] for i in val_nuclei_indices]
            
            # Store sample indices for consistency
            self.train_sample_indices = self.train_indices
            self.val_sample_indices = self.val_indices
            
            # Create datasets
            train_dataset = NucleiClassificationDataset(
                nuclei_instances=train_instances,
                augmentations=get_classification_augmentations(training=True),
                normalize=True
            )
            
            val_dataset = NucleiClassificationDataset(
                nuclei_instances=val_instances,
                augmentations=get_classification_augmentations(training=False),
                normalize=True
            )
            
            # Create dataloaders
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
                persistent_workers=True if self.num_workers > 0 else False
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True if self.num_workers > 0 else False
            )
            
            print(f"[NucleiDataModule] Resumed: {len(train_instances)} train, {len(val_instances)} val")
        else:
            # Create new split using the improved stratified method
            self.train_loader, self.val_loader, self.train_sample_indices, self.val_sample_indices = create_nuclei_dataloaders(
                nuclei_instances=self.nuclei_instances,
                train_ratio=self.train_ratio,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                random_seed=self.random_seed
            )
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation dataloaders."""
        if self.train_loader is None or self.val_loader is None:
            self.setup()
        return self.train_loader, self.val_loader
    
    def get_sample_indices(self) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """Get the actual sample indices used for train/val split (for proper saving)."""
        if self.train_loader is None or self.val_loader is None:
            self.setup()
        return self.train_sample_indices, self.val_sample_indices
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get class distribution in the dataset."""
        if self.nuclei_instances is None:
            self.nuclei_instances = load_nuclei_dataset(self.dataset_path)
        
        class_counts = {}
        class_names = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
        
        for nucleus in self.nuclei_instances:
            # Use original class_id (1-5), then map to class names
            class_id = nucleus['class_id']  # Keep original 1-5
            if 1 <= class_id <= 5:
                class_name = class_names[class_id - 1]  # Map to 0-4 for indexing
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return class_counts 