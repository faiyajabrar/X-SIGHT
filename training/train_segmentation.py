"""Advanced Attention U-Net training script with state-of-the-art optimization techniques.

This is the main training script that includes cutting-edge optimization for maximum performance:
- Progressive resizing for better feature learning
- OneCycleLR scheduling with superconvergence
- Exponential Moving Average (EMA) of weights
- Stochastic Weight Averaging (SWA)
- Advanced augmentations (MixUp, multi-scale)
- Hybrid loss with Focal and Boundary components
- Test Time Augmentation (TTA) for validation

Usage:
    python training/train.py --lr 3e-4 --epochs 60 --batch_size 16
    
Performance Features:
- Progressive resize: Better feature learning from coarse to fine
- OneCycleLR: Superconvergence for faster training
- EMA weights: Better validation performance 
- SWA: Improved final model performance
- Advanced augmentations: Better generalization
- Hybrid loss: Better boundary detection
- TTA: More robust validation metrics
"""

import sys
import os
import platform
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
import numpy as np
import albumentations as A
import cv2
import math
import copy
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
import json

from utils.pannuke_dataset import PanNukeDataset
from models.attention_unet import AttentionUNet

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix for Windows multiprocessing
if platform.system() == 'Windows':
    torch.multiprocessing.set_start_method('spawn', force=True)


def _calculate_class_wise_dice(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 6):
    """Calculate Dice score for each class individually."""
    eps = 1e-6
    class_names = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
    
    dice_scores = {}
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        
        if target_c.sum() == 0 and pred_c.sum() == 0:
            dice_c = torch.tensor(1.0, device=preds.device)  # Perfect score for empty class
        elif target_c.sum() == 0:
            dice_c = torch.tensor(0.0, device=preds.device)  # No ground truth
        else:
            inter = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_c = (2 * inter + eps) / (union + eps)
        
        dice_scores[f'dice_class_{c}_{class_names[c]}'] = dice_c
    
    return dice_scores


def _frequency_weighted_dice_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 6):
    """Frequency-weighted Dice that includes ALL classes with REASONABLE weighting."""
    eps = 1e-6
    
    # Frequency weights (square root of inverse frequency - less extreme)
    pixel_freq = torch.tensor([0.832818975, 0.0866185198, 0.0177438743, 0.0373645720, 0.000691303678, 0.0247627551], device=preds.device)
    class_weights = torch.sqrt(1.0 / (pixel_freq + eps))
    # Cap maximum weight to prevent extreme values
    class_weights = torch.clamp(class_weights, max=10.0)
    # Fixed normalization: maintain relative weights but normalize to mean=1.0
    class_weights = class_weights / class_weights.mean()
    
    dice_per_class = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        
        if target_c.sum() == 0 and pred_c.sum() == 0:
            dice_c = torch.tensor(1.0, device=preds.device)  # Perfect score for empty class
        else:
            inter = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_c = (2 * inter + eps) / (union + eps)
        
        # Apply frequency weighting
        weighted_dice = dice_c * class_weights[c]
        dice_per_class.append(weighted_dice)
    
    return torch.stack(dice_per_class).sum() / class_weights.sum()


class EMAWrapper:
    """Exponential Moving Average of model weights for better validation performance."""
    
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Ensure shadow tensor has same dtype and device as current parameter
                if (self.shadow[name].dtype != param.data.dtype or 
                    self.shadow[name].device != param.data.device):
                    self.shadow[name] = self.shadow[name].to(param.data.dtype).to(param.data.device)
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                # Ensure shadow tensor matches the parameter dtype and device before assignment
                shadow_data = self.shadow[name]
                if (shadow_data.dtype != param.data.dtype or 
                    shadow_data.device != param.data.device):
                    shadow_data = shadow_data.to(param.data.dtype).to(param.data.device)
                param.data = shadow_data

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def mixup_data(x, y, alpha=0.4):
    """Apply MixUp augmentation to input batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class HybridLoss(nn.Module):
    """Advanced hybrid loss combining Dice, Focal, and Boundary losses."""
    
    def __init__(self, num_classes=6, epsilon=1e-6, focal_alpha=0.25, focal_gamma=2.0, 
                 boundary_weight=0.1, dice_weight=0.7, focal_weight=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Normalize loss weights to sum to 1.0 for stable training
        total_weight = dice_weight + focal_weight + boundary_weight
        self.dice_weight = dice_weight / total_weight
        self.focal_weight = focal_weight / total_weight
        self.boundary_weight = boundary_weight / total_weight
        
        print(f"Hybrid loss weights (normalized): dice={self.dice_weight:.3f}, focal={self.focal_weight:.3f}, boundary={self.boundary_weight:.3f}")
        
        # Frequency weights for Dice component
        pixel_freq = torch.tensor([0.832818975, 0.0866185198, 0.0177438743, 0.0373645720, 0.000691303678, 0.0247627551])
        self.class_weights = torch.sqrt(1.0 / (pixel_freq + self.epsilon))
        self.class_weights = torch.clamp(self.class_weights, max=10.0)
        # Fixed normalization: maintain relative weights but normalize to mean=1.0
        self.class_weights = self.class_weights / self.class_weights.mean()
        print(f"Hybrid loss class weights: {self.class_weights.tolist()}")
    
    def dice_loss(self, y_pred, y_true):
        """Frequency-weighted Dice loss component."""
        y_pred_soft = F.softmax(y_pred, dim=1)
        y_true_onehot = F.one_hot(y_true, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        class_weights = self.class_weights.to(y_pred_soft.device)
        
        dice_coeffs = []
        for i in range(self.num_classes):
            intersection = torch.sum(y_true_onehot[:, i] * y_pred_soft[:, i])
            union = torch.sum(y_true_onehot[:, i]) + torch.sum(y_pred_soft[:, i])
            dice_coeff = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
            dice_coeffs.append(dice_coeff)
        
        dice_tensor = torch.stack(dice_coeffs)
        weighted_dice = (dice_tensor * class_weights).sum() / class_weights.sum()
        return 1.0 - weighted_dice
    
    def focal_loss(self, y_pred, y_true):
        """Focal loss for hard example mining."""
        y_pred_soft = F.softmax(y_pred, dim=1)
        y_true_onehot = F.one_hot(y_true, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Compute focal weights
        pt = torch.sum(y_pred_soft * y_true_onehot, dim=1)  # Shape: [B, H, W]
        alpha_t = self.focal_alpha
        focal_weight = alpha_t * (1 - pt) ** self.focal_gamma
        
        # Cross entropy
        ce_loss = F.cross_entropy(y_pred, y_true, reduction='none')
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()
    
    def boundary_loss(self, y_pred, y_true):
        """Boundary loss for better edge detection - Fixed for progressive resizing."""
        y_pred_soft = F.softmax(y_pred, dim=1)
        
        # Get spatial dimensions
        B, C, H, W = y_pred_soft.shape
        
        # Compute prediction gradients (edge detection) - handle edge cases
        if W > 1:
            pred_dx = torch.abs(y_pred_soft[:, :, :, 1:] - y_pred_soft[:, :, :, :-1])  # [B, C, H, W-1]
        else:
            pred_dx = torch.zeros(B, C, H, 1, device=y_pred_soft.device)
            
        if H > 1:
            pred_dy = torch.abs(y_pred_soft[:, :, 1:, :] - y_pred_soft[:, :, :-1, :])  # [B, C, H-1, W]
        else:
            pred_dy = torch.zeros(B, C, 1, W, device=y_pred_soft.device)
        
        # Target boundaries using simple gradient - handle edge cases
        y_true_float = y_true.float()
        if W > 1:
            true_dx = torch.abs(y_true_float[:, :, 1:] - y_true_float[:, :, :-1])  # [B, H, W-1]
        else:
            true_dx = torch.zeros(B, H, 1, device=y_true.device)
            
        if H > 1:
            true_dy = torch.abs(y_true_float[:, 1:, :] - y_true_float[:, :-1, :])  # [B, H-1, W]
        else:
            true_dy = torch.zeros(B, 1, W, device=y_true.device)
        
        # Simplified boundary loss - just use MSE on summed gradients
        # Sum across classes for prediction gradients
        pred_dx_sum = pred_dx.sum(dim=1)  # [B, H, W-1]
        pred_dy_sum = pred_dy.sum(dim=1)  # [B, H-1, W]
        
        # Normalize gradients to [0, 1] range
        pred_dx_norm = torch.clamp(pred_dx_sum / (pred_dx_sum.max() + self.epsilon), 0, 1)
        pred_dy_norm = torch.clamp(pred_dy_sum / (pred_dy_sum.max() + self.epsilon), 0, 1)
        
        # Normalize true gradients to [0, 1] range
        true_dx_norm = torch.clamp(true_dx / (true_dx.max() + self.epsilon), 0, 1)
        true_dy_norm = torch.clamp(true_dy / (true_dy.max() + self.epsilon), 0, 1)
        
        # Ensure spatial dimensions match for MSE computation
        min_h_dx = min(pred_dx_norm.shape[1], true_dx_norm.shape[1])
        min_w_dx = min(pred_dx_norm.shape[2], true_dx_norm.shape[2])
        
        min_h_dy = min(pred_dy_norm.shape[1], true_dy_norm.shape[1])
        min_w_dy = min(pred_dy_norm.shape[2], true_dy_norm.shape[2])
        
        # Compute MSE losses with matching dimensions
        if min_h_dx > 0 and min_w_dx > 0:
            boundary_loss_x = F.mse_loss(
                pred_dx_norm[:, :min_h_dx, :min_w_dx], 
                true_dx_norm[:, :min_h_dx, :min_w_dx]
            )
        else:
            boundary_loss_x = torch.tensor(0.0, device=y_pred.device)
        
        if min_h_dy > 0 and min_w_dy > 0:
            boundary_loss_y = F.mse_loss(
                pred_dy_norm[:, :min_h_dy, :min_w_dy], 
                true_dy_norm[:, :min_h_dy, :min_w_dy]
            )
        else:
            boundary_loss_y = torch.tensor(0.0, device=y_pred.device)
        
        return (boundary_loss_x + boundary_loss_y) / 2
    
    def forward(self, y_pred, y_true):
        dice_loss = self.dice_loss(y_pred, y_true)
        focal_loss = self.focal_loss(y_pred, y_true)
        boundary_loss = self.boundary_loss(y_pred, y_true)
        
        total_loss = (self.dice_weight * dice_loss + 
                     self.focal_weight * focal_loss + 
                     self.boundary_weight * boundary_loss)
        
        return total_loss


def get_progressive_augmentations(epoch, max_epochs):
    """Progressive augmentation strategy - start simple, add complexity."""
    base_prob = min(0.8, 0.3 + (epoch / max_epochs) * 0.5)  # Gradually increase augmentation intensity
    
    augmentations = [
        # Always apply basic flips
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
    ]
    
    # Add geometric transforms progressively
    if epoch > 5:
        augmentations.extend([
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=20,
                interpolation=1,
                border_mode=0,
                p=base_prob * 0.6
            ),
            A.GridDistortion(distort_limit=0.15, p=base_prob * 0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=base_prob * 0.3),
        ])
    
    # Add color/contrast augmentations
    if epoch > 10:
        augmentations.extend([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=base_prob * 0.4
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=20,
                val_shift_limit=15,
                p=base_prob * 0.4
            ),
        ])
    
    # Add noise and blur
    if epoch > 15:
        augmentations.extend([
            A.OneOf([
                A.GaussianBlur(blur_limit=(1, 5), p=0.4),
                A.MedianBlur(blur_limit=5, p=0.4),
                A.GaussNoise(var_limit=(10, 80), p=0.4),
            ], p=base_prob * 0.4),
        ])
    
    return A.Compose(augmentations)


class AdvancedAttentionModel(pl.LightningModule):
    """Lightning module with state-of-the-art optimization techniques."""

    def __init__(self, lr=3e-4, dropout=0.1, weight_decay=1e-4, 
                 warmup_epochs=5, total_epochs=60, min_lr_factor=0.01, steps_per_epoch=None,
                 use_mixup=True, mixup_alpha=0.4, use_ema=True, ema_decay=0.9999,
                 progressive_resize=True, start_size=128, end_size=256, use_tta=False, resume_path=None):
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.model = AttentionUNet(n_classes=6, dropout=dropout, pretrained=True)
        
        # Advanced hybrid loss
        self.criterion = HybridLoss(num_classes=6, epsilon=1e-6)
        
        # Optimization parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_factor = min_lr_factor
        self.steps_per_epoch = steps_per_epoch
        
        # Advanced training features
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_ema = use_ema
        self.use_tta = use_tta
        self.progressive_resize = progressive_resize
        self.start_size = start_size
        self.end_size = end_size
        
        # Ensure start_size < end_size for valid progressive training
        if self.start_size >= self.end_size:
            logger.warning(f"Invalid progressive resize parameters: start_size ({self.start_size}) >= end_size ({self.end_size})")
            self.start_size = 128  # Reset to a safe default
            logger.warning(f"Reset start_size to {self.start_size}")
        
        # Initialize EMA
        if self.use_ema:
            self.ema = EMAWrapper(self.model, decay=ema_decay)
        
        # Progressive resize tracking
        self.current_size = start_size if progressive_resize else end_size
        
        # Progressive resizing cache - FIX: Cache current image size per epoch
        self._cached_image_size = None
        self._cached_epoch = -1
        
        # Resume state management
        self.resume_path = resume_path
        self.training_state_path = Path('lightning_logs/training_state.json')
        self.data_split_path = Path('lightning_logs/data_split.json')
        
        # Best metrics tracking for resume
        self.best_val_dice = 0.0
        self.best_val_loss = float('inf')
        
        # Load resume state if provided
        if resume_path:
            self.load_training_state()
    
    def validate_data_split_consistency(self, current_train_indices, current_val_indices):
        """Validate that resumed data split matches current split for perfect consistency."""
        if not hasattr(self, '_train_indices') or not hasattr(self, '_val_indices'):
            logger.info("🆕 No saved data split found - using current split")
            return True
        
        if (self._train_indices is None or self._val_indices is None or
            current_train_indices is None or current_val_indices is None):
            logger.warning("⚠️  Missing data split indices - consistency cannot be verified")
            return False
        
        # Check if splits match exactly
        train_match = set(self._train_indices) == set(current_train_indices)
        val_match = set(self._val_indices) == set(current_val_indices)
        
        if train_match and val_match:
            logger.info("✅ Data split consistency verified - perfect resume guaranteed")
            return True
        else:
            logger.error("❌ DATA SPLIT MISMATCH DETECTED!")
            logger.error(f"   Saved train indices: {len(self._train_indices)}, Current: {len(current_train_indices)}")
            logger.error(f"   Saved val indices: {len(self._val_indices)}, Current: {len(current_val_indices)}")
            logger.error(f"   Train match: {train_match}, Val match: {val_match}")
            logger.error("   This could cause validation Dice score drops!")
            logger.error("   Consider using saved indices or starting fresh training.")
            return False

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        """Update progressive augmentations at the start of each epoch."""
        # Cache the current image size for this epoch to ensure consistency
        if self.progressive_resize:
            self._cached_image_size = self._calculate_current_image_size()
            self._cached_epoch = self.current_epoch
            logger.info(f"Progressive augmentations updated for epoch {self.current_epoch}: image size {self._cached_image_size}x{self._cached_image_size}")
        
        # Access the training dataloader's dataset
        train_dataloader = self.trainer.train_dataloader
        if hasattr(train_dataloader, 'dataset'):
            dataset = train_dataloader.dataset
            # Check if it's a Subset (from train/val split)
            if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'update_epoch'):
                dataset.dataset.update_epoch(self.current_epoch)
            # Or if it's directly a ProgressiveDataset
            elif hasattr(dataset, 'update_epoch'):
                dataset.update_epoch(self.current_epoch)
    
    def _calculate_current_image_size(self):
        """Calculate the current image size based on progressive resizing."""
        if not self.progressive_resize:
            return self.end_size
            
        # Use mock current_epoch for testing if available, otherwise use actual current_epoch
        current_epoch = getattr(self, '_current_epoch', self.current_epoch)
        
        # Reach full size at 70% of training for optimal progressive learning (84 epochs for 120 total)
        progress = min(1.0, current_epoch / (self.total_epochs * 0.7))
        current_size = int(self.start_size + (self.end_size - self.start_size) * progress)
        return max(self.start_size, min(self.end_size, current_size))
    
    def get_current_image_size(self):
        """Get the current image size, using cached value for consistency within epoch."""
        if self.progressive_resize:
            # Use cached size if available for current epoch
            if self._cached_epoch == self.current_epoch and self._cached_image_size is not None:
                return self._cached_image_size
            else:
                # Calculate and cache for new epoch
                self._cached_image_size = self._calculate_current_image_size()
                self._cached_epoch = self.current_epoch
                return self._cached_image_size
        return self.end_size

    def training_step(self, batch, batch_idx):
        imgs, masks = batch['image'], batch['mask']
        
        # Progressive resizing
        if self.progressive_resize:
            target_size = self.get_current_image_size()
            if imgs.shape[-1] != target_size:
                imgs = F.interpolate(imgs, size=(target_size, target_size), mode='bilinear', align_corners=False)
                masks = F.interpolate(masks.unsqueeze(1).float(), size=(target_size, target_size), mode='nearest').squeeze(1).long()
        
        # Check for NaN in inputs
        if torch.isnan(imgs).any() or torch.isnan(masks).any():
            logger.warning(f"NaN detected in inputs at batch {batch_idx}")
            return None
        
        # Apply MixUp augmentation
        if self.use_mixup and self.training:
            if np.random.random() < 0.3:  # Apply MixUp to 30% of batches
                imgs, masks_a, masks_b, lam = mixup_data(imgs, masks, self.mixup_alpha)
                logits = self(imgs)
                
                if torch.isnan(logits).any():
                    logger.warning(f"NaN detected in logits at batch {batch_idx}")
                    return None
                
                loss = mixup_criterion(self.criterion, logits, masks_a, masks_b, lam)
            else:
                logits = self(imgs)
                if torch.isnan(logits).any():
                    return None
                loss = self.criterion(logits, masks)
        else:
            logits = self(imgs)
            if torch.isnan(logits).any():
                return None
            loss = self.criterion(logits, masks)
        
        if torch.isnan(loss):
            logger.warning(f"NaN detected in training loss at batch {batch_idx}")
            return None
        
        # Update EMA weights
        if self.use_ema and self.training:
            self.ema.update()
        
        # Compute monitoring metrics
        preds = torch.argmax(logits, dim=1)
        dice_score = _frequency_weighted_dice_score(preds, masks, num_classes=6)
        accuracy = (preds == masks).float().mean()
        
        # Calculate class-wise Dice scores
        class_dice_scores = _calculate_class_wise_dice(preds, masks, num_classes=6)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_dice', dice_score, on_epoch=True)
        self.log('train_acc', accuracy, prog_bar=True, on_epoch=True)
        self.log('current_image_size', float(self.get_current_image_size()), on_epoch=True)
        
        # Log class-wise training Dice scores
        for class_name, score in class_dice_scores.items():
            self.log(f'train_{class_name}', score, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch['image'], batch['mask']
        
        # Apply same progressive resizing to validation data for consistency
        if self.progressive_resize:
            target_size = self.get_current_image_size()
            if imgs.shape[-1] != target_size:
                imgs = F.interpolate(imgs, size=(target_size, target_size), mode='bilinear', align_corners=False)
                masks = F.interpolate(masks.unsqueeze(1).float(), size=(target_size, target_size), mode='nearest').squeeze(1).long()
        
        # Test Time Augmentation (TTA) for better validation metrics
        tta_preds = []
        
        # Original image
        logits = self(imgs)
        tta_preds.append(F.softmax(logits, dim=1))
        
        # TTA transformations
        if self.use_tta:
            # Horizontal flip
            imgs_hflip = torch.flip(imgs, dims=[3])
            logits_hflip = self(imgs_hflip)
            logits_hflip = torch.flip(logits_hflip, dims=[3])
            tta_preds.append(F.softmax(logits_hflip, dim=1))
            
            # Vertical flip
            imgs_vflip = torch.flip(imgs, dims=[2])
            logits_vflip = self(imgs_vflip)
            logits_vflip = torch.flip(logits_vflip, dims=[2])
            tta_preds.append(F.softmax(logits_vflip, dim=1))
        
        # Average TTA predictions
        if len(tta_preds) > 1:
            avg_probs = torch.stack(tta_preds).mean(0)
            logits = torch.log(avg_probs + 1e-8)  # Convert back to logits
        
        # Use EMA weights for validation if available
        if self.use_ema and hasattr(self, 'ema'):
            self.ema.apply_shadow()
            logits_ema = self(imgs)
            self.ema.restore()
            
            # Blend EMA and current model predictions
            logits = 0.7 * logits + 0.3 * logits_ema
        
        loss = self.criterion(logits, masks)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        dice_score = _frequency_weighted_dice_score(preds, masks, num_classes=6)
        accuracy = (preds == masks).float().mean()
        
        # Class-wise validation metrics
        class_dice_scores = _calculate_class_wise_dice(preds, masks, num_classes=6)
        
        # Check data distribution for debugging
        unique_preds = torch.unique(preds).cpu().numpy()
        unique_targets = torch.unique(masks).cpu().numpy()
        logger.info(f"Epoch {self.current_epoch} unique preds {unique_preds}, targets {unique_targets}")
        
        # Log class-wise Dice scores with proper names
        class_names = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
        dice_str_parts = []
        for i, name in enumerate(class_names):
            score = class_dice_scores[f'dice_class_{i}_{name}'].item()
            dice_str_parts.append(f"{name}: {score:.3f}")
        logger.info(f"Epoch {self.current_epoch} Class-wise Dice: {', '.join(dice_str_parts)}")
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dice', dice_score, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        
        # Log individual class performance
        for class_name, score in class_dice_scores.items():
            self.log(f'val_{class_name}', score)
        
        return loss

    def configure_optimizers(self):
        """Configure advanced optimizer with OneCycleLR for superconvergence."""
        
        # Use AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Calculate total steps
        if self.steps_per_epoch is None:
            logger.warning("steps_per_epoch not provided, using fallback approximation")
            steps_per_epoch = 445
        else:
            steps_per_epoch = self.steps_per_epoch
            
        total_steps = self.total_epochs * steps_per_epoch
        
        logger.info(f"OneCycleLR scheduler: {steps_per_epoch} steps/epoch, {total_steps} total steps")
        
        # OneCycleLR for superconvergence (optimized for 120-epoch training)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=0.25,  # 25% warmup (30 epochs for 120 total)
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25,  # Initial LR = max_lr / div_factor
            final_div_factor=10000  # Final LR = max_lr / final_div_factor
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "onecycle"
            }
        }
    
    def save_training_state(self):
        """Save training state for resuming."""
        self.training_state_path.parent.mkdir(exist_ok=True)
        
        # Get current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr'] if self.trainer else self.lr
        
        # Save EMA state if available
        ema_state = None
        if self.use_ema and hasattr(self, 'ema'):
            try:
                # Convert shadow weights to CPU and check for NaN/Inf
                shadow_dict = {}
                for name, param in self.ema.shadow.items():
                    cpu_param = param.cpu()
                    if torch.isnan(cpu_param).any() or torch.isinf(cpu_param).any():
                        logger.warning(f"⚠️  Skipping EMA shadow parameter {name} due to NaN/Inf values")
                        continue
                    shadow_dict[name] = cpu_param.tolist()
                
                # Convert backup weights if available
                backup_dict = {}
                if self.ema.backup:
                    for name, param in self.ema.backup.items():
                        cpu_param = param.cpu()
                        if torch.isnan(cpu_param).any() or torch.isinf(cpu_param).any():
                            logger.warning(f"⚠️  Skipping EMA backup parameter {name} due to NaN/Inf values")
                            continue
                        backup_dict[name] = cpu_param.tolist()
                
                ema_state = {
                    'decay': self.ema.decay,
                    'shadow': shadow_dict,
                    'backup': backup_dict
                }
            except Exception as e:
                logger.warning(f"⚠️  Failed to save EMA state: {e}")
                ema_state = None
        
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step if hasattr(self, 'global_step') else 0,
            'best_val_dice': self.best_val_dice,
            'best_val_loss': self.best_val_loss,
            'current_lr': current_lr,
            'current_image_size': self.current_size,
            'data_split': {
                'train_indices': getattr(self, '_train_indices', None),
                'val_indices': getattr(self, '_val_indices', None)
            },
            'ema_state': ema_state,
            'random_state': {
                'python': np.random.get_state()[1].tolist(),  # Fix: Save all 625 elements
                'numpy_pos': int(np.random.get_state()[2]),  # Fix: Save position separately
                'torch': torch.get_rng_state().tolist()
            },
            'hyperparameters': {
                'lr': self.lr,
                'dropout': self.hparams.get('dropout', 0.1),
                'weight_decay': self.weight_decay,
                'total_epochs': self.total_epochs,
                'progressive_resize': self.progressive_resize,
                'start_size': self.start_size,
                'end_size': self.end_size,
                'use_mixup': self.use_mixup,
                'use_ema': self.use_ema,
                'use_tta': self.use_tta
            },
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(self.training_state_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"💾 Training state saved to {self.training_state_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save training state: {e}")
            logger.error("Resume capability may be compromised!")
        logger.info(f"   ✅ Data split indices: {'saved' if state['data_split']['train_indices'] else 'not available'}")
        logger.info(f"   ✅ EMA state: {'saved' if ema_state else 'not available'}")
        logger.info(f"   ✅ Random states: saved for perfect reproducibility")
    
    def load_training_state(self):
        """Load training state for resuming."""
        if not self.training_state_path.exists():
            logger.warning(f"⚠️  Training state file not found: {self.training_state_path}")
            return False
        
        try:
            with open(self.training_state_path, 'r') as f:
                state = json.load(f)
            
            # Restore metrics
            self.best_val_dice = state.get('best_val_dice', 0.0)
            self.best_val_loss = state.get('best_val_loss', float('inf'))
            self.current_size = state.get('current_image_size', self.start_size)
            
            # Restore data split indices
            if 'data_split' in state:
                self._train_indices = state['data_split'].get('train_indices', None)
                self._val_indices = state['data_split'].get('val_indices', None)
                logger.info(f"   📊 Data split indices restored: {len(self._train_indices) if self._train_indices else 0} train, {len(self._val_indices) if self._val_indices else 0} val")
            
            # Restore EMA state
            if 'ema_state' in state and state['ema_state'] and self.use_ema and hasattr(self, 'ema'):
                try:
                    ema_state = state['ema_state']
                    self.ema.decay = ema_state['decay']
                    
                    # Restore shadow weights (use current parameter device/dtype for safety)
                    for name, param_list in ema_state['shadow'].items():
                        if name in self.ema.shadow:
                            # Get device and dtype from current parameter
                            current_param = self.ema.shadow[name]
                            device = current_param.device
                            dtype = current_param.dtype
                            self.ema.shadow[name] = torch.tensor(param_list, device=device, dtype=dtype)
                    
                    # Restore backup weights if available
                    if ema_state['backup']:
                        for name, param_list in ema_state['backup'].items():
                            if name in self.ema.shadow:  # Only restore if corresponding shadow exists
                                device = self.ema.shadow[name].device
                                self.ema.backup[name] = torch.tensor(param_list, device=device)
                    
                    logger.info(f"   ✅ EMA state restored successfully")
                except Exception as e:
                    logger.warning(f"   ⚠️  Failed to restore EMA state: {e}")
            
            # Restore random states for reproducibility
            if 'random_state' in state:
                random_state = state['random_state']
                if 'python' in random_state and 'numpy_pos' in random_state:
                    # Fix: Properly reconstruct numpy random state
                    np.random.set_state(('MT19937', np.array(random_state['python']), random_state['numpy_pos'], 0, 0.0))
                if 'torch' in random_state:
                    torch.set_rng_state(torch.tensor(random_state['torch'], dtype=torch.uint8))
                logger.info(f"   ✅ Random states restored for reproducibility")
            
            logger.info(f"🔄 Training state loaded from {self.training_state_path}")
            logger.info(f"   Resuming from epoch {state.get('epoch', 0)}")
            logger.info(f"   Best val dice: {self.best_val_dice:.4f}")
            logger.info(f"   Current image size: {self.current_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load training state: {e}")
            return False
    
    def on_validation_epoch_end(self):
        """Track best metrics and save training state."""
        # Get current validation metrics
        val_dice = self.trainer.callback_metrics.get('val_dice', 0.0)
        val_loss = self.trainer.callback_metrics.get('val_loss', float('inf'))
        
        # Update best metrics
        if val_dice > self.best_val_dice:
            self.best_val_dice = val_dice
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        
        # Save training state after each epoch
        self.save_training_state()
        
        # Log progress with resume info
        logger.info(f"📊 Epoch {self.current_epoch}: val_dice={val_dice:.4f} (best={self.best_val_dice:.4f})")
        logger.info(f"🔄 Training state saved - can resume from epoch {self.current_epoch + 1}")


class ProgressiveDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies progressive augmentations based on current epoch."""
    
    def __init__(self, base_dataset, total_epochs=60):
        self.base_dataset = base_dataset
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # Start with basic augmentations
        self.current_augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
        ])
    
    def update_epoch(self, epoch):
        """Update augmentations based on current epoch."""
        self.current_epoch = epoch
        self.current_augmentations = get_progressive_augmentations(epoch, self.total_epochs)
        
        # Update the base dataset's augmentations
        self.base_dataset.augmentations = self.current_augmentations
        logger.info(f"Progressive augmentations updated for epoch {epoch}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        return self.base_dataset[idx]


def get_dataloaders(batch_size=16, total_epochs=60, base_size=256, save_split=True, saved_split_path=None):
    """Get dataloaders with progressive augmentations and dataset's built-in transforms.
    
    Args:
        save_split: Whether to save the data split indices for perfect resume
        saved_split_path: Path to saved split indices (for resume)
    """
    
    # Create base datasets - training will use progressive augmentations, validation none
    base_train_dataset = PanNukeDataset(root='Dataset', augmentations=None, validate_dataset=False, base_size=base_size)  # Start with no augs
    ds_val = PanNukeDataset(root='Dataset', augmentations=None, validate_dataset=False, base_size=base_size)

    # Wrap training dataset with progressive augmentations
    ds_train_progressive = ProgressiveDataset(base_train_dataset, total_epochs=total_epochs)

    # Try to load existing split for perfect resume
    train_indices = None
    val_indices = None
    
    if saved_split_path and Path(saved_split_path).exists():
        try:
            with open(saved_split_path, 'r') as f:
                split_data = json.load(f)
            # Ensure indices are integers (JSON may load them as different types)
            train_indices = [int(idx) for idx in split_data['train_indices']]
            val_indices = [int(idx) for idx in split_data['val_indices']]
            logger.info(f"🔄 Loaded existing data split from {saved_split_path}")
            logger.info(f"   Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load saved split: {e}. Creating new split.")
    
    # Create new split if not loaded
    if train_indices is None or val_indices is None:
        # Deterministic split for reproducibility - same seed always gives same split
        val_size = int(0.1 * len(ds_train_progressive))
        train_size = len(ds_train_progressive) - val_size
        
        # Use fixed seed for completely reproducible data splits across runs
        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset = torch.utils.data.random_split(
            range(len(ds_train_progressive)), [train_size, val_size], generator=generator
        )
        
        train_indices = train_subset.indices
        val_indices = val_subset.indices
        
        # Save the split for perfect resume capability
        if save_split and saved_split_path:
            try:
                Path(saved_split_path).parent.mkdir(exist_ok=True)
                split_data = {
                    'train_indices': train_indices,
                    'val_indices': val_indices,
                    'total_samples': len(ds_train_progressive),
                    'train_size': len(train_indices),
                    'val_size': len(val_indices),
                    'split_ratio': len(val_indices) / len(ds_train_progressive),
                    'seed': 42,
                    'timestamp': datetime.now().isoformat()
                }
                with open(saved_split_path, 'w') as f:
                    json.dump(split_data, f, indent=2)
                logger.info(f"💾 Data split saved to {saved_split_path} for perfect resume")
            except Exception as e:
                logger.warning(f"⚠️  Failed to save data split: {e}. Resume capability may be limited.")
        
        logger.info("📊 Created new deterministic data split")
    
    ds_train = torch.utils.data.Subset(ds_train_progressive, train_indices)
    ds_val = torch.utils.data.Subset(ds_val, val_indices)

    # Use num_workers=0 on Windows
    num_workers = 0 if platform.system() == 'Windows' else 4

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, persistent_workers=False
    )
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=False
    )
    
    logger.info(f"📊 Data split: {len(train_indices)} train samples, {len(val_indices)} validation samples")
    logger.info("🔄 Using deterministic data split with saved indices for perfect resume")
    
    return train_loader, val_loader, train_indices, val_indices


def main():
    parser = argparse.ArgumentParser()
    # Basic training parameters
    parser.add_argument('--lr', type=float, default=3e-4, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for Attention U-Net')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=120, help='Number of epochs')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    
    # Advanced optimization parameters
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--min_lr_factor', type=float, default=0.01, help='Minimum LR as factor of initial LR')
    parser.add_argument('--early_stopping_patience', type=int, default=30, help='Early stopping patience (epochs)')
    parser.add_argument('--grad_clip_val', type=float, default=1.0, help='Gradient clipping value')
    
    # State-of-the-art features
    parser.add_argument('--use_swa', action='store_true', help='Use Stochastic Weight Averaging')
    parser.add_argument('--swa_start_epoch', type=int, default=60, help='Epoch to start SWA')
    parser.add_argument('--use_tta', action='store_true', help='Use Test Time Augmentation for validation')
    parser.add_argument('--disable_ema', action='store_true', help='Disable Exponential Moving Average')
    
    # Resume functionality
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint file for resuming')
    
    args = parser.parse_args()
    
    print("🚀 ADVANCED ATTENTION U-NET TRAINING 🚀")
    print("="*50)
    print(f"Platform: {platform.system()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("\n📋 Training Configuration:")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Dropout: {args.dropout}")
    print("\n⚡ Advanced Optimization:")
    print(f"  Optimizer: AdamW (weight_decay={args.weight_decay})")
    print(f"  Scheduler: OneCycleLR for superconvergence")
    print(f"  Min LR factor: {args.min_lr_factor} (final LR = {args.lr * args.min_lr_factor:.2e})")
    print(f"  Gradient clipping: {args.grad_clip_val}")
    print(f"  Early stopping patience: {args.early_stopping_patience} epochs")
    print(f"  Stochastic Weight Averaging: {'✅ Enabled' if args.use_swa else '❌ Disabled'}")
    print("\n🎯 State-of-the-Art Features:")
    print("  - Attention U-Net architecture")  
    print("  - CLAHE + Z-score preprocessing (built into dataset)")
    print("  - Progressive resizing training (128→256px, full size at epoch 84)")
    print("  - Progressive augmentations (complexity increases with epoch)")
    print("  - MixUp augmentation + advanced transforms")
    print("  - Hybrid loss (Dice + Focal + Boundary)")
    print("  - Exponential Moving Average (EMA) weights")
    print("  - Test Time Augmentation (TTA) for validation")
    print("  - Class-wise performance monitoring")
    print(f"  - Test Time Augmentation: {'✅ Enabled' if args.use_tta else '❌ Disabled'}")
    print(f"  - Exponential Moving Average: {'❌ Disabled' if args.disable_ema else '✅ Enabled'}")
    print("="*50)
    
    # Get dataloaders with progressive augmentations and perfect resume capability
    data_split_path = Path('lightning_logs/data_split.json')
    
    # First attempt to get dataloaders (may use saved split if available)
    train_loader, val_loader, train_indices, val_indices = get_dataloaders(
        batch_size=args.batch_size, 
        total_epochs=args.epochs, 
        base_size=256,
        save_split=True,
        saved_split_path=str(data_split_path)
    )
    
    # Calculate actual steps per epoch for proper LR scheduling
    steps_per_epoch = len(train_loader)
    print(f"📊 Dataset info: {len(train_loader)} train batches, {len(val_loader)} val batches")
    print(f"🔄 Learning rate scheduler will use {steps_per_epoch} steps per epoch")
    print(f"🎨 Progressive augmentations: Start simple, increase complexity over {args.epochs} epochs")
    print(f"💾 Data split indices saved for perfect resume capability")
    
    # Handle resume functionality
    resume_checkpoint = None
    if args.resume:
        if args.checkpoint_path and Path(args.checkpoint_path).exists():
            resume_checkpoint = args.checkpoint_path
            print(f"🔄 Resuming from specified checkpoint: {resume_checkpoint}")
        else:
            # Look for latest checkpoint
            checkpoint_dir = Path('lightning_logs')
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob('**/last.ckpt'))
                if checkpoints:
                    resume_checkpoint = str(checkpoints[-1])  # Use most recent
                    print(f"🔄 Auto-resuming from latest checkpoint: {resume_checkpoint}")
                else:
                    print("⚠️  No checkpoint found for resume, starting fresh")
            else:
                print("⚠️  No lightning_logs directory found, starting fresh")
    
    # Create model with state-of-the-art optimization techniques
    model = AdvancedAttentionModel(
        lr=args.lr, 
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr_factor=args.min_lr_factor,
        steps_per_epoch=steps_per_epoch,
        use_tta=args.use_tta,
        use_ema=not args.disable_ema,
        resume_path=resume_checkpoint
    )
    
    # Store data split indices in model for perfect resume capability
    # But preserve any loaded indices from training state first
    if not hasattr(model, '_train_indices') or model._train_indices is None:
        model._train_indices = train_indices
    if not hasattr(model, '_val_indices') or model._val_indices is None:
        model._val_indices = val_indices
    
    # Validate data split consistency for perfect resume (only if resuming)
    if args.resume and resume_checkpoint:
        is_consistent = model.validate_data_split_consistency(train_indices, val_indices)
        if not is_consistent and hasattr(model, '_train_indices') and model._train_indices is not None:
            print("\n" + "="*80)
            print("🔄 USING SAVED DATA SPLIT FOR PERFECT RESUME")
            print("="*80)
            print("Detected data split mismatch. Automatically using saved indices from previous training.")
            print("This ensures validation Dice scores remain consistent across resume.")
            print("="*80)
            
            # Recreate dataloaders with saved indices
            try:
                # Create the same base datasets
                base_train_dataset = PanNukeDataset(root='Dataset', augmentations=None, validate_dataset=False, base_size=256)
                ds_val = PanNukeDataset(root='Dataset', augmentations=None, validate_dataset=False, base_size=256)
                ds_train_progressive = ProgressiveDataset(base_train_dataset, total_epochs=args.epochs)
                
                # Use saved indices
                ds_train = torch.utils.data.Subset(ds_train_progressive, model._train_indices)
                ds_val_resume = torch.utils.data.Subset(ds_val, model._val_indices)
                
                # Recreate dataloaders
                num_workers = 0 if platform.system() == 'Windows' else 4
                train_loader = DataLoader(
                    ds_train, batch_size=args.batch_size, shuffle=True, 
                    num_workers=num_workers, pin_memory=True, persistent_workers=False
                )
                val_loader = DataLoader(
                    ds_val_resume, batch_size=args.batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True, persistent_workers=False
                )
                
                # Update indices to match what we're actually using
                train_indices = model._train_indices
                val_indices = model._val_indices
                
                print(f"✅ Successfully using saved data split: {len(train_indices)} train, {len(val_indices)} val samples")
                
            except Exception as e:
                logger.error(f"❌ Failed to recreate dataloaders with saved indices: {e}")
                print("Falling back to current data split - validation scores may be inconsistent.")
        
        elif not is_consistent:
            print("\n" + "="*80)
            print("⚠️  WARNING: DATA SPLIT INCONSISTENCY DETECTED!")
            print("="*80)
            print("The current data split doesn't match the saved split from previous training.")
            print("This could cause validation Dice scores to appear to drop artificially.")
            print("No saved indices available to automatically correct this.")
            print("="*80)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_dice',  # Monitor overall weighted Dice score
        mode='max',
        save_top_k=3,
        filename='advanced-{epoch:02d}-{val_dice:.3f}',
        save_last=True
    )
    
    # Additional callback to monitor rare class (Dead cells) performance
    dead_cell_checkpoint = ModelCheckpoint(
        monitor='val_dice_class_4_Dead',  # Monitor Dead cell Dice specifically
        mode='max',
        save_top_k=1,
        filename='advanced-best-dead-{epoch:02d}-{val_dice_class_4_Dead:.3f}',
        save_last=False
    )
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_dice',
        mode='max',
        patience=args.early_stopping_patience,
        verbose=True,
        min_delta=0.001  # Minimum change to qualify as improvement
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Add Stochastic Weight Averaging if enabled
    callbacks = [checkpoint_callback, dead_cell_checkpoint, early_stopping, lr_monitor]
    if args.use_swa:
        swa_callback = StochasticWeightAveraging(
            swa_lrs=args.lr * 0.1,  # SWA learning rate (10% of initial)
            swa_epoch_start=args.swa_start_epoch,
            annealing_epochs=10  # Number of epochs to anneal SWA LR
        )
        callbacks.append(swa_callback)
        print(f"🔄 SWA enabled: Starting at epoch {args.swa_start_epoch} with LR {args.lr * 0.1:.2e}")
    
    # Create trainer with state-of-the-art optimization techniques
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.gpus,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        callbacks=callbacks,
        log_every_n_steps=50,
        enable_progress_bar=True,
        precision=32,  # Use full 32-bit precision
        gradient_clip_val=args.grad_clip_val,  # Enhanced gradient clipping
        gradient_clip_algorithm='norm',  # Use L2 norm clipping
        # Optimize for best performance
        enable_model_summary=True,
        benchmark=True,  # Optimize cudnn for consistent input sizes
    )
    
    # Train
    if resume_checkpoint:
        print(f"\n🔄 Resuming training from checkpoint!")
        print(f"🏃 Training will continue with preserved state and data split")
    else:
        print(f"\n🏃 Starting fresh training with proper learning rate scheduling!")
    
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_checkpoint)
    
    # Training summary
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"📁 Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"🎯 Best validation Dice score: {checkpoint_callback.best_model_score:.3f}")
    
    if early_stopping.stopped_epoch > 0:
        print(f"⏰ Early stopping triggered at epoch {early_stopping.stopped_epoch}")
        print("   Training stopped automatically when convergence was detected.")
    else:
        print("⏱️  Training completed full epochs without early stopping")
        print("   Consider increasing epochs if loss is still decreasing.")


if __name__ == '__main__':
    # Essential for Windows multiprocessing
    if platform.system() == 'Windows':
        torch.multiprocessing.freeze_support()
    
    main() 