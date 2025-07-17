"""Backup Training Script: Proven Adam + Cosine Annealing Configuration

This is a reliable backup training script using the proven configuration that achieved
0.576 validation Dice score. It includes:
- Adam optimizer (not AdamW) for proven stability
- Cosine annealing with warmup scheduler
- Frequency-weighted Dice loss (proven effective)
- Basic gradient clipping
- No experimental features (MixUp, EMA, SWA, etc.)

Usage:
    python training/train_backup.py --lr 2e-4 --epochs 50 --batch_size 16
    
Proven Results:
- Validation Dice: 0.576 (69% improvement over baseline)
- Neoplastic: 0.773, Dead: 0.202, Background: 0.962
- Stable and reliable training without experimental features
"""

import sys
import os
import platform
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
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
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize
    
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


class ProvenDiceLoss(nn.Module):
    """Frequency-weighted Dice Loss - the proven configuration that works."""
    
    def __init__(self, num_classes=6, epsilon=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        
        # Frequency weights (square root of inverse frequency - proven effective)
        pixel_freq = torch.tensor([0.832818975, 0.0866185198, 0.0177438743, 0.0373645720, 0.000691303678, 0.0247627551])
        self.class_weights = torch.sqrt(1.0 / (pixel_freq + self.epsilon))
        # Cap maximum weight to prevent extreme values
        self.class_weights = torch.clamp(self.class_weights, max=10.0)
        # Normalize so weights sum to num_classes (maintains scale)
        self.class_weights = self.class_weights / self.class_weights.sum() * self.num_classes
        print(f"Proven Dice loss class weights: {self.class_weights.tolist()}")
    
    def forward(self, y_pred, y_true):
        # Convert logits to probabilities
        y_pred_soft = F.softmax(y_pred, dim=1)
        
        # Convert target to one-hot 
        y_true_onehot = F.one_hot(y_true, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Move class weights to same device
        class_weights = self.class_weights.to(y_pred_soft.device)
        
        dice_coeffs = []
        for i in range(self.num_classes):
            # Calculate Dice coefficient for each class
            intersection = torch.sum(y_true_onehot[:, i] * y_pred_soft[:, i])
            union = torch.sum(y_true_onehot[:, i]) + torch.sum(y_pred_soft[:, i])
            
            dice_coeff = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
            dice_coeffs.append(dice_coeff)
        
        # Apply frequency weights to dice coefficients
        dice_tensor = torch.stack(dice_coeffs)
        weighted_dice = (dice_tensor * class_weights).sum() / class_weights.sum()
        
        # Return loss (1 - weighted dice coefficient)
        return 1.0 - weighted_dice


class BackupAttentionModel(pl.LightningModule):
    """Backup model with proven Adam + Cosine Annealing configuration."""

    def __init__(self, lr=2e-4, dropout=0.1, weight_decay=1e-4, 
                 warmup_epochs=5, total_epochs=50, min_lr_factor=0.01, steps_per_epoch=None):
        super().__init__()
        self.save_hyperparameters()
        
        # Use proven Attention U-Net model
        self.model = AttentionUNet(n_classes=6, dropout=dropout, pretrained=True)
        
        # Use proven Dice loss
        self.dice_loss = ProvenDiceLoss(num_classes=6, epsilon=1e-6)
        
        # Store hyperparameters for optimizer configuration
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_factor = min_lr_factor
        self.steps_per_epoch = steps_per_epoch

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch['image'], batch['mask']
        
        # Check for NaN in inputs
        if torch.isnan(imgs).any() or torch.isnan(masks).any():
            logger.warning(f"NaN detected in inputs at batch {batch_idx}")
            return None
        
        logits = self(imgs)
        
        # Check for NaN in outputs
        if torch.isnan(logits).any():
            logger.warning(f"NaN detected in logits at batch {batch_idx}")
            return None
        
        # Proven Dice loss only
        loss = self.dice_loss(logits, masks)
        
        if torch.isnan(loss):
            logger.warning(f"NaN detected in training loss at batch {batch_idx}")
            return None
        
        # Compute monitoring metrics
        preds = torch.argmax(logits, dim=1)
        dice_score = _frequency_weighted_dice_score(preds, masks, num_classes=6)
        accuracy = (preds == masks).float().mean()
        
        # Calculate class-wise Dice scores
        class_dice_scores = _calculate_class_wise_dice(preds, masks, num_classes=6)
        
        # Logging
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_dice', dice_score, on_epoch=True)
        self.log('train_acc', accuracy, on_epoch=True)
        
        # Log class-wise dice scores
        for class_name, score in class_dice_scores.items():
            self.log(f'train_{class_name}', score, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch['image'], batch['mask']
        
        logits = self(imgs)
        loss = self.dice_loss(logits, masks)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        dice_score = _frequency_weighted_dice_score(preds, masks, num_classes=6)
        accuracy = (preds == masks).float().mean()
        
        # Calculate class-wise Dice scores
        class_dice_scores = _calculate_class_wise_dice(preds, masks, num_classes=6)
        
        # Logging
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dice', dice_score, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        
        # Log class-wise dice scores
        for class_name, score in class_dice_scores.items():
            self.log(f'val_{class_name}', score)
        
        return dice_score

    def configure_optimizers(self):
        """Configure Adam optimizer with cosine annealing and warmup."""
        
        # Use Adam (not AdamW) for proven stability
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # If steps_per_epoch is not provided, estimate it
        if self.steps_per_epoch is None:
            self.steps_per_epoch = 445  # Approximate based on dataset size
        
        total_steps = self.total_epochs * self.steps_per_epoch
        warmup_steps = self.warmup_epochs * self.steps_per_epoch
        
        # Cosine annealing with warmup - proven effective
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            
            # Cosine annealing
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_factor + (1.0 - self.min_lr_factor) * cosine_factor
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "cosine_annealing_warmup"
            }
        }


def get_dataloaders(batch_size=16):
    """Get dataloaders for backup training."""
    
    # Simple augmentations for proven configuration
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5,
            border_mode=cv2.BORDER_CONSTANT, value=0
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        ], p=0.3),
    ])
    
    # Create datasets
    ds_train = PanNukeDataset(root='Dataset', augmentations=train_transforms, validate_dataset=False)
    ds_val = PanNukeDataset(root='Dataset', augmentations=None, validate_dataset=False)

    # Split datasets
    total_size = len(ds_train)
    val_size = int(0.1 * total_size)
    train_size = total_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(
        range(total_size), [train_size, val_size], generator=generator
    )
    
    ds_train = torch.utils.data.Subset(ds_train, train_subset.indices)
    ds_val = torch.utils.data.Subset(ds_val, val_subset.indices)

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
    
    logger.info(f"Training samples: {len(ds_train)}")
    logger.info(f"Validation samples: {len(ds_val)}")
    
    return train_loader, val_loader


def main():
    """Main training function with proven configuration."""
    parser = argparse.ArgumentParser(description='Backup Attention U-Net Training')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (proven: 2e-4)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (proven: 50)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (proven: 16)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (proven: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (proven: 1e-4)')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs (proven: 5)')
    parser.add_argument('--min_lr_factor', type=float, default=0.01, help='Min LR factor (proven: 0.01)')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("BACKUP TRAINING: Proven Adam + Cosine Annealing Configuration")
    logger.info("=" * 80)
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Dropout: {args.dropout}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Warmup epochs: {args.warmup_epochs}")
    logger.info("Features: Adam optimizer + Cosine annealing + Frequency-weighted Dice loss")
    logger.info("Expected: Validation Dice ~0.576 (proven reliable performance)")
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=args.batch_size)
    steps_per_epoch = len(train_loader)
    
    # Create model
    model = BackupAttentionModel(
        lr=args.lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr_factor=args.min_lr_factor,
        steps_per_epoch=steps_per_epoch
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_dice',
        mode='max',
        save_top_k=3,
        dirpath='lightning_logs/backup_checkpoints',
        filename='backup-{epoch:02d}-{val_dice:.3f}',
        save_last=True,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    early_stopping = EarlyStopping(
        monitor='val_dice',
        mode='max',
        patience=15,  # Longer patience for stable training
        verbose=True,
        min_delta=0.001
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        log_every_n_steps=50,
        gradient_clip_val=1.0,  # Conservative gradient clipping
        enable_progress_bar=True,
        precision=32,  # Full precision for maximum stability
    )
    
    # Train model
    logger.info("Starting backup training with proven configuration...")
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model and report final metrics
    best_model = BackupAttentionModel.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        lr=args.lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr_factor=args.min_lr_factor,
        steps_per_epoch=steps_per_epoch
    )
    
    logger.info("=" * 80)
    logger.info("BACKUP TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Best model: {checkpoint_callback.best_model_path}")
    logger.info(f"Best validation Dice: {checkpoint_callback.best_model_score:.4f}")
    logger.info("Expected reliable performance similar to proven 0.576 validation Dice")
    logger.info("This configuration provides stable training without experimental features")


if __name__ == "__main__":
    main() 