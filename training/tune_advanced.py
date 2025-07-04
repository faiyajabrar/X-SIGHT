"""Advanced Hyperparameter Tuning with Optuna for Attention U-Net.

This script performs comprehensive hyperparameter optimization using the same
advanced training pipeline as the main training script, including:
- Progressive resizing and augmentations
- Hybrid loss optimization
- Advanced optimizer tuning
- Architecture parameter tuning
- Training strategy optimization

Usage:
    python training/tune_advanced.py --n_trials 100 --study_name attention_unet_optimization
"""

import sys
import os
import platform
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
import numpy as np
import albumentations as A
import pickle
import json
from datetime import datetime
from pathlib import Path
import traceback

from utils.pannuke_dataset import PanNukeDataset
from models.attention_unet import AttentionUNet

# Import functions from main training script
from training.train import (
    _frequency_weighted_dice_score,
    EMAWrapper, 
    mixup_data, 
    mixup_criterion,
    HybridLoss,
    ProgressiveDataset
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix for Windows multiprocessing
if platform.system() == 'Windows':
    torch.multiprocessing.set_start_method('spawn', force=True)


class TuningAttentionModel(pl.LightningModule):
    """Lightning module for hyperparameter tuning with Optuna integration."""

    def __init__(self, trial, total_epochs=10, steps_per_epoch=445):
        super().__init__()
        self.trial = trial
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        
        # Suggest hyperparameters
        self.lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        self.dropout = trial.suggest_float('dropout', 0.05, 0.3)
        self.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        
        # Advanced optimization parameters
        self.grad_clip_val = trial.suggest_float('grad_clip_val', 0.1, 2.0)
        self.mixup_alpha = trial.suggest_float('mixup_alpha', 0.1, 0.8)
        self.ema_decay = trial.suggest_float('ema_decay', 0.995, 0.9999)
        
        # Progressive training parameters (multiples of 32 for ResNet compatibility)
        self.start_size = trial.suggest_categorical('start_size', [128, 160, 192])
        self.end_size = trial.suggest_categorical('end_size', [224, 256, 288])
        
        # Ensure start_size < end_size for valid progressive training
        if self.start_size >= self.end_size:
            self.start_size = 128  # Reset to smallest valid size
        
        # Loss function parameters
        self.focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.5)
        self.focal_gamma = trial.suggest_float('focal_gamma', 1.0, 3.0)
        self.dice_weight = trial.suggest_float('dice_weight', 0.5, 0.8)
        self.focal_weight = trial.suggest_float('focal_weight', 0.1, 0.3)
        self.boundary_weight = trial.suggest_float('boundary_weight', 0.05, 0.2)
        
        # OneCycleLR parameters
        self.pct_start = trial.suggest_float('pct_start', 0.2, 0.4)
        self.div_factor = trial.suggest_int('div_factor', 10, 50)
        self.final_div_factor = trial.suggest_int('final_div_factor', 100, 10000, log=True)
        
        # Advanced features toggles
        self.use_mixup = trial.suggest_categorical('use_mixup', [True, False])
        self.use_ema = trial.suggest_categorical('use_ema', [True, False])
        self.progressive_resize = trial.suggest_categorical('progressive_resize', [True, False])
        
        # Model architecture
        self.model = AttentionUNet(n_classes=6, dropout=self.dropout, pretrained=True)
        
        # Advanced hybrid loss with tuned parameters
        self.criterion = HybridLoss(
            num_classes=6, 
            epsilon=1e-6,
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma,
            boundary_weight=self.boundary_weight,
            dice_weight=self.dice_weight,
            focal_weight=self.focal_weight
        )
        
        # Initialize EMA if enabled
        if self.use_ema:
            self.ema = EMAWrapper(self.model, decay=self.ema_decay)
        
        # Progressive resize tracking
        self.current_size = self.start_size if self.progressive_resize else self.end_size
        
        # Track best validation score for pruning
        self.best_val_dice = 0.0

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        """Update progressive augmentations at the start of each epoch."""
        train_dataloader = self.trainer.train_dataloader
        if hasattr(train_dataloader, 'dataset'):
            dataset = train_dataloader.dataset
            if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'update_epoch'):
                dataset.dataset.update_epoch(self.current_epoch)
            elif hasattr(dataset, 'update_epoch'):
                dataset.update_epoch(self.current_epoch)
    
    def get_current_image_size(self):
        """Progressive resizing: start small, gradually increase to full size."""
        if not self.progressive_resize:
            return self.end_size
            
        progress = min(1.0, self.current_epoch / (self.total_epochs * 0.5))  # Faster progression for short training
        current_size = int(self.start_size + (self.end_size - self.start_size) * progress)
        return max(self.start_size, min(self.end_size, current_size))

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
            return None
        
        # Apply MixUp augmentation
        if self.use_mixup and self.training:
            if np.random.random() < 0.3:  # Apply MixUp to 30% of batches
                imgs, masks_a, masks_b, lam = mixup_data(imgs, masks, self.mixup_alpha)
                logits = self(imgs)
                
                if torch.isnan(logits).any():
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
            return None
        
        # Update EMA weights
        if self.use_ema and self.training:
            self.ema.update()
        
        # Compute monitoring metrics
        preds = torch.argmax(logits, dim=1)
        dice_score = _frequency_weighted_dice_score(preds, masks, num_classes=6)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_dice', dice_score, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch['image'], batch['mask']
        
        # Use EMA weights for validation if available
        if self.use_ema and hasattr(self, 'ema'):
            self.ema.apply_shadow()
            logits = self(imgs)
            self.ema.restore()
        else:
            logits = self(imgs)
        
        loss = self.criterion(logits, masks)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        dice_score = _frequency_weighted_dice_score(preds, masks, num_classes=6)
        
        # Compute pixel accuracy
        correct_pixels = (preds == masks).sum().float()
        total_pixels = masks.numel()
        accuracy = correct_pixels / total_pixels
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dice', dice_score, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        
        return dice_score

    def on_validation_epoch_end(self):
        """Report intermediate results to Optuna for pruning."""
        val_dice = self.trainer.callback_metrics.get('val_dice', 0.0)
        
        # Update best score
        if val_dice > self.best_val_dice:
            self.best_val_dice = val_dice
        
        # Report to Optuna
        self.trial.report(val_dice, self.current_epoch)
        
        # Check if trial should be pruned
        if self.trial.should_prune():
            raise optuna.TrialPruned()

    def configure_optimizers(self):
        """Configure optimizer and scheduler with suggested hyperparameters."""
        
        # Use AdamW optimizer with tuned parameters
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        total_steps = self.total_epochs * self.steps_per_epoch
        
        # OneCycleLR with tuned parameters
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=self.pct_start,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor
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


def get_tuning_dataloaders(batch_size=16, total_epochs=10):
    """Get dataloaders for hyperparameter tuning (smaller dataset for speed)."""
    
    # Create base datasets
    base_train_dataset = PanNukeDataset(root='Dataset', augmentations=None, validate_dataset=False)
    ds_val = PanNukeDataset(root='Dataset', augmentations=None, validate_dataset=False)

    # Wrap training dataset with progressive augmentations
    ds_train_progressive = ProgressiveDataset(base_train_dataset, total_epochs=total_epochs)

    # Use smaller subset for faster tuning
    total_size = len(ds_train_progressive)
    val_size = int(0.1 * total_size)
    train_size = total_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(
        range(total_size), [train_size, val_size], generator=generator
    )
    
    ds_train = torch.utils.data.Subset(ds_train_progressive, train_subset.indices)
    ds_val = torch.utils.data.Subset(ds_val, val_subset.indices)

    # Use num_workers=0 on Windows
    num_workers = 0 if platform.system() == 'Windows' else 2

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, persistent_workers=False
    )
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=False
    )
    
    return train_loader, val_loader


def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    
    try:
        # Get dataloaders
        train_loader, val_loader = get_tuning_dataloaders(batch_size=16, total_epochs=10)
        steps_per_epoch = len(train_loader)
        
        # Create model with suggested hyperparameters
        model = TuningAttentionModel(trial, total_epochs=10, steps_per_epoch=steps_per_epoch)
        
        # Setup callbacks for tuning (no checkpointing needed)
        early_stopping = EarlyStopping(
            monitor='val_dice',
            mode='max',
            patience=4,  # Shorter patience for faster tuning (10 epochs total)
            verbose=False,
            min_delta=0.001
        )
        
        # Create trainer for tuning (shorter epochs, less logging)
        trainer = pl.Trainer(
            max_epochs=10,  # Shorter training for faster tuning
            devices=1,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            callbacks=[early_stopping],  # Only early stopping, no checkpointing
            logger=False,  # Disable logging for speed
            enable_progress_bar=True,  # Enable progress bar to see epoch progress
            precision=32,
            gradient_clip_val=model.grad_clip_val,
            gradient_clip_algorithm='norm',
            enable_model_summary=False,
            enable_checkpointing=False,  # Disable checkpointing for speed
        )
        
        # Train the model
        trainer.fit(model, train_loader, val_loader)
        
        # Return the best validation Dice score
        return model.best_val_dice
        
    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        logger.error(traceback.format_exc())
        return 0.0  # Return poor score for failed trials


def save_best_params(study, study_name):
    """Save the best parameters to files."""
    
    # Create tune_runs directory if it doesn't exist
    tune_dir = Path('tune_runs')
    tune_dir.mkdir(exist_ok=True)
    
    # Save study object
    with open(tune_dir / f'{study_name}.pkl', 'wb') as f:
        pickle.dump(study, f)
    
    # Save best parameters as JSON
    best_params = study.best_params.copy()
    best_params['best_value'] = study.best_value
    best_params['n_trials'] = len(study.trials)
    best_params['timestamp'] = datetime.now().isoformat()
    
    with open(tune_dir / f'{study_name}_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Save detailed results
    results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            result = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            }
            results.append(result)
    
    with open(tune_dir / f'{study_name}_all_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to tune_runs/{study_name}*")


def main():
    parser = argparse.ArgumentParser(description='Advanced Hyperparameter Tuning with Optuna')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials to run')
    parser.add_argument('--study_name', type=str, default='attention_unet_advanced', help='Study name')
    parser.add_argument('--storage', type=str, default=None, help='Database URL for study storage')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs (use 1 for GPU)')
    parser.add_argument('--resume', action='store_true', help='Resume existing study')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    
    args = parser.parse_args()
    
    print("🎯 ADVANCED HYPERPARAMETER TUNING WITH OPTUNA 🎯")
    print("="*60)
    print(f"Study name: {args.study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Platform: {platform.system()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("\n🔍 Hyperparameters to optimize:")
    print("  - Learning rate (1e-5 to 1e-2)")
    print("  - Dropout (0.05 to 0.3)")
    print("  - Weight decay (1e-6 to 1e-2)")
    print("  - Gradient clipping (0.1 to 2.0)")
    print("  - MixUp alpha (0.1 to 0.8)")
    print("  - EMA decay (0.995 to 0.9999)")
    print("  - Progressive sizing (96-160 to 224-288)")
    print("  - Loss weights (Dice, Focal, Boundary)")
    print("  - OneCycleLR parameters")
    print("  - Feature toggles (MixUp, EMA, Progressive)")
    print("="*60)
    
    # Create tune_runs directory if it doesn't exist
    tune_dir = Path('tune_runs')
    tune_dir.mkdir(exist_ok=True)
    
    # Setup storage
    if args.storage:
        storage = args.storage
    else:
        storage = f"sqlite:///tune_runs/{args.study_name}.db"
    
    # Create or load study
    if args.resume:
        try:
            study = optuna.load_study(
                study_name=args.study_name,
                storage=storage,
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
            )
            print(f"📂 Resumed study with {len(study.trials)} existing trials")
        except:
            print("⚠️  Could not resume study, creating new one")
            study = optuna.create_study(
                direction='maximize',
                study_name=args.study_name,
                storage=storage,
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1),
                load_if_exists=True
            )
    else:
        study = optuna.create_study(
            direction='maximize',
            study_name=args.study_name,
            storage=storage,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1),
            load_if_exists=True
        )
    
    print(f"\n🚀 Starting optimization...")
    
    # Run optimization
    study.optimize(
        objective, 
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        show_progress_bar=True
    )
    
    # Print results
    print(f"\n✅ OPTIMIZATION COMPLETE!")
    print(f"🏆 Best trial: {study.best_trial.number}")
    print(f"🎯 Best validation Dice: {study.best_value:.4f}")
    print(f"\n📊 Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    save_best_params(study, args.study_name)
    
    print(f"\n💾 Results saved to tune_runs/{args.study_name}*")
    print("🎯 Use these parameters in your training script for optimal performance!")


if __name__ == '__main__':
    # Essential for Windows multiprocessing
    if platform.system() == 'Windows':
        torch.multiprocessing.freeze_support()
    
    main() 