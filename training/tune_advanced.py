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
        
        # Track reported epochs to avoid duplicate reporting to Optuna
        self.reported_epochs = set()
        
        # Trial state for mid-trial resume
        self.trial_checkpoint_dir = Path('tune_runs') / 'trial_checkpoints'
        self.trial_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.trial_checkpoint_path = self.trial_checkpoint_dir / f'trial_{trial.number}_checkpoint.json'

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
        
        # Check for NaN in inputs with detailed logging
        if torch.isnan(imgs).any():
            logger.error(f"Trial {self.trial.number}: NaN detected in input images at batch {batch_idx}")
            logger.error(f"  Image stats: min={imgs.min():.6f}, max={imgs.max():.6f}, mean={imgs.mean():.6f}")
            raise RuntimeError("NaN in input images")
        
        if torch.isnan(masks).any():
            logger.error(f"Trial {self.trial.number}: NaN detected in masks at batch {batch_idx}")
            raise RuntimeError("NaN in masks")
        
        # Check for invalid mask values
        if masks.min() < 0 or masks.max() >= 6:
            logger.error(f"Trial {self.trial.number}: Invalid mask values - min={masks.min()}, max={masks.max()}")
            raise RuntimeError("Invalid mask values")

        # Apply MixUp augmentation
        if self.use_mixup and self.training:
            if np.random.random() < 0.3:  # Apply MixUp to 30% of batches
                imgs, masks_a, masks_b, lam = mixup_data(imgs, masks, self.mixup_alpha)
                logits = self(imgs)
                
                if torch.isnan(logits).any():
                    logger.error(f"Trial {self.trial.number}: NaN detected in logits (mixup) at batch {batch_idx}")
                    logger.error(f"  Logits stats: min={logits.min():.6f}, max={logits.max():.6f}")
                    raise RuntimeError("NaN in logits during mixup")
                
                loss = mixup_criterion(self.criterion, logits, masks_a, masks_b, lam)
            else:
                logits = self(imgs)
                if torch.isnan(logits).any():
                    logger.error(f"Trial {self.trial.number}: NaN detected in logits (no-mixup) at batch {batch_idx}")
                    logger.error(f"  Logits stats: min={logits.min():.6f}, max={logits.max():.6f}")
                    raise RuntimeError("NaN in logits")
                loss = self.criterion(logits, masks)
        else:
            logits = self(imgs)
            if torch.isnan(logits).any():
                logger.error(f"Trial {self.trial.number}: NaN detected in logits (standard) at batch {batch_idx}")
                logger.error(f"  Logits stats: min={logits.min():.6f}, max={logits.max():.6f}")
                raise RuntimeError("NaN in logits")
            loss = self.criterion(logits, masks)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Trial {self.trial.number}: Invalid loss detected at batch {batch_idx}")
            logger.error(f"  Loss value: {loss.item()}")
            logger.error(f"  Logits stats: min={logits.min():.6f}, max={logits.max():.6f}, mean={logits.mean():.6f}")
            logger.error(f"  Learning rate: {self.trainer.optimizers[0].param_groups[0]['lr']:.2e}")
            raise RuntimeError("Invalid loss value")

        # Update EMA weights
        if self.use_ema and self.training:
            self.ema.update()

        # Compute monitoring metrics
        preds = torch.argmax(logits, dim=1)
        dice_score = _frequency_weighted_dice_score(preds, masks, num_classes=6)
        
        # Check dice score validity
        if torch.isnan(dice_score) or torch.isinf(dice_score):
            logger.error(f"Trial {self.trial.number}: Invalid dice score at batch {batch_idx}: {dice_score}")
            dice_score = torch.tensor(0.0, device=dice_score.device)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_dice', dice_score, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch['image'], batch['mask']
        
        # Apply same progressive resizing to validation data for consistency
        if self.progressive_resize:
            target_size = self.get_current_image_size()
            if imgs.shape[-1] != target_size:
                imgs = F.interpolate(imgs, size=(target_size, target_size), mode='bilinear', align_corners=False)
                masks = F.interpolate(masks.unsqueeze(1).float(), size=(target_size, target_size), mode='nearest').squeeze(1).long()

        # Check for invalid inputs
        if torch.isnan(imgs).any():
            logger.error(f"Trial {self.trial.number}: NaN in validation images at batch {batch_idx}")
            raise RuntimeError("NaN in validation images")
        
        if torch.isnan(masks).any() or masks.min() < 0 or masks.max() >= 6:
            logger.error(f"Trial {self.trial.number}: Invalid validation masks at batch {batch_idx}")
            raise RuntimeError("Invalid validation masks")

        # Use EMA weights for validation if available
        if self.use_ema and hasattr(self, 'ema'):
            self.ema.apply_shadow()
            logits = self(imgs)
            self.ema.restore()
        else:
            logits = self(imgs)

        if torch.isnan(logits).any():
            logger.error(f"Trial {self.trial.number}: NaN in validation logits at batch {batch_idx}")
            logger.error(f"  Logits stats: min={logits.min():.6f}, max={logits.max():.6f}")
            raise RuntimeError("NaN in validation logits")

        loss = self.criterion(logits, masks)
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Trial {self.trial.number}: Invalid validation loss at batch {batch_idx}: {loss.item()}")
            raise RuntimeError("Invalid validation loss")

        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        dice_score = _frequency_weighted_dice_score(preds, masks, num_classes=6)

        # Compute pixel accuracy
        correct_pixels = (preds == masks).sum().float()
        total_pixels = masks.numel()
        accuracy = correct_pixels / total_pixels
        
        # Check metric validity
        if torch.isnan(dice_score) or torch.isinf(dice_score):
            logger.error(f"Trial {self.trial.number}: Invalid validation dice at batch {batch_idx}: {dice_score}")
            dice_score = torch.tensor(0.0, device=dice_score.device)
        
        if torch.isnan(accuracy) or torch.isinf(accuracy):
            logger.error(f"Trial {self.trial.number}: Invalid validation accuracy at batch {batch_idx}: {accuracy}")
            accuracy = torch.tensor(0.0, device=accuracy.device)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dice', dice_score, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)

        return dice_score

    def save_trial_checkpoint(self):
        """Save trial state for mid-trial resume."""
        # Convert tensor values to Python types for JSON serialization
        best_val_dice = self.best_val_dice
        if hasattr(best_val_dice, 'item'):
            best_val_dice = best_val_dice.item()
        elif torch.is_tensor(best_val_dice):
            best_val_dice = float(best_val_dice)
        
        checkpoint_data = {
            'trial_number': self.trial.number,
            'completed_epoch': self.current_epoch,
            'best_val_dice': best_val_dice,
            'reported_epochs': list(self.reported_epochs),
            'trial_params': self.trial.params,
            'train_indices': getattr(self, '_train_indices', None),
            'val_indices': getattr(self, '_val_indices', None),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.trial_checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def load_trial_checkpoint(self):
        """Load trial state for mid-trial resume."""
        if self.trial_checkpoint_path.exists():
            with open(self.trial_checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.best_val_dice = checkpoint_data.get('best_val_dice', 0.0)
            self.reported_epochs = set(checkpoint_data.get('reported_epochs', []))
            
            # Load data split indices for consistent resumption
            self._train_indices = checkpoint_data.get('train_indices', None)
            self._val_indices = checkpoint_data.get('val_indices', None)
            
            return checkpoint_data.get('completed_epoch', -1)
        return -1

    def cleanup_trial_checkpoint(self):
        """Remove trial checkpoint after successful completion."""
        if self.trial_checkpoint_path.exists():
            self.trial_checkpoint_path.unlink()

    def on_validation_epoch_end(self):
        """Report intermediate results to Optuna for pruning."""
        val_dice = self.trainer.callback_metrics.get('val_dice', 0.0)
        
        # Convert tensor to float for consistent handling
        if torch.is_tensor(val_dice):
            val_dice = val_dice.item()
        
        # Update best score
        if val_dice > self.best_val_dice:
            self.best_val_dice = val_dice
        
        # Report to Optuna only if this epoch hasn't been reported yet
        if self.current_epoch not in self.reported_epochs:
            self.trial.report(val_dice, self.current_epoch)
            self.reported_epochs.add(self.current_epoch)
        
        # Save trial checkpoint after each epoch
        self.save_trial_checkpoint()
        
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


def find_incomplete_trial():
    """Find the most recent incomplete trial checkpoint."""
    checkpoint_dir = Path('tune_runs') / 'trial_checkpoints'
    if not checkpoint_dir.exists():
        return None
    
    checkpoint_files = list(checkpoint_dir.glob('trial_*_checkpoint.json'))
    if not checkpoint_files:
        return None
    
    # Find the most recent checkpoint
    latest_checkpoint = None
    latest_time = None
    
    for checkpoint_file in checkpoint_files:
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            timestamp = datetime.fromisoformat(data['timestamp'])
            if latest_time is None or timestamp > latest_time:
                latest_time = timestamp
                latest_checkpoint = data
                latest_checkpoint['checkpoint_file'] = checkpoint_file
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    
    return latest_checkpoint

def cleanup_completed_trial_checkpoints(study):
    """Remove checkpoints for trials that are now completed in the study."""
    checkpoint_dir = Path('tune_runs') / 'trial_checkpoints'
    if not checkpoint_dir.exists():
        return
    
    completed_trial_numbers = {trial.number for trial in study.trials 
                              if trial.state == optuna.trial.TrialState.COMPLETE}
    
    for checkpoint_file in checkpoint_dir.glob('trial_*_checkpoint.json'):
        try:
            trial_number = int(checkpoint_file.stem.split('_')[1])
            if trial_number in completed_trial_numbers:
                checkpoint_file.unlink()
        except (ValueError, IndexError):
            continue

def get_tuning_dataloaders(batch_size=16, total_epochs=10, train_indices=None, val_indices=None):
    """Get dataloaders for hyperparameter tuning (smaller dataset for speed)."""
    
    # Create base datasets
    base_train_dataset = PanNukeDataset(root='Dataset', augmentations=None, validate_dataset=False)
    ds_val = PanNukeDataset(root='Dataset', augmentations=None, validate_dataset=False)

    # Wrap training dataset with progressive augmentations
    ds_train_progressive = ProgressiveDataset(base_train_dataset, total_epochs=total_epochs)

    # Use provided indices or create new split
    if train_indices is not None and val_indices is not None:
        logger.info("🔄 Using saved data split indices for consistent resumption")
        # Convert to lists if they're not already (JSON serialization)
        if isinstance(train_indices, str):
            train_indices = json.loads(train_indices)
        if isinstance(val_indices, str):
            val_indices = json.loads(val_indices)
    else:
        # Create new split with deterministic seed
        total_size = len(ds_train_progressive)
        val_size = int(0.1 * total_size)
        train_size = total_size - val_size
        
        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset = torch.utils.data.random_split(
            range(total_size), [train_size, val_size], generator=generator
        )
        train_indices = train_subset.indices
        val_indices = val_subset.indices
        logger.info("📊 Created new data split with deterministic seed")
    
    ds_train = torch.utils.data.Subset(ds_train_progressive, train_indices)
    ds_val = torch.utils.data.Subset(ds_val, val_indices)

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
    
    return train_loader, val_loader, train_indices, val_indices


def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    
    try:
        logger.info(f"🚀 Starting trial {trial.number} with parameters: {trial.params}")
        
        # Create model with suggested hyperparameters first to access checkpoint loading
        model = TuningAttentionModel(trial, total_epochs=10, steps_per_epoch=445)  # Use approximate steps first
        
        # Check if we can resume this trial from a checkpoint
        resume_epoch = model.load_trial_checkpoint()
        start_epoch = max(0, resume_epoch + 1)  # Start from next epoch after checkpoint
        
        # Get saved data split indices if resuming
        saved_train_indices = getattr(model, '_train_indices', None)
        saved_val_indices = getattr(model, '_val_indices', None)
        
        # Get dataloaders with consistent data split
        train_loader, val_loader, train_indices, val_indices = get_tuning_dataloaders(
            batch_size=16, total_epochs=10, 
            train_indices=saved_train_indices, val_indices=saved_val_indices
        )
        
        # Save data split indices in model for future checkpoints
        model._train_indices = train_indices
        model._val_indices = val_indices
        
        # Update steps per epoch with actual value
        steps_per_epoch = len(train_loader)
        model.steps_per_epoch = steps_per_epoch
        
        if resume_epoch >= 0:
            logger.info(f"🔄 Resuming trial {trial.number} from epoch {start_epoch}")
        
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
        
        # Train the model (will skip completed epochs if resuming)
        if start_epoch < 10:  # Only train if there are epochs left
            # Manually set the current epoch for resume
            trainer.fit_loop.epoch_progress.current.ready = start_epoch
            trainer.fit_loop.epoch_progress.current.started = start_epoch
            trainer.fit_loop.epoch_progress.current.processed = start_epoch
            
            logger.info(f"🏋️ Training trial {trial.number} from epoch {start_epoch} to 10")
            trainer.fit(model, train_loader, val_loader)
        else:
            logger.info(f"Trial {trial.number} already completed all epochs")
        
        # Clean up checkpoint on successful completion
        model.cleanup_trial_checkpoint()
        
        # Return the best validation Dice score
        best_dice = model.best_val_dice
        logger.info(f"✅ Trial {trial.number} completed successfully with dice: {best_dice:.4f}")
        return best_dice
        
    except optuna.TrialPruned:
        logger.info(f"✂️ Trial {trial.number} was pruned")
        raise
    except KeyboardInterrupt:
        logger.info(f"⌨️ Trial {trial.number} interrupted by user (Ctrl+C)")
        # Save checkpoint if model was created
        if 'model' in locals():
            try:
                model.save_trial_checkpoint()
                logger.info(f"💾 Trial {trial.number} state saved for potential resume")
            except:
                pass
        raise optuna.TrialPruned()  # Convert KeyboardInterrupt to pruned trial
    except RuntimeError as e:
        error_msg = str(e).lower()
        
        if "nan" in error_msg:
            logger.error(f"💥 Trial {trial.number} failed due to NaN values: {e}")
            logger.error(f"   Parameters: {trial.params}")
            logger.error("   This suggests unstable hyperparameters - consider:")
            logger.error("   - Lower learning rate")
            logger.error("   - Higher epsilon in loss")
            logger.error("   - Lower focal gamma")
            logger.error("   - Disable progressive resizing")
            
        elif "cuda" in error_msg or "memory" in error_msg:
            logger.error(f"🧠 Trial {trial.number} failed due to memory/CUDA issue: {e}")
            logger.error("   Try reducing batch size or model complexity")
            
        elif "size" in error_msg or "dimension" in error_msg:
            logger.error(f"📐 Trial {trial.number} failed due to tensor size mismatch: {e}")
            logger.error("   This usually happens with progressive resizing")
            
        else:
            logger.error(f"⚠️ Trial {trial.number} failed with RuntimeError: {e}")
            logger.error(traceback.format_exc())
        
        # Save checkpoint if model was created for debugging
        if 'model' in locals():
            try:
                model.save_trial_checkpoint()
                logger.info(f"💾 Failed trial {trial.number} state saved for debugging")
            except:
                pass
        
        # Don't return 0.0 - let trial fail properly
        raise optuna.TrialPruned()  # Convert runtime errors to pruned trials
        
    except Exception as e:
        logger.error(f"💀 Trial {trial.number} failed with unexpected error: {e}")
        logger.error(traceback.format_exc())
        
        # Save parameters for debugging
        logger.error(f"   Failed parameters: {trial.params}")
        
        # Save checkpoint if model was created
        if 'model' in locals():
            try:
                model.save_trial_checkpoint()
                logger.info(f"💾 Failed trial {trial.number} state saved for debugging")
            except:
                pass
        
        # Don't return 0.0 - let trial fail properly
        raise optuna.TrialPruned()  # Convert all errors to pruned trials


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
    
    # Clean up completed trial checkpoints
    cleanup_completed_trial_checkpoints(study)
    
    # Check for incomplete trials to resume
    incomplete_trial = find_incomplete_trial()
    if incomplete_trial and args.resume:
        print(f"\n🔄 Found incomplete trial {incomplete_trial['trial_number']}")
        print(f"   Completed {incomplete_trial['completed_epoch'] + 1}/10 epochs")
        print(f"   Best Dice so far: {incomplete_trial['best_val_dice']:.4f}")
        print(f"   Will resume from epoch {incomplete_trial['completed_epoch'] + 2}")
        
        # Enqueue the incomplete trial with its exact parameters
        try:
            study.enqueue_trial(incomplete_trial['trial_params'])
            print(f"📝 Enqueued incomplete trial for resumption")
        except Exception as e:
            logger.error(f"Failed to enqueue incomplete trial: {e}")
            print(f"⚠️  Will proceed with new trials instead")
    
    # Create callback to save intermediate results
    def save_intermediate_callback(study, trial):
        """Save results after each trial completion."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            save_best_params(study, args.study_name)
    
    # Calculate trials status for better progress tracking
    total_existing_trials = len(study.trials)
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    
    print(f"\n📊 Current study status:")
    print(f"   Total trials attempted: {total_existing_trials}")
    print(f"   Successful trials: {completed_trials}")
    print(f"   Failed trials: {failed_trials}")
    print(f"   Target trials: {args.n_trials}")
    
    # Calculate remaining trials based on total attempts, not just successful ones
    remaining_trials = max(0, args.n_trials - total_existing_trials)
    
    if remaining_trials > 0:
        print(f"\n🎯 Running {remaining_trials} additional trials...")
        print(f"   Overall progress: {total_existing_trials}/{args.n_trials} trials attempted")
        print(f"   This session will run: {remaining_trials} new trials")
        
        # Custom progress tracking callback
        def progress_callback(study, trial):
            current_total = len(study.trials)
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            truly_successful = len([t for t in completed_trials if t.value > 0.01])  # Filter out 0.0 fallback values
            failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
            
            print(f"📊 Overall Progress: {current_total}/{args.n_trials} trials")
            print(f"   ✅ Truly successful: {truly_successful}")
            print(f"   ❌ Failed: {failed_trials}")
            print(f"   🔄 Returned fallback (0.0): {len(completed_trials) - truly_successful}")
            
            if truly_successful > 0:
                best_value = max(t.value for t in completed_trials if t.value > 0.01)
                print(f"   🏆 Current best: {best_value:.4f}")
            
            save_intermediate_callback(study, trial)
        
        # Run optimization for remaining trials
        study.optimize(
            objective, 
            n_trials=remaining_trials,
            timeout=args.timeout,
            n_jobs=args.n_jobs,
            show_progress_bar=True,  # This shows session progress (0/remaining_trials)
            callbacks=[progress_callback]
        )
    else:
        print(f"\n✅ Target of {args.n_trials} trials already reached!")
        print(f"   Success rate: {completed_trials}/{total_existing_trials} = {100*completed_trials/total_existing_trials:.1f}%")
    
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