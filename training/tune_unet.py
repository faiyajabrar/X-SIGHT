"""Hyper-parameter tuning for Attention U-Net on PanNuke with Optuna + PyTorch-Lightning.

Usage:
    pip install optuna pytorch-lightning torchinfo
    python tune_unet.py --max-trials 50 --gpus 1

The script searches over:
    • learning rate (1e-5 … 1e-2 log-uniform)
    • weight decay (1e-6 … 1e-3 log-uniform)
    • dropout (0.0 … 0.3 uniform)
    • batch size (4, 8, 12, 16)
    • optimiser (AdamW / SGD)
    • scheduler (None / CosineAnnealing / ReduceLROnPlateau)

Metrics:
    Validation Dice score (mean over 6 classes).

The best trial's checkpoint is saved to `tune_runs/best_trial/`.

Recommended starting point for long training:
    epochs ≈ 80-100,
    lr ≈ 3e-4 (AdamW),
    weight_decay ≈ 1e-4,
    dropout ≈ 0.1,
    batch_size = 8-12 depending on GPU memory.
"""

import argparse
from pathlib import Path

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

from utils.pannuke_dataset import PanNukeDataset
from models.attention_unet import AttentionUNet

# --------------------------------------------------
class LitSegModel(pl.LightningModule):
    def __init__(self, lr, weight_decay, dropout, optimiser, scheduler, n_classes: int = 6):
        super().__init__()
        self.save_hyperparameters()
        self.model = AttentionUNet(n_classes=n_classes, dropout=dropout)

    # ----------------
    def forward(self, x):
        return self.model(x)

    # ----------------
    def dice_coef(self, preds, targets, eps: float = 1e-6):
        preds = torch.argmax(preds, dim=1)
        dice_per_class = []
        for cls in range(1, 7):  # ignore background
            pred_c = (preds == cls).float()
            tgt_c = (targets == cls).float()
            inter = (pred_c * tgt_c).sum((1, 2))
            denom = pred_c.sum((1, 2)) + tgt_c.sum((1, 2)) + eps
            dice = (2 * inter / denom).mean()
            dice_per_class.append(dice)
        return torch.stack(dice_per_class).mean()

    # ----------------
    def common_step(self, batch, stage: str):
        imgs, masks = batch['image'], batch['mask']
        logits = self(imgs)
        ce = F.cross_entropy(logits, masks)
        dice = self.dice_coef(logits, masks)
        self.log(f'{stage}_loss', ce, prog_bar=True, on_epoch=True)
        self.log(f'{stage}_dice', dice, prog_bar=True, on_epoch=True)
        return ce

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self.common_step(batch, 'val')

    # ----------------
    def configure_optimizers(self):
        hp = self.hparams
        if hp.optimiser == 'adamw':
            opt = torch.optim.AdamW(self.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
        else:
            opt = torch.optim.SGD(self.parameters(), lr=hp.lr, momentum=0.9, weight_decay=hp.weight_decay)

        if hp.scheduler == 'cosine':
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp.max_epochs)
            return {'optimizer': opt, 'lr_scheduler': sched}
        elif hp.scheduler == 'plateau':
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
            return {
                'optimizer': opt,
                'lr_scheduler': {'scheduler': sched, 'monitor': 'val_loss'}
            }
        else:
            return opt

# --------------------------------------------------

def get_dataloaders(batch_size, num_workers=4):
    dataset = PanNukeDataset(root='Dataset', parts=['Part 1', 'Part 2', 'Part 3'],
                             image_dirname='Images', mask_dirname='Masks', img_size=(256, 256))
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    ds_train, ds_val = torch.utils.data.random_split(dataset, [train_size, val_size])

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader_train, loader_val

# --------------------------------------------------

def objective(trial: optuna.Trial):
    # hyper-parameter search space
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 12, 16])
    optimiser = trial.suggest_categorical('optimiser', ['adamw', 'sgd'])
    scheduler = trial.suggest_categorical('scheduler', ['none', 'cosine', 'plateau'])

    train_loader, val_loader = get_dataloaders(batch_size)

    model = LitSegModel(lr=lr, weight_decay=weight_decay, dropout=dropout,
                        optimiser=optimiser, scheduler=scheduler)

    max_epochs = 30  # shorter per-trial; full run recommendation at the top docstring
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_dice')
    ckpt_cb = ModelCheckpoint(dirpath=f'tune_runs/{trial.number}', monitor='val_dice', mode='max', save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices='auto',
        precision=16 if torch.cuda.is_available() else 32,
        callbacks=[pruning_callback, ckpt_cb, LearningRateMonitor('epoch'), EarlyStopping('val_dice', mode='max', patience=8)],
        logger=False,
    )

    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics['val_dice'].item()

# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-trials', type=int, default=50)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction='maximize', pruner=pruner, study_name='attention_unet')
    study.optimize(objective, n_trials=args.max_trials, timeout=None)

    print('Best trial:')
    best = study.best_trial
    print(f'  Value: {best.value}')
    print('  Params:')
    for k, v in best.params.items():
        print(f'    {k}: {v}')

    # Save study
    Path('tune_runs').mkdir(exist_ok=True)
    optuna.study.save_study_pickle(study, 'tune_runs/attention_unet_optuna.pkl')

if __name__ == '__main__':
    main() 