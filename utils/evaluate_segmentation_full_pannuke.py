#!/usr/bin/env python3
"""
Evaluate a trained segmentation model on the entire PanNuke dataset.

Reports:
- Overall pixel accuracy
- Frequency-weighted Dice (same weighting as training)
- Per-class Dice, IoU, Precision, Recall, F1
- Macro/weighted metrics (excluding background)

Saves metrics to JSON if --output is provided.
"""

import sys
import os
import platform
from pathlib import Path
import argparse
import json
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Ensure project root is on sys.path (this file lives under utils/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Local imports
from utils.pannuke_dataset import PanNukeDataset
from training.train_segmentation import AdvancedAttentionModel


# Fix for Windows multiprocessing
if platform.system() == 'Windows':
    torch.multiprocessing.set_start_method('spawn', force=True)


def _get_device(device_arg: str) -> torch.device:
    if device_arg == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available. Falling back to CPU.")
        return torch.device('cpu')
    return torch.device(device_arg)


def _compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """Compute confusion matrix for a batch and return counts [C, C].

    Rows = ground truth, Columns = predictions.
    """
    with torch.no_grad():
        preds = preds.view(-1).to(torch.int64)
        targets = targets.view(-1).to(torch.int64)
        k = targets * num_classes + preds
        binc = torch.bincount(k, minlength=num_classes * num_classes)
        conf = binc.view(num_classes, num_classes)
    return conf


def _dice_from_counts(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


def _iou_from_counts(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (tp + eps) / (tp + fp + fn + eps)


def _safe_div(n: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return torch.where(d > 0, n / d, torch.zeros_like(n))


def evaluate_full(
    checkpoint_path: str,
    dataset_root: str = 'Dataset',
    batch_size: int = 8,
    device_str: str = 'cuda',
    num_workers: int = 0,
    base_size: int = 256,
    output_json: str | None = None,
) -> Dict[str, Any]:
    device = _get_device(device_str)
    num_classes = 6
    class_names = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']

    # Load dataset and dataloader
    dataset = PanNukeDataset(root=dataset_root, augmentations=None, validate_dataset=False, base_size=base_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=False
    )

    # Load model (LightningModule) and move to device
    # Override resume_path to avoid loading training state during evaluation
    model = AdvancedAttentionModel.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        resume_path=None
    )
    model.eval()
    model.to(device)

    # Accumulators
    total_confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_pixels = 0
    correct_pixels = 0

    # Iterate over full dataset
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            # Accuracy (pixel-wise)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

            # Confusion matrix
            total_confusion += _compute_confusion_matrix(preds.cpu(), masks.cpu(), num_classes)

    # Derive metrics from confusion matrix
    conf = total_confusion.to(torch.float64)
    support_gt = conf.sum(dim=1)  # per-class ground truth pixels
    pred_count = conf.sum(dim=0)  # per-class predicted pixels
    tp = torch.diag(conf)
    fp = pred_count - tp
    fn = support_gt - tp

    # Per-class metrics
    dice_per_class = _dice_from_counts(tp, fp, fn)
    iou_per_class = _iou_from_counts(tp, fp, fn)
    precision_per_class = _safe_div(tp, tp + fp)
    recall_per_class = _safe_div(tp, tp + fn)
    f1_per_class = _safe_div(2 * precision_per_class * recall_per_class, precision_per_class + recall_per_class)

    # Overall metrics
    overall_accuracy = correct_pixels / max(1, total_pixels)

    # Frequency-weighted Dice (match training weighting)
    pixel_freq = torch.tensor(
        [0.832818975, 0.0866185198, 0.0177438743, 0.0373645720, 0.000691303678, 0.0247627551],
        dtype=torch.float64
    )
    class_weights = torch.sqrt(1.0 / (pixel_freq + 1e-6))
    class_weights = torch.clamp(class_weights, max=10.0)
    class_weights = class_weights / class_weights.mean()
    fw_dice = (dice_per_class * class_weights).sum() / class_weights.sum()

    # Macro/weighted metrics (exclude background for classification-style summaries)
    non_bg = torch.arange(1, num_classes)
    support_non_bg = support_gt[non_bg]
    total_support_non_bg = support_non_bg.sum().item()

    macro_f1_non_bg = f1_per_class[non_bg].mean().item() if len(non_bg) > 0 else 0.0
    weighted_f1_non_bg = (
        (f1_per_class[non_bg] * support_non_bg).sum().item() / total_support_non_bg
        if total_support_non_bg > 0 else 0.0
    )
    mean_iou_including_bg = iou_per_class.mean().item()
    mean_iou_excluding_bg = iou_per_class[non_bg].mean().item()
    mean_dice_including_bg = dice_per_class.mean().item()
    mean_dice_excluding_bg = dice_per_class[non_bg].mean().item()

    # Assemble report
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            'support': int(support_gt[i].item()),
            'precision': float(precision_per_class[i].item()),
            'recall': float(recall_per_class[i].item()),
            'f1': float(f1_per_class[i].item()),
            'dice': float(dice_per_class[i].item()),
            'iou': float(iou_per_class[i].item()),
        }

    results: Dict[str, Any] = {
        'dataset_size': len(dataset),
        'total_pixels': int(total_pixels),
        'overall_accuracy': float(overall_accuracy),
        'frequency_weighted_dice': float(fw_dice.item()),
        'mean_dice_including_bg': float(mean_dice_including_bg),
        'mean_dice_excluding_bg': float(mean_dice_excluding_bg),
        'mean_iou_including_bg': float(mean_iou_including_bg),
        'mean_iou_excluding_bg': float(mean_iou_excluding_bg),
        'macro_f1_excluding_bg': float(macro_f1_non_bg),
        'weighted_f1_excluding_bg': float(weighted_f1_non_bg),
        'per_class': per_class,
        'class_order': class_names,
    }

    # Optional JSON dump
    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Saved metrics to {out_path}")

    # Pretty print summary
    print("\n📈 Full PanNuke Evaluation")
    print("=" * 60)
    print(f"Samples: {len(dataset)} | Pixels: {results['total_pixels']:,}")
    print(f"Accuracy: {results['overall_accuracy']:.4f}")
    print(f"FW Dice:  {results['frequency_weighted_dice']:.4f}")
    print(f"mDice (incl bg): {results['mean_dice_including_bg']:.4f} | (excl bg): {results['mean_dice_excluding_bg']:.4f}")
    print(f"mIoU  (incl bg): {results['mean_iou_including_bg']:.4f} | (excl bg): {results['mean_iou_excluding_bg']:.4f}")
    print(f"Macro-F1 (excl bg): {results['macro_f1_excluding_bg']:.4f} | Weighted-F1 (excl bg): {results['weighted_f1_excluding_bg']:.4f}")
    print("-" * 60)
    for i, name in enumerate(class_names):
        pc = results['per_class'][name]
        print(
            f"{i}:{name:12s} | support={pc['support']:7d} | dice={pc['dice']:.4f} | iou={pc['iou']:.4f} | "
            f"prec={pc['precision']:.4f} | rec={pc['recall']:.4f} | f1={pc['f1']:.4f}"
        )
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model on full PanNuke dataset')
    parser.add_argument('--checkpoint', type=str,
                        default='lightning_logs/segmentation/version_1/checkpoints/advanced-epoch=112-val_dice=0.656.ckpt',
                        help='Path to Lightning checkpoint (.ckpt)')
    parser.add_argument('--dataset_root', type=str, default='Dataset', help='Path to PanNuke dataset root')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--base_size', type=int, default=256, help='Resize size used by dataset')
    parser.add_argument('--output', type=str, default='metrics.json', help='Path to save JSON metrics')

    args = parser.parse_args()

    # On Windows default to num_workers=0
    if platform.system() == 'Windows' and args.num_workers != 0:
        print("ℹ️  For Windows, forcing num_workers=0 for stability.")
        args.num_workers = 0

    evaluate_full(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        device_str=args.device,
        num_workers=args.num_workers,
        base_size=args.base_size,
        output_json=args.output,
    )


if __name__ == '__main__':
    main()


