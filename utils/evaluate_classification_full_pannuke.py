#!/usr/bin/env python3
"""
Evaluate the trained nuclei classifier on the entire nuclei dataset (full run).

Reports (over the whole dataset):
- Overall accuracy
- Weighted F1, Macro F1
- Weighted precision, Weighted recall
- AUROC (macro)
- Per-class precision, recall, F1, support
- Confusion matrix (counts)

Saves metrics to JSON if --output is provided.
"""

import sys
import os
import platform
from pathlib import Path
import argparse
import json
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Ensure project root is on sys.path (this file lives under utils/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.nuclei_dataset import load_nuclei_dataset, NucleiClassificationDataset, get_classification_augmentations
from training.train_classifier import StateOfTheArtClassifierLightning


# Fix for Windows multiprocessing
if platform.system() == 'Windows':
    torch.multiprocessing.set_start_method('spawn', force=True)


def _get_device(device_arg: str) -> torch.device:
    if device_arg == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available. Falling back to CPU.")
        return torch.device('cpu')
    return torch.device(device_arg)


@torch.no_grad()
def evaluate_full_classifier(
    checkpoint_path: str,
    dataset_path: str,
    batch_size: int = 64,
    device_str: str = 'cuda',
    num_workers: int = 0,
    target_size: int = 224,
    output_json: str | None = None,
) -> Dict[str, Any]:
    device = _get_device(device_str)

    # Load nuclei instances
    nuclei_instances = load_nuclei_dataset(dataset_path)

    # Create dataset (no training augs; normalization + resize only)
    dataset = NucleiClassificationDataset(
        nuclei_instances=nuclei_instances,
        augmentations=get_classification_augmentations(training=False),
        normalize=True,
        target_size=target_size
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=False
    )

    # Load Lightning module and extract underlying model; prevent resume side-effects
    lightning_model = StateOfTheArtClassifierLightning.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        resume_path=None
    )
    lightning_model.eval()
    model = lightning_model.model
    model.eval()
    model.to(device)

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        outputs = model(images)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.stack(all_probs, axis=0) if len(all_probs) > 0 else None

    # Overall metrics
    accuracy = float((y_true == y_pred).mean())

    # Classification report for weighted/macro and per-class metrics
    class_names = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    weighted_f1 = float(report['weighted avg']['f1-score'])
    macro_f1 = float(report['macro avg']['f1-score'])
    weighted_precision = float(report['weighted avg']['precision'])
    weighted_recall = float(report['weighted avg']['recall'])

    # AUROC (macro, one-vs-rest) – requires probabilities
    try:
        auroc_macro = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')) if y_prob is not None else None
    except Exception:
        auroc_macro = None

    # Per-class metrics and supports
    per_class: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(class_names):
        key = name if name in report else str(i)
        cls_rep = report.get(key, {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0})
        per_class[name] = {
            'precision': float(cls_rep.get('precision', 0.0)),
            'recall': float(cls_rep.get('recall', 0.0)),
            'f1': float(cls_rep.get('f1-score', 0.0)),
            'support': int(cls_rep.get('support', 0)),
        }

    # Confusion matrix (counts)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_list = cm.astype(int).tolist()

    results: Dict[str, Any] = {
        'dataset_size': int(len(dataset)),
        'overall_accuracy': accuracy,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'auroc_macro': auroc_macro,
        'per_class': per_class,
        'class_order': class_names,
        'confusion_matrix': cm_list,
    }

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Saved metrics to {out_path}")

    # Pretty print summary
    print("\n📈 Full Nuclei Classification Evaluation")
    print("=" * 60)
    print(f"Instances: {len(dataset)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 (w):   {weighted_f1:.4f} | F1 (macro): {macro_f1:.4f}")
    if auroc_macro is not None:
        print(f"AUROC (macro): {auroc_macro:.4f}")
    print("-" * 60)
    for name, m in per_class.items():
        print(f"{name:12s} | support={m['support']:7d} | P={m['precision']:.4f} | R={m['recall']:.4f} | F1={m['f1']:.4f}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate nuclei classifier on full dataset')
    parser.add_argument('--checkpoint', type=str,
                        default='lightning_logs/classifier/classifier_efficientnet_b3_20250727_002713/version_0/checkpoints/classifier-epoch=19-val_f1=0.806.ckpt',
                        help='Path to classifier checkpoint (.ckpt)')
    parser.add_argument('--dataset', type=str, default='nuclei_dataset.pkl', help='Path to nuclei dataset (.pkl)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--target_size', type=int, default=224)
    parser.add_argument('--output', type=str, default='metrics_classifier.json')

    args = parser.parse_args()

    # On Windows default to num_workers=0
    if platform.system() == 'Windows' and args.num_workers != 0:
        print("ℹ️  For Windows, forcing num_workers=0 for stability.")
        args.num_workers = 0

    evaluate_full_classifier(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        device_str=args.device,
        num_workers=args.num_workers,
        target_size=args.target_size,
        output_json=args.output,
    )


if __name__ == '__main__':
    main()


