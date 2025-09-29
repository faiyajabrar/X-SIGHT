#!/usr/bin/env python3
"""
Evaluate the two-stage pipeline (segmentation + classification) over the entire PanNuke dataset.

Outputs dataset-wide metrics comparing the final pipeline type map against ground truth:
- Overall pixel accuracy (excluding background)
- Weighted F1, Macro F1
- Per-class precision, recall, F1, Dice, IoU, supports
- Confusion matrix (counts)
- Nuclei statistics: total detected and predicted class counts

Saves metrics to JSON if --output is provided.
"""

import sys
import os
import platform
from pathlib import Path
import argparse
import json
from typing import Dict, Any, Tuple

import numpy as np
import torch
import cv2

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.pannuke_dataset import PanNukeDataset
from two_stage_pipeline import TwoStageNucleiPipeline


if platform.system() == 'Windows':
    torch.multiprocessing.set_start_method('spawn', force=True)


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d > 0 else 0.0


def _compute_counts_from_masks(gt_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = gt_mask.reshape(-1)
    pr = pred_mask.reshape(-1)
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    idx = gt * num_classes + pr
    binc = np.bincount(idx, minlength=num_classes * num_classes)
    conf[:] = binc.reshape(num_classes, num_classes)
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    return tp, fp, fn


def _build_pipeline_type_mask(image_shape: Tuple[int, int], extracted_nuclei: list, classifications: dict) -> np.ndarray:
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for nucleus_info in extracted_nuclei:
        nucleus_id = nucleus_info['nucleus_id']
        if nucleus_id not in classifications:
            continue
        classification = classifications[nucleus_id]
        pred_class_0_4 = int(classification['predicted_class'])
        pred_class_1_5 = pred_class_0_4 + 1
        centroid = nucleus_info.get('centroid', (H // 2, W // 2))
        area = int(nucleus_info.get('area', 100))
        cy, cx = int(centroid[0]), int(centroid[1])
        radius = max(3, int(np.sqrt(max(1, area) / np.pi)))
        cy = max(radius, min(H - radius, cy))
        cx = max(radius, min(W - radius, cx))
        y, x = np.ogrid[:H, :W]
        circle = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        mask[circle] = pred_class_1_5
    return mask


def evaluate_pipeline_full(
    dataset_root: str = 'Dataset',
    segmentation_model: str = 'lightning_logs/segmentation/version_1/checkpoints/advanced-epoch=112-val_dice=0.656.ckpt',
    classifier_model: str = 'lightning_logs/classifier/classifier_efficientnet_b3_20250727_002713/version_0/checkpoints/classifier-epoch=19-val_f1=0.806.ckpt',
    device: str = 'cuda',
    base_size: int = 256,
    output_json: str | None = None,
) -> Dict[str, Any]:
    # Init dataset (no transforms; we'll read raw arrays/paths)
    ds = PanNukeDataset(root=dataset_root, augmentations=None, validate_dataset=False, base_size=base_size)

    # Init pipeline
    pipeline = TwoStageNucleiPipeline(
        segmentation_model_path=segmentation_model,
        classifier_model_path=classifier_model,
        device=device,
        enable_gradcam=False,
        enable_shap=False
    )

    # Accumulators
    class_names = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
    num_classes = len(class_names)
    conf_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    nuclei_total = 0
    predicted_nuclei_counts = {name: 0 for name in class_names[1:]}

    # Iterate over all samples, reconstruct raw image and GT mask
    for idx in range(len(ds)):
        # Load raw image/mask depending on storage mode
        if ds.storage_mode == 'files':
            img_path = ds.image_paths[idx]
            msk_path = ds.mask_paths[idx]
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mask6 = np.load(msk_path)
            if mask6.ndim == 2:
                oh = np.zeros((*mask6.shape, 6), dtype=np.uint8)
                for c in range(1, 7):
                    oh[:, :, c - 1] = (mask6 == c).astype(np.uint8)
                mask6 = oh
        else:
            part_idx, local_idx = ds._index_map[idx]
            image = ds._parts_images[part_idx][local_idx]
            mask6 = ds._parts_masks[part_idx][local_idx]

        # Ensure types
        image = np.ascontiguousarray(image).astype(np.uint8)
        mask6 = np.ascontiguousarray(mask6)

        # GT type mask using dataset conversion
        gt_type_mask = ds._mask_to_class_map(mask6)

        # Run pipeline on the image array
        results = pipeline.analyze_image_array(image=image, save_results=False, visualize=False)

        seg_pred = results['segmentation']['prediction_mask'].cpu().numpy()
        extracted_nuclei = results['extracted_nuclei']
        classifications = results['classifications']

        # Compose final pipeline type mask from classifications (circles at centroids)
        pipeline_type_mask = _build_pipeline_type_mask(seg_pred.shape, extracted_nuclei, classifications)

        # Accumulate confusion counts
        tp, fp, fn = _compute_counts_from_masks(gt_type_mask, pipeline_type_mask, num_classes=num_classes)
        # Build confusion via tp/fp/fn is insufficient; compute full confusion directly
        gt_flat = gt_type_mask.reshape(-1)
        pr_flat = pipeline_type_mask.reshape(-1)
        binc = np.bincount(gt_flat * num_classes + pr_flat, minlength=num_classes * num_classes)
        conf_total += binc.reshape(num_classes, num_classes)

        # Nuclei stats
        nuclei_total += len(extracted_nuclei)
        for nucleus_id, cls in classifications.items():
            name = class_names[cls['predicted_class'] + 1]
            predicted_nuclei_counts[name] += 1

    # Derive metrics from confusion matrix
    tp = np.diag(conf_total).astype(np.float64)
    fp = conf_total.sum(axis=0).astype(np.float64) - tp
    fn = conf_total.sum(axis=1).astype(np.float64) - tp

    # Exclude background for summary metrics
    idxs = list(range(1, num_classes))
    supports = conf_total.sum(axis=1).astype(np.float64)

    precision = np.array([_safe_div(tp[i], tp[i] + fp[i]) for i in range(num_classes)])
    recall = np.array([_safe_div(tp[i], tp[i] + fn[i]) for i in range(num_classes)])
    f1 = np.array([_safe_div(2 * precision[i] * recall[i], precision[i] + recall[i]) for i in range(num_classes)])
    dice = np.array([_safe_div(2 * tp[i], 2 * tp[i] + fp[i] + fn[i]) for i in range(num_classes)])
    iou = np.array([_safe_div(tp[i], tp[i] + fp[i] + fn[i]) for i in range(num_classes)])

    non_bg_mask = supports[0] < 0  # placeholder unused
    total_non_bg = supports[idxs].sum()
    overall_accuracy_excl_bg = _safe_div(sum(conf_total[i, i] for i in idxs), total_non_bg)
    weighted_f1 = _safe_div((f1[idxs] * supports[idxs]).sum(), supports[idxs].sum())
    macro_f1 = float(f1[idxs].mean()) if len(idxs) > 0 else 0.0

    # Assemble per-class
    per_class = {}
    class_names = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
    for i, name in enumerate(class_names):
        per_class[name] = {
            'support': int(supports[i]),
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'dice': float(dice[i]),
            'iou': float(iou[i]),
        }

    results = {
        'dataset_images': len(ds),
        'overall_accuracy_excluding_background': float(overall_accuracy_excl_bg),
        'weighted_f1_excluding_background': float(weighted_f1),
        'macro_f1_excluding_background': float(macro_f1),
        'per_class': per_class,
        'class_order': class_names,
        'confusion_matrix': conf_total.astype(int).tolist(),
        'nuclei_detected_total': int(nuclei_total),
        'predicted_nuclei_class_counts': predicted_nuclei_counts,
        'models': {
            'segmentation': segmentation_model,
            'classifier': classifier_model
        }
    }

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Saved metrics to {out_path}")

    # Pretty print
    print("\n🧬 Two-Stage Pipeline Full-Dataset Evaluation")
    print("=" * 60)
    print(f"Images: {len(ds)} | Nuclei detected: {nuclei_total}")
    print(f"Accuracy (excl bg): {results['overall_accuracy_excluding_background']:.4f}")
    print(f"F1 (weighted, excl bg): {results['weighted_f1_excluding_background']:.4f} | F1 (macro, excl bg): {results['macro_f1_excluding_background']:.4f}")
    print("-" * 60)
    for name in class_names[1:]:
        m = per_class[name]
        print(f"{name:12s} | sup={m['support']:7d} | P={m['precision']:.4f} | R={m['recall']:.4f} | F1={m['f1']:.4f} | Dice={m['dice']:.4f} | IoU={m['iou']:.4f}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Two-Stage Pipeline on full PanNuke dataset')
    parser.add_argument('--dataset_root', type=str, default='Dataset')
    parser.add_argument('--segmentation_model', type=str, default='lightning_logs/segmentation/version_1/checkpoints/advanced-epoch=112-val_dice=0.656.ckpt')
    parser.add_argument('--classifier_model', type=str, default='lightning_logs/classifier/classifier_efficientnet_b3_20250727_002713/version_0/checkpoints/classifier-epoch=19-val_f1=0.806.ckpt')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--base_size', type=int, default=256)
    parser.add_argument('--output', type=str, default='metrics_pipeline.json')

    args = parser.parse_args()

    evaluate_pipeline_full(
        dataset_root=args.dataset_root,
        segmentation_model=args.segmentation_model,
        classifier_model=args.classifier_model,
        device=args.device,
        base_size=args.base_size,
        output_json=args.output,
    )


if __name__ == '__main__':
    main()


