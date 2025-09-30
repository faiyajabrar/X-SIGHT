#!/usr/bin/env python3
"""
Visualize a single nucleus instance from a nuclei dataset with its prediction,
and export two composite images:
- Original + Grad-CAM + Prediction
- Original + SHAP + Prediction

Example (Windows PowerShell):
  python utils/visualize_single_nucleus.py \
    --checkpoint lightning_logs/classifier/your_run/version_0/checkpoints/classifier-epoch=XX.ckpt \
    --dataset nuclei_dataset.pkl \
    --index 0 \
    --out_dir out --prefix nucleus_0
"""

import sys
import os
import platform
from pathlib import Path
import argparse
from typing import Optional

import numpy as np
import torch
import cv2

# Ensure project root on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nuclei_dataset import load_nuclei_dataset, NucleiClassificationDataset  # noqa: E402
from utils.evaluate_classifier import NucleusClassifierEvaluator  # noqa: E402


def denormalize_to_uint8_imagenet(img_chw: np.ndarray) -> np.ndarray:
    """Denormalize an ImageNet-normalized CHW image to HWC uint8 RGB."""
    # Expect CHW. If already HWC, transpose back after channel ops
    chw = img_chw
    if chw.ndim == 3 and chw.shape[0] in (1, 3):
        pass
    elif chw.ndim == 3 and chw.shape[-1] in (1, 3):
        chw = np.transpose(chw, (2, 0, 1))
    else:
        raise ValueError("Expected 3D image with channel dimension.")
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    img = (chw.astype(np.float32) * std + mean)
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def visualize_single(
    checkpoint: str,
    dataset_path: str,
    index: int = 0,
    device: str = "cuda",
    out_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    enable_gradcam: bool = True,
    enable_shap: bool = True,
    shap_bg: int = 8,
    shap_nsamples: int = 20,
    shap_seed: int = 42,
    shap_use_validation_split: bool = False,
    shap_split_path: Optional[str] = 'lightning_logs/classifier/dataset_split.json',
    shap_bg_mode: str = "dataset_random",  # 'dataset_random' | 'same_sample' | 'val_split'
    gradcam_layer_idx: int = -2,
    gradcam_blur_ksize: int = 3,
    gradcam_blur_sigma: float = 0.0,
    show: bool = False,
    shap_overlay_mode: str = "masked",
    shap_top_percent: float = 0.15,
    shap_alpha: float = 0.6,
):
    # Load nuclei instances metadata
    nuclei_instances = load_nuclei_dataset(dataset_path)
    if len(nuclei_instances) == 0:
        raise RuntimeError("Dataset is empty.")
    index = int(index)
    if not (0 <= index < len(nuclei_instances)):
        raise IndexError(f"index {index} out of range [0,{len(nuclei_instances)-1}]")

    # Create dataset without augs (normalize + ToTensor only)
    ds = NucleiClassificationDataset(
        nuclei_instances=nuclei_instances,
        augmentations=None,
        normalize=True,
        target_size=224,
    )
    sample = ds[index]
    image_t = sample["image"]  # CHW tensor, normalized
    true_label_0based = int(sample["label"].item())

    # Initialize evaluator and configure explainability
    evaluator = NucleusClassifierEvaluator(model_path=checkpoint, device=device)
    evaluator._gc_layer_idx = gradcam_layer_idx
    evaluator._gc_blur_ksize = gradcam_blur_ksize
    evaluator._gc_blur_sigma = gradcam_blur_sigma
    evaluator._register_gradcam_hooks()

    # Predict
    logits, probs_t, pred_t = evaluator.predict_sample(image_t)
    probs = probs_t.detach().cpu().numpy()
    pred = int(pred_t.item())

    # Grad-CAM
    gradcam_overlay_rgb = None
    if enable_gradcam:
        try:
            _, gradcam_overlay_rgb = evaluator.generate_gradcam(image_t, pred)
        except Exception as e:
            print(f"[WARN] Grad-CAM failed: {e}")
            gradcam_overlay_rgb = None

    # SHAP: create small background batch and compute overlay
    shap_overlay_rgb = None
    shap_heat_uint8 = None
    if enable_shap:
        try:
            # Build background from random indices (or as many as available)
            total = len(ds)
            bg_num = min(max(1, int(shap_bg)), total)
            # Align with evaluator: use numpy RNG and optional validation split
            np.random.seed(int(shap_seed))
            if shap_bg_mode == "same_sample":
                # Prefer nuclei from the same original sample/image as the target instance
                base_inst = nuclei_instances[index]
                sample_idx_key = base_inst.get('sample_idx', base_inst.get('global_instance_id', None))
                pool = []
                if sample_idx_key is not None:
                    for i, nuc in enumerate(nuclei_instances):
                        if i == index:
                            continue
                        sid = nuc.get('sample_idx', nuc.get('global_instance_id', None))
                        if sid == sample_idx_key:
                            pool.append(i)
                pool = np.array(pool, dtype=int)
                if len(pool) >= bg_num:
                    bg_indices = np.random.choice(pool, size=bg_num, replace=False)
                elif len(pool) > 0:
                    # take all from same sample; if not enough, fill randomly
                    extra = np.random.choice([i for i in range(total) if i not in pool and i != index], size=bg_num - len(pool), replace=False)
                    bg_indices = np.concatenate([pool, extra])
                else:
                    bg_indices = np.random.choice(total, size=bg_num, replace=False)
            elif shap_use_validation_split and shap_split_path and os.path.exists(shap_split_path):
                nuclei_instances_all = load_nuclei_dataset(dataset_path)
                import json
                with open(shap_split_path, 'r') as f:
                    split_data = json.load(f)
                val_indices = split_data.get('val_indices', [])
                # Map sample indices to nucleus indices
                sample_groups = {}
                for i, nucleus in enumerate(nuclei_instances_all):
                    sample_idx = nucleus.get('sample_idx', nucleus.get('global_instance_id', i // 10))
                    sample_groups.setdefault(sample_idx, []).append(i)
                pool = []
                for sidx in val_indices:
                    pool.extend(sample_groups.get(sidx, []))
                pool = np.array([p for p in pool if p < total], dtype=int)
                if len(pool) >= bg_num:
                    bg_indices = np.random.choice(pool, size=bg_num, replace=False)
                else:
                    bg_indices = np.random.choice(total, size=bg_num, replace=False)
            else:
                bg_indices = np.random.choice(total, size=bg_num, replace=False)
            bg_tensors = [ds[int(j)]["image"] for j in bg_indices]
            background = torch.stack(bg_tensors)
            explainer = evaluator._create_shap_explainer(background)
            if explainer is not None:
                shap_overlay_rgb, _, shap_heat_uint8 = evaluator.generate_shap_overlay(
                    image=image_t,
                    pred_class=pred,
                    explainer=explainer,
                    nsamples=int(shap_nsamples),
                )
            else:
                print("[INFO] SHAP not available or failed to initialize. Skipping.")
        except Exception as e:
            print(f"[WARN] SHAP failed: {e}")
            shap_overlay_rgb = None
            shap_heat_uint8 = None

    # Prepare outputs
    if out_dir is None:
        out_dir = "out"
    if prefix is None:
        prefix = f"nucleus_{index}"
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Prepare base image
    base_rgb = denormalize_to_uint8_imagenet(image_t.detach().cpu().numpy())

    # Optionally convert SHAP overlay to masked-top-percent style to avoid full wash
    if enable_shap and shap_overlay_rgb is not None and shap_heat_uint8 is not None:
        try:
            mode = (shap_overlay_mode or "masked").lower()
            if mode == "masked":
                heat = shap_heat_uint8.astype(np.float32) / 255.0
                top = float(max(0.0, min(1.0, shap_top_percent)))
                thr = float(np.quantile(heat, 1.0 - top)) if np.any(heat > 0) else 1.0
                mask = (heat >= thr).astype(np.float32)
                mask3 = np.repeat(mask[:, :, None], 3, axis=2)
                heat_rgb = cv2.cvtColor(cv2.applyColorMap(shap_heat_uint8, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
                a = float(max(0.0, min(1.0, shap_alpha)))
                shaped = (base_rgb.astype(np.float32) * (1.0 - a * mask3) + heat_rgb.astype(np.float32) * (a * mask3))
                shap_overlay_rgb = shaped.clip(0, 255).astype(np.uint8)
        except Exception:
            pass

    # Helper to save composite figure
    class_names = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
    pred_name = class_names[pred]
    true_name = class_names[true_label_0based]
    conf = float(probs[pred])

    def save_combo(overlay_rgb, overlay_title, filename):
        if overlay_rgb is None:
            return None
        try:
            import matplotlib.pyplot as plt  # Lazy import
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(base_rgb); axes[0].set_title("Original"); axes[0].axis('off')
            axes[1].imshow(overlay_rgb); axes[1].set_title(overlay_title); axes[1].axis('off')
            bars = axes[2].bar(range(len(class_names)), probs,
                               color=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1)], alpha=0.85)
            axes[2].set_xticks(range(len(class_names)))
            axes[2].set_xticklabels([n[:4] for n in class_names])
            axes[2].set_ylim(0, 1)
            axes[2].set_title(f"Pred: {pred_name} (p={conf:.2f})\nTrue: {true_name}")
            for j, b in enumerate(bars):
                if probs[j] > 0.06:
                    axes[2].text(b.get_x() + b.get_width()/2, probs[j] + 0.02, f"{probs[j]:.2f}",
                                 ha='center', va='bottom', fontsize=8)
            bars[pred].set_edgecolor('red'); bars[pred].set_linewidth(2)
            bars[true_label_0based].set_edgecolor('green'); bars[true_label_0based].set_linewidth(2)
            plt.tight_layout()
            out_path = out_dir_path / filename
            fig.savefig(str(out_path), dpi=220, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {overlay_title} composite to: {out_path}")
            return out_path
        except Exception:
            # Fallback: save overlay only
            out_path = out_dir_path / (Path(filename).stem + "_overlay.png")
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
            print(f"Saved overlay to: {out_path}")
            return out_path

    gradcam_combo_path = save_combo(gradcam_overlay_rgb, "Grad-CAM", f"{prefix}_gradcam_combo.png")
    shap_combo_path = save_combo(shap_overlay_rgb, "SHAP", f"{prefix}_shap_combo.png")

    # Optional preview
    if show:
        try:
            import matplotlib.pyplot as plt  # Lazy import
            previews = []
            if gradcam_combo_path is not None:
                previews.append((cv2.cvtColor(cv2.imread(str(gradcam_combo_path)), cv2.COLOR_BGR2RGB), 'Grad-CAM Combo'))
            if shap_combo_path is not None:
                previews.append((cv2.cvtColor(cv2.imread(str(shap_combo_path)), cv2.COLOR_BGR2RGB), 'SHAP Combo'))
            if previews:
                fig, axes = plt.subplots(1, len(previews), figsize=(12, 5))
                if len(previews) == 1:
                    axes = [axes]
                for i, (img, title) in enumerate(previews):
                    axes[i].imshow(img); axes[i].set_title(title); axes[i].axis('off')
                plt.tight_layout(); plt.show()
        except Exception:
            pass

    return gradcam_combo_path, shap_combo_path


def main():
    if platform.system() == 'Windows':
        torch.multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Visualize a single nucleus with prediction, Grad-CAM, and SHAP")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default='lightning_logs/classifier/classifier_efficientnet_b3_20250727_002713/version_0/checkpoints/classifier-epoch=19-val_f1=0.806.ckpt',
        help="Path to classifier checkpoint .ckpt",
    )
    parser.add_argument("--dataset", type=str, default="nuclei_dataset.pkl", help="Path to nuclei dataset .pkl")
    parser.add_argument("--index", type=int, default=0, help="Index of nucleus in dataset")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory to save images")
    parser.add_argument("--prefix", type=str, default=None, help="Filename prefix (default nucleus_{index})")

    parser.add_argument("--enable_gradcam", action="store_true", default=True, help="Enable Grad-CAM overlay")
    parser.add_argument("--disable_gradcam", dest="enable_gradcam", action="store_false", help="Disable Grad-CAM")
    parser.add_argument("--enable_shap", action="store_true", default=True, help="Enable SHAP overlay")
    parser.add_argument("--disable_shap", dest="enable_shap", action="store_false", help="Disable SHAP")
    parser.add_argument("--shap_bg", type=int, default=8, help="Background batch size for SHAP explainer")
    parser.add_argument("--shap_nsamples", type=int, default=20, help="Number of GradientExplainer samples")
    parser.add_argument("--shap_seed", type=int, default=42, help="Random seed for SHAP background selection")
    parser.add_argument("--shap_use_validation_split", action="store_true", help="Use validation split for SHAP background, if split file exists")
    parser.add_argument("--shap_split_path", type=str, default='lightning_logs/classifier/dataset_split.json', help="Path to dataset split JSON")
    parser.add_argument("--shap_bg_mode", type=str, default="dataset_random", choices=["dataset_random", "same_sample", "val_split"], help="How to select SHAP background pool")
    parser.add_argument("--gradcam_layer_idx", type=int, default=-2, help="timm feature layer index for Grad-CAM")
    parser.add_argument("--gradcam_blur_ksize", type=int, default=3, help="Gaussian blur kernel size for CAM (0 to disable)")
    parser.add_argument("--gradcam_blur_sigma", type=float, default=0.0, help="Gaussian blur sigma for CAM")
    parser.add_argument("--show", action="store_true", help="Display a quick preview figure")
    # SHAP overlay controls
    parser.add_argument("--shap_overlay_mode", type=str, default="masked", choices=["masked", "full"], help="How to render SHAP overlay in composites")
    parser.add_argument("--shap_top_percent", type=float, default=0.15, help="Top percentile of SHAP heat to overlay when masked mode is enabled (0..1)")
    parser.add_argument("--shap_alpha", type=float, default=0.6, help="Overlay alpha for SHAP masked regions")

    args = parser.parse_args()

    visualize_single(
        checkpoint=args.checkpoint,
        dataset_path=args.dataset,
        index=args.index,
        device=args.device,
        out_dir=args.out_dir,
        prefix=args.prefix,
        enable_gradcam=args.enable_gradcam,
        enable_shap=args.enable_shap,
        shap_bg=args.shap_bg,
        shap_nsamples=args.shap_nsamples,
        shap_seed=args.shap_seed,
        shap_use_validation_split=args.shap_use_validation_split,
        shap_split_path=args.shap_split_path,
        shap_bg_mode=args.shap_bg_mode,
        gradcam_layer_idx=args.gradcam_layer_idx,
        gradcam_blur_ksize=args.gradcam_blur_ksize,
        gradcam_blur_sigma=args.gradcam_blur_sigma,
        show=bool(args.show),
        shap_overlay_mode=args.shap_overlay_mode,
        shap_top_percent=args.shap_top_percent,
        shap_alpha=args.shap_alpha,
    )


if __name__ == "__main__":
    main()


