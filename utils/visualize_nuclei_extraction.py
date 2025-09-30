"""
Visualize nuclei extraction from a sample image.

Two usage modes:
1) Ground-truth mask mode (recommended if you have PanNuke 6-channel masks):
   python utils/visualize_nuclei_extraction.py --image path/to/img.png --mask6 path/to/mask.npy --out out/vis.png

2) Segmentation model mode (no ground-truth mask):
   python utils/visualize_nuclei_extraction.py --image path/to/img.png --seg_model path/to/seg.ckpt --out out/vis.png

This will produce a composite visualization containing:
- Original image
- Segmentation mask (if available)
- Overlay of detected nuclei with class-colored boxes
- Montage of extracted nucleus patches
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    # When executed from project root
    from utils.pannuke_dataset import CLAHETransform, ZScoreTransform, PanNukeDataset
    from utils.nuclei_extraction import (
        extract_nuclei_instances,
        extract_nuclei_from_prediction,
    )
except Exception:
    # When executed from inside utils directory
    from pannuke_dataset import CLAHETransform, ZScoreTransform, PanNukeDataset
    from nuclei_extraction import (
        extract_nuclei_instances,
        extract_nuclei_from_prediction,
    )


# Class color palette (align with pipeline): index 0..5
CLASS_COLORS = np.array(
    [
        [0, 0, 0],       # 0: Background
        [255, 0, 0],     # 1: Neoplastic
        [0, 255, 0],     # 2: Inflammatory
        [0, 0, 255],     # 3: Connective
        [255, 255, 0],   # 4: Dead
        [255, 0, 255],   # 5: Epithelial
    ],
    dtype=np.uint8,
)

CLASS_NAMES = [
    "Background",
    "Neoplastic",
    "Inflammatory",
    "Connective",
    "Dead",
    "Epithelial",
]


def load_image_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def colorize_segmentation(seg_mask: np.ndarray) -> np.ndarray:
    if seg_mask is None:
        return None
    if seg_mask.ndim == 3:
        return seg_mask
    return CLASS_COLORS[seg_mask]


def draw_overlay(image_rgb: np.ndarray, nuclei_instances: List[dict]) -> np.ndarray:
    vis = image_rgb.copy()
    for nucleus in nuclei_instances:
        y1, x1, y2, x2 = nucleus.get("bbox", (0, 0, 0, 0))
        class_id = int(nucleus.get("class_id", 0))
        class_id = max(0, min(class_id, len(CLASS_COLORS) - 1))
        color = CLASS_COLORS[class_id].tolist()
        cv2.rectangle(vis, (x1, y1), (x2, y2), color=color, thickness=2)
        label = CLASS_NAMES[class_id]
        if "confidence" in nucleus:
            try:
                label += f" {float(nucleus['confidence']):.2f}"
            except Exception:
                pass
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color=(255, 255, 255), thickness=-1)
        cv2.putText(vis, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def make_patch_montage(patches: List[np.ndarray], tile_size: int = 224, cols: int = 8, max_patches: int = 32) -> np.ndarray:
    if len(patches) == 0:
        return np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    n = min(len(patches), max_patches)
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols
    canvas = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)
    for i in range(n):
        r = i // cols
        c = i % cols
        patch = patches[i]
        if patch.shape[0] != tile_size or patch.shape[1] != tile_size:
            patch = cv2.resize(patch, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        canvas[r * tile_size : (r + 1) * tile_size, c * tile_size : (c + 1) * tile_size] = patch
    return canvas


def seg_preprocess_transforms(base_size: int = 256):
    return A.Compose(
        [
            A.Resize(base_size, base_size),
            CLAHETransform(),
            ZScoreTransform(),
            ToTensorV2(transpose_mask=True),
        ]
    )


def run_segmentation_extract(
    image_rgb: np.ndarray,
    seg_model_path: str,
    device: str = "cuda",
    min_area: int = 50,
    max_area: int = 5000,
    context_padding: int = 32,
) -> Tuple[np.ndarray, List[dict]]:
    device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    # Lazy import to avoid dependency when only using dataset mode
    AdvancedAttentionModel = None
    try:
        from training.train_segmentation import AdvancedAttentionModel as _AAM
        AdvancedAttentionModel = _AAM
    except Exception:
        try:
            from train_segmentation import AdvancedAttentionModel as _AAM
            AdvancedAttentionModel = _AAM
        except Exception as e:
            raise ModuleNotFoundError(
                "AdvancedAttentionModel not found. Provide a valid --seg_model only if the training module is available, or use --dataset_root/--mask6 modes."
            ) from e
    model = AdvancedAttentionModel.load_from_checkpoint(seg_model_path, map_location=device, resume_path=None)
    model = model.to(device)
    model.eval()

    transforms = seg_preprocess_transforms(base_size=256)
    t = transforms(image=image_rgb)
    img_t = t["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)  # [B, 6, H, W]
    logits = logits.squeeze(0)  # [6, H, W]
    pred_mask = torch.argmax(logits, dim=0).cpu().numpy()

    # For extraction use normalized tensor [C,H,W] as expected by helper
    img_chw = t["image"].cpu()  # [3, H, W], z-score normalized
    nuclei_instances = extract_nuclei_from_prediction(
        image=img_chw,
        prediction_logits=logits,
        min_area=min_area,
        max_area=max_area,
        context_padding=context_padding,
        target_size=224,
    )

    return pred_mask, nuclei_instances


def extract_from_mask6(
    image_rgb: np.ndarray,
    mask6_path: str,
    min_area: int = 10,
    max_area: int = 50000,
    context_padding: int = 32,
) -> List[dict]:
    mask6 = np.load(mask6_path)
    if mask6.ndim == 2:
        # Convert label map (uint16) to 6-channel one-hot
        oh = np.zeros((*mask6.shape, 6), dtype=np.uint8)
        for c in range(1, 7):
            oh[:, :, c - 1] = (mask6 == c).astype(np.uint8)
        mask6 = oh
    mask6 = np.ascontiguousarray(mask6)
    return extract_nuclei_instances(
        image=image_rgb,
        mask_6channel=mask6,
        min_area=min_area,
        max_area=max_area,
        context_padding=context_padding,
        target_size=224,
    )


def mask6_to_class_map(mask6: np.ndarray) -> np.ndarray:
    """Convert a 6-channel PanNuke mask to a single-channel class map (0..5)."""
    if mask6.ndim != 3 or mask6.shape[-1] != 6:
        raise ValueError("mask6 must have shape [H, W, 6]")
    background_mask = mask6[:, :, 5] > 0
    nuc_bin = (mask6[:, :, :5] > 0).astype(np.uint8)
    class_map = np.argmax(nuc_bin, axis=2).astype(np.uint8) + 1
    no_nucleus_mask = (nuc_bin.sum(axis=2) == 0)
    class_map[no_nucleus_mask] = 0
    class_map[background_mask] = 0
    return class_map


def extract_one_from_pannuke(
    dataset_root: str,
    sample_idx: int = 0,
    base_size: int = 256,
    min_area: int = 10,
    max_area: int = 50000,
    context_padding: int = 32,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Load one sample from PanNuke and extract nuclei using ground-truth 6-channel mask.

    Returns: image_rgb, gt_class_map, nuclei_instances
    """
    # Define a small wrapper dataset to return transformed image tensor and raw 6ch mask
    class PanNuke6ChannelDataset(PanNukeDataset):
        def __getitem__(self, idx: int):
            if self.storage_mode == "files":
                img_path = self.image_paths[idx]
                mask_path = self.mask_paths[idx]
                image = cv2.imread(img_path)
                if image is None:
                    raise IOError(f"Failed to read image file: {img_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask_6channel = np.load(mask_path)
                if mask_6channel.ndim == 2:
                    mask_onehot = np.zeros((*mask_6channel.shape, 6), dtype=np.uint8)
                    for c in range(1, 7):
                        mask_onehot[:, :, c - 1] = (mask_6channel == c).astype(np.uint8)
                    mask_6channel = mask_onehot
            else:
                part_idx, local_idx = self._index_map[idx]
                imgs_arr = self._parts_images[part_idx]
                masks_arr = self._parts_masks[part_idx]
                image = imgs_arr[local_idx]
                mask_6channel = masks_arr[local_idx]

            image = np.ascontiguousarray(image).astype(np.uint8)
            mask_6channel = np.ascontiguousarray(mask_6channel)

            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
                image = augmented["image"]

            # Apply standard preprocessing (resize, CLAHE, Z-score) to image only
            transforms = []
            if self.base_size is not None:
                transforms.append(A.Resize(self.base_size, self.base_size))
            transforms.extend([CLAHETransform(), ZScoreTransform(), ToTensorV2()])
            transform = A.Compose(transforms)
            transformed = transform(image=image)
            return {"image": transformed["image"], "mask_6channel": mask_6channel}

    ds = PanNuke6ChannelDataset(
        root=dataset_root,
        augmentations=None,
        validate_dataset=False,
        base_size=base_size,
    )
    if sample_idx < 0 or sample_idx >= len(ds):
        raise IndexError(f"sample_idx {sample_idx} out of range (0..{len(ds)-1})")
    sample = ds[sample_idx]
    image_tensor = sample["image"]  # [C,H,W], z-score
    mask6 = sample["mask_6channel"]  # [H,W,6]

    # Denormalize image tensor to uint8 RGB for display
    image_np = image_tensor.cpu().numpy()
    if image_np.ndim == 3 and image_np.shape[0] == 3:
        image_np = np.transpose(image_np, (1, 2, 0))
    if image_np.dtype in (np.float32, np.float64):
        img_clipped = np.clip(image_np, -3, 3)
        img_min, img_max = img_clipped.min(), img_clipped.max()
        if img_max > img_min:
            image_rgb = ((img_clipped - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            image_rgb = np.zeros_like(img_clipped, dtype=np.uint8)
    elif image_np.max() <= 1.0:
        image_rgb = (image_np * 255).astype(np.uint8)
    else:
        image_rgb = np.clip(image_np, 0, 255).astype(np.uint8)

    nuclei_instances = extract_nuclei_instances(
        image=image_rgb,
        mask_6channel=mask6,
        min_area=min_area,
        max_area=max_area,
        context_padding=context_padding,
        target_size=224,
    )
    gt_class_map = mask6_to_class_map(mask6)
    return image_rgb, gt_class_map, nuclei_instances


def build_figure(
    image_rgb: np.ndarray,
    seg_mask: np.ndarray,
    overlay_rgb: np.ndarray,
    patches: List[np.ndarray],
    out_path: str,
    title: str,
    max_patches: int = 24,
):
    import matplotlib.pyplot as plt

    seg_vis = colorize_segmentation(seg_mask) if seg_mask is not None else None
    montage = make_patch_montage(patches, tile_size=224, cols=6, max_patches=max_patches)

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    if seg_vis is not None:
        axes[0, 1].imshow(seg_vis)
        axes[0, 1].set_title("Segmentation")
        axes[0, 1].axis("off")
    else:
        axes[0, 1].axis("off")

    axes[1, 0].imshow(overlay_rgb)
    axes[1, 0].set_title("Extraction Overlay")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(montage)
    axes[1, 1].set_title("Extracted Nucleus Patches")
    axes[1, 1].axis("off")

    plt.tight_layout()
    Path(os.path.dirname(out_path) or ".").mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize nuclei extraction from an image")
    parser.add_argument("--image", type=str, required=False, default=None, help="Path to RGB image (png/jpg)")
    parser.add_argument("--mask6", type=str, default=None, help="Optional path to 6-channel PanNuke mask (.npy)")
    parser.add_argument("--seg_model", type=str, default=None, help="Optional path to segmentation checkpoint (.ckpt)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for model mode")
    parser.add_argument("--out", type=str, default="out/nuclei_extraction.png", help="Output visualization path")
    # PanNuke dataset single-sample mode
    parser.add_argument("--dataset_root", type=str, default=None, help="PanNuke dataset root to sample from")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of sample in dataset")
    parser.add_argument("--base_size", type=int, default=256, help="Base size for PanNuke preprocessing")

    # Extraction parameters
    parser.add_argument("--min_area", type=int, default=50, help="Minimum nucleus area (px)")
    parser.add_argument("--max_area", type=int, default=5000, help="Maximum nucleus area (px)")
    parser.add_argument("--context_padding", type=int, default=32, help="Context padding around nucleus (px)")
    parser.add_argument("--max_patches", type=int, default=24, help="Max patches to show in montage")

    args = parser.parse_args()

    seg_mask_np = None
    nuclei_instances = []

    if args.dataset_root is not None:
        image_rgb, seg_mask_np, nuclei_instances = extract_one_from_pannuke(
            dataset_root=args.dataset_root,
            sample_idx=args.sample_idx,
            base_size=args.base_size,
            min_area=max(1, args.min_area),
            max_area=max(args.max_area, args.min_area + 1),
            context_padding=max(0, args.context_padding),
        )
        title_image_name = f"PanNuke[{args.sample_idx}]"
    else:
        image_rgb = load_image_rgb(args.image)

        if args.mask6 is None and args.seg_model is None:
            raise ValueError("Provide either --dataset_root or one of --mask6/--seg_model for extraction.")

        if args.mask6 is not None:
            # Load mask and extract
            # Also derive class map for visualization
            mask6 = np.load(args.mask6)
            if mask6.ndim == 2:
                oh = np.zeros((*mask6.shape, 6), dtype=np.uint8)
                for c in range(1, 7):
                    oh[:, :, c - 1] = (mask6 == c).astype(np.uint8)
                mask6 = oh
            seg_mask_np = mask6_to_class_map(mask6)
            nuclei_instances = extract_nuclei_instances(
                image=image_rgb,
                mask_6channel=mask6,
                min_area=max(1, args.min_area),
                max_area=max(args.max_area, args.min_area + 1),
                context_padding=max(0, args.context_padding),
                target_size=224,
            )
        elif args.seg_model is not None:
            seg_mask_np, nuclei_instances = run_segmentation_extract(
                image_rgb=image_rgb,
                seg_model_path=args.seg_model,
                device=args.device,
                min_area=max(1, args.min_area),
                max_area=max(args.max_area, args.min_area + 1),
                context_padding=max(0, args.context_padding),
            )
        title_image_name = Path(args.image).name

    # Build overlay and montage
    overlay = draw_overlay(image_rgb, nuclei_instances)
    patches = [n["patch"] for n in nuclei_instances]

    title = f"Nuclei Extraction: {title_image_name} | {len(nuclei_instances)} nuclei"
    build_figure(
        image_rgb=image_rgb,
        seg_mask=seg_mask_np,
        overlay_rgb=overlay,
        patches=patches,
        out_path=args.out,
        title=title,
        max_patches=args.max_patches,
    )

    print(f"✅ Saved visualization to: {args.out}")


if __name__ == "__main__":
    main()


