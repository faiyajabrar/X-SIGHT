from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch

# Ensure project root is on sys.path when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dataset and its internal preprocessing transforms
from utils.pannuke_dataset import PanNukeDataset, CLAHETransform, ZScoreTransform
from utils.nuclei_dataset import get_classification_augmentations


def _colorize_mask(class_map: np.ndarray) -> np.ndarray:
    """Map class IDs [0..5] to RGB colors for visualization."""
    palette = np.array(
        [
            [0, 0, 0],        # 0: background – black
            [255, 0, 0],      # 1: Neoplastic – red
            [0, 255, 0],      # 2: Inflammatory – green
            [0, 0, 255],      # 3: Connective – blue
            [255, 255, 0],    # 4: Dead – yellow
            [255, 0, 255],    # 5: Epithelial – magenta
        ],
        dtype=np.uint8,
    )
    class_map_safe = np.clip(class_map.astype(np.int64), 0, 5)
    return palette[class_map_safe]


def _to_display_image(img: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert various image formats to displayable float RGB [0,1]."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] != img.shape[-1]:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32)
    # Heuristic: if data is in [0..255] and non-negative, scale by 255; otherwise min-max
    mn, mx = float(img.min()), float(img.max())
    if mn >= 0.0 and mx <= 255.0 and mx > 1.5:
        img = img / 255.0
    else:
        img = (img - mn) / (mx - mn + 1e-7)
    img = np.clip(img, 0.0, 1.0)
    return img


def _get_step_name(t: dict) -> str:
    """Return a readable transform name from a ReplayCompose transform dict."""
    cls_full = t.get("__class_fullname__", "Transform")
    name = cls_full.split(".")[-1]
    applied = t.get("applied", False)
    return f"{name}{' [applied]' if applied else ' [skipped]'}"


def _flatten_transforms(comp: Optional[A.Compose]) -> List[A.BasicTransform]:
    """Flatten an Albumentations Compose into a list of atomic transforms.

    - Expands OneOf into individual candidates
    - Ignores ToTensor/Normalize-like post-transforms for augmentation catalog
    """
    if comp is None:
        return []

    atomic: List[A.BasicTransform] = []

    def visit(node):
        if isinstance(node, A.Compose):
            for child in node.transforms:
                visit(child)
        elif isinstance(node, A.OneOf):
            for choice in node.transforms:
                visit(choice)
        else:
            # Filter out strictly non-augmentation utility transforms if any slip in
            name = node.__class__.__name__
            if name in {"ToTensorV2", "Normalize"}:
                return
            atomic.append(node)

    visit(comp)
    return atomic


def _force_apply_transform(img: np.ndarray, msk: np.ndarray, t: A.BasicTransform) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a single transform with guaranteed application using OneOf wrapper."""
    wrapper = A.Compose([A.OneOf([t], p=1.0)])
    out = wrapper(image=img, mask=msk)
    return out["image"], out.get("mask", msk)


def _catalog_augmentations(
    raw_img: np.ndarray,
    class_map: np.ndarray,
    augmentations: Optional[A.Compose],
) -> Tuple[List[str], List[np.ndarray]]:
    """Produce one visualization per augmentation type, forcing each to apply.

    Returns labels and images; masks are used only to keep geometry consistent.
    """
    labels: List[str] = ["Raw"]
    images: List[np.ndarray] = [raw_img]

    # Flatten and deduplicate by transform class name
    flat = _flatten_transforms(augmentations)
    unique_by_name = []
    seen = set()
    for tr in flat:
        name = tr.__class__.__name__
        if name not in seen:
            seen.add(name)
            unique_by_name.append(tr)

    for tr in unique_by_name:
        try:
            img_t, _ = _force_apply_transform(raw_img, class_map, tr)
            images.append(img_t)
            labels.append(tr.__class__.__name__)
        except Exception:
            # Skip transforms that cannot be applied standalone
            continue

    return labels, images


def _apply_augmentations_stepwise(
    image: np.ndarray,
    mask: np.ndarray,
    augmentations: Optional[A.Compose],
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    """Apply optional Albumentations pipeline step-by-step using ReplayCompose.

    Returns labels, images, masks for each step (including initial raw).
    """
    labels: List[str] = ["Raw"]
    images: List[np.ndarray] = [image]
    masks: List[np.ndarray] = [mask]

    if augmentations is None or len(getattr(augmentations, "transforms", [])) == 0:
        return labels, images, masks

    # Build a ReplayCompose from the provided pipeline to capture exact random params
    replay_compose = A.ReplayCompose(list(augmentations.transforms))
    out = replay_compose(image=image, mask=mask)
    replay = out["replay"]

    # Step through each recorded transform deterministically
    for i, t in enumerate(replay.get("transforms", [])):
        partial_replay = {k: v for k, v in replay.items()}
        partial_replay["transforms"] = replay["transforms"][: i + 1]
        step_res = A.ReplayCompose.replay(partial_replay, image=image, mask=mask)
        labels.append(_get_step_name(t))
        images.append(step_res["image"])
        masks.append(step_res["mask"])

    return labels, images, masks


def _apply_base_transforms_stepwise(
    image: np.ndarray,
    mask: np.ndarray,
    base_size: Optional[int],
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    """Apply built-in preprocessing (Resize, CLAHE, Z-Score, ToTensor) step-by-step."""
    labels: List[str] = []
    images: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    current_img = image
    current_msk = mask

    if base_size is not None:
        resized = A.Resize(base_size, base_size)(image=current_img, mask=current_msk)
        current_img, current_msk = resized["image"], resized["mask"]
        labels.append("Resize")
        images.append(current_img)
        masks.append(current_msk)

    clahe = CLAHETransform()
    current_img = clahe(image=current_img)["image"]
    labels.append("CLAHE")
    images.append(current_img)
    masks.append(current_msk)

    zscore = ZScoreTransform()
    current_img = zscore(image=current_img)["image"]
    labels.append("Z-Score")
    images.append(current_img)
    masks.append(current_msk)

    to_tensor = ToTensorV2(transpose_mask=True)
    tt = to_tensor(image=current_img, mask=current_msk)
    current_img, current_msk = tt["image"], tt["mask"]
    labels.append("ToTensorV2")
    images.append(current_img)
    masks.append(current_msk)

    return labels, images, masks


def _plot_steps(
    labels: List[str],
    images: List[np.ndarray | torch.Tensor],
    save_path: Optional[str] = None,
    show: bool = True,
    ncols: int = 6,
    nrows: Optional[int] = None,
):
    """Render a grid (images only) across processing steps in row-major order.

    If nrows is provided, it overrides the automatic computation to allow
    forcing a fixed grid layout (e.g., 3x6).
    """
    num_steps = len(labels)
    ncols = max(1, int(ncols))
    if nrows is None:
        nrows = (num_steps + ncols - 1) // ncols
    else:
        nrows = max(1, int(nrows))

    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = np.array([axs])
    elif ncols == 1:
        axs = np.array([[ax] for ax in axs])

    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axs[r, c]
            if idx < num_steps:
                disp_img = _to_display_image(images[idx])
                ax.imshow(disp_img)
                ax.set_title(labels[idx], fontsize=9)
                ax.axis("off")
                idx += 1
            else:
                ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def visualize_sample(
    dataset_root: str,
    idx: int = 0,
    base_size: Optional[int] = 256,
    pipeline: str = "segmentation",
    ensure_all: bool = True,
    epoch: int = 0,
    total_epochs: int = 60,
    save_path: Optional[str] = None,
    show: bool = True,
    ncols: int = 6,
):
    """Visualize PanNuke sample across augmentation and preprocessing steps.

    pipeline options:
      - 'pannuke': only built-in dataset preprocessing (no external augmentations)
      - 'segmentation': use training progressive augmentations from train_segmentation.py
      - 'classification': use augmentations from nuclei_dataset.get_classification_augmentations(training=True)
      - 'all': union of segmentation + classification augmentations (deduplicated by class)
    """
    augmentations: Optional[A.Compose] = None
    if pipeline in ("segmentation", "all"):
        try:
            from training.train_segmentation import get_progressive_augmentations
            seg_comp = get_progressive_augmentations(epoch=epoch, max_epochs=total_epochs)
        except Exception:
            seg_comp = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            ])
    else:
        seg_comp = None

    if pipeline in ("classification", "all"):
        try:
            cls_comp = get_classification_augmentations(training=True)
        except Exception:
            cls_comp = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=20, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),
                A.GaussNoise(var_limit=(5.0, 15.0), p=0.15),
                A.GaussianBlur(blur_limit=2, p=0.1),
            ])
    else:
        cls_comp = None

    if pipeline == "segmentation":
        augmentations = seg_comp
    elif pipeline == "classification":
        augmentations = cls_comp
    elif pipeline == "all":
        # Combine transforms, deduplicate by class name while preserving order
        combined: List[A.BasicTransform] = []
        seen = set()
        def add_transforms(comp: Optional[A.Compose]):
            if comp is None:
                return
            for t in comp.transforms:
                name = t.__class__.__name__
                if name not in seen:
                    seen.add(name)
                    combined.append(t)
        add_transforms(seg_comp)
        add_transforms(cls_comp)
        augmentations = A.Compose(combined) if combined else None
    else:
        # 'pannuke' → no external augmentations, only built-in preprocessing
        augmentations = None

    # Build dataset to leverage its indexing/mapping without applying transforms
    ds = PanNukeDataset(root=dataset_root, augmentations=None, validate_dataset=False, base_size=None)

    # Load raw image and class map without dataset transforms
    if ds.storage_mode == "files":
        img_path = ds.image_paths[idx]
        mask_path = ds.mask_paths[idx]
        import cv2
        raw_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        raw_mask_6 = np.load(mask_path)
        if raw_mask_6.ndim == 2:
            mask_onehot = np.zeros((*raw_mask_6.shape, 6), dtype=np.uint8)
            for c in range(1, 7):
                mask_onehot[:, :, c - 1] = (raw_mask_6 == c).astype(np.uint8)
            raw_mask_6 = mask_onehot
    else:
        part_idx, local_idx = ds._index_map[idx]
        raw_img = ds._parts_images[part_idx][local_idx]
        raw_mask_6 = ds._parts_masks[part_idx][local_idx]

    # Convert 6-channel mask to semantic class map using the dataset's logic
    class_map = ds._mask_to_class_map(np.ascontiguousarray(raw_mask_6))
    raw_img = np.ascontiguousarray(raw_img).astype(np.uint8)

    if ensure_all:
        # Build a specific catalog: include Raw and selected transforms only
        labels = ["Raw"]
        images = [raw_img]

        # Use provided base_size for resize if available, otherwise default to 256
        resize_size = base_size if base_size is not None else 256

        specific_transforms: List[A.BasicTransform] = [
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.RandomRotate90(p=1.0),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=1.0),
            A.GridDistortion(distort_limit=0.15, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=1.0),
            A.ColorJitter(brightness=0.1, contrast=0.15, saturation=0.1, hue=0.02, p=1.0),
            A.GaussianBlur(blur_limit=(1, 5), p=1.0),
            A.GaussNoise(var_limit=(10, 80), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.Resize(resize_size, resize_size),
            CLAHETransform(),
            ZScoreTransform(),
        ]

        for tr in specific_transforms:
            try:
                img_t, _ = _force_apply_transform(raw_img, class_map, tr)
                images.append(img_t)
                # Pretty label for known custom transforms
                if isinstance(tr, CLAHETransform):
                    labels.append("CLAHE")
                elif isinstance(tr, ZScoreTransform):
                    labels.append("Z-Score")
                elif isinstance(tr, A.Resize):
                    labels.append("Resize")
                else:
                    labels.append(tr.__class__.__name__)
            except Exception:
                continue

        # Force a 3x6 grid: 1 raw + 15 augmentations = 16 tiles; remaining slots hidden
        _plot_steps(labels, images, save_path=save_path, show=show, ncols=6, nrows=3)
        return

    # Otherwise: stepwise replay + built-in preprocessing
    aug_labels, aug_images, aug_masks = _apply_augmentations_stepwise(raw_img, class_map, augmentations)
    aug_final_img = aug_images[-1]
    aug_final_mask = aug_masks[-1]
    base_labels, base_images, _ = _apply_base_transforms_stepwise(aug_final_img, aug_final_mask, base_size)
    labels = aug_labels + base_labels
    images = aug_images + base_images
    _plot_steps(labels, images, save_path=save_path, show=show, ncols=ncols)


def main():
    parser = argparse.ArgumentParser(description="Visualize PanNuke augmentations and preprocessing stages (images only)")
    parser.add_argument("--dataset-root", type=str, default="Dataset", help="Root directory containing PanNuke parts")
    parser.add_argument("--index", type=int, default=0, help="Sample index to visualize")
    parser.add_argument("--base-size", type=int, default=256, help="Resize dimension used by dataset (None to skip)")
    parser.add_argument("--no-resize", action="store_true", help="Skip resizing in base transforms")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="segmentation",
        choices=["pannuke", "segmentation", "classification", "all"],
        help="Which augmentation pipeline to visualize"
    )
    parser.add_argument("--epoch", type=int, default=0, help="Epoch used to derive progressive augmentations")
    parser.add_argument("--total-epochs", type=int, default=60, help="Total epochs for progressive schedule")
    parser.add_argument("--save", type=str, default=None, help="Path to save the figure (PNG)")
    parser.add_argument("--ensure-all", action="store_true", default=True, help="Force-show every augmentation type once")
    parser.add_argument("--cols", type=int, default=6, help="Number of columns in the output grid")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot interactively")
    args = parser.parse_args()

    base_size = None if args.no_resize else args.base_size

    visualize_sample(
        dataset_root=args.dataset_root,
        idx=args.index,
        base_size=base_size,
        pipeline=args.pipeline,
        epoch=args.epoch,
        total_epochs=args.total_epochs,
        save_path=args.save,
        show=not args.no_show,
        ncols=args.cols,
    )


if __name__ == "__main__":
    main()


