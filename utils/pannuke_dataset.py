from __future__ import annotations
import os
from glob import glob
from typing import List, Tuple, Optional, Sequence, Union, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CLAHETransform(A.ImageOnlyTransform):
    """CLAHE preprocessing."""
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(16, 16), always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l_channel)
        
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return enhanced_rgb


class ZScoreTransform(A.ImageOnlyTransform):
    """Z-score normalization."""
    
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = img.astype(np.float32)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        std = np.std(img, axis=(0, 1), keepdims=True)
        
        std = np.maximum(std, 1e-6)
        normalized_image = (img - mean) / std
        normalized_image = np.clip(normalized_image, -10.0, 10.0)
        
        return normalized_image


class PanNukeDataset(Dataset):
    """PyTorch Dataset for the PanNuke nucleus segmentation dataset.

    The official PanNuke release (v1.0) is divided into three folds (part0, part1, part2)
    each containing image patches (.png) and their corresponding 6-channel masks (.npy).

    This dataset implementation concatenates the three parts automatically so that they can
    be treated as a single dataset. It includes built-in preprocessing transforms and 
    returns PyTorch tensors ready for training.
    """

    def __init__(
        self,
        root: str,
        parts: Sequence[str] = ("Part 1", "Part 2", "Part 3"),
        image_dirname: str = "Images",
        mask_dirname: str = "Masks",
        augmentations: Optional[A.Compose] = None,
        validate_dataset: bool = True,
        base_size: int = 256,
    ) -> None:
        """Parameters
        ----------
        root : str
            Root directory that contains the PanNuke parts (e.g. ``/path/to/PanNuke``).
        parts : sequence of str, default ("Part 1", "Part 2", "Part 3")
            The sub-directories to concatenate. You can override this to use a subset.
        image_dirname : str, default "Images"
            Name of the directory inside each part that holds RGB image patches.
        mask_dirname : str, default "Masks"
            Name of the directory inside each part that holds corresponding *6-channel* masks.
        augmentations : A.Compose, optional
            Additional augmentations to apply (rotation, flips, etc.). Will be applied before
            the built-in CLAHE + Z-score normalization.
        validate_dataset : bool, default True
            If ``True``, performs basic validation checks on dataset integrity.
        base_size : int, default 256
            Base size for resizing images. Can be overridden for progressive resizing.
        """
        super().__init__()

        self.augmentations = augmentations
        self.base_size = base_size

        # Built-in preprocessing transforms (applied to all samples)
        transforms = []
        if base_size is not None:
            transforms.append(A.Resize(base_size, base_size))
        
        transforms.extend([
            CLAHETransform(),
            ZScoreTransform(),
            ToTensorV2(transpose_mask=True)
        ])
        
        self.base_transforms = A.Compose(transforms)

        self.image_paths: List[str] = []
        self.mask_paths: List[str] = []

        # Two possible storage modes: (1) individual files (.png, .npy) per sample; (2) aggregated
        # arrays saved as images.npy and masks.npy. We try to auto-detect which mode to use.

        self.storage_mode = None  # "files" or "aggregated"

        # Aggregated containers per part (lazy memmap)
        self._parts_images: List[np.ndarray] = []  # list of memmap arrays (N, H, W, C)
        self._parts_masks: List[np.ndarray] = []   # list of memmap arrays (N, H, W, 6)

        self._index_map: List[Tuple[int, int]] = []  # dataset idx -> (part_idx, local_idx)
        
        for p_idx, p in enumerate(parts):
            part_dir = os.path.join(root, p)
            if not os.path.isdir(part_dir):
                raise FileNotFoundError(f"Part directory not found: {part_dir}")

            # Heuristic: if images.npy exists under `Images` (or image_dirname), treat as aggregated
            images_npy_path = os.path.join(part_dir, image_dirname, "images.npy")
            masks_npy_path = os.path.join(part_dir, mask_dirname, "masks.npy")

            if os.path.isfile(images_npy_path) and os.path.isfile(masks_npy_path):
                # Aggregated mode for this part
                if self.storage_mode is None:
                    self.storage_mode = "aggregated"
                elif self.storage_mode != "aggregated":
                    raise RuntimeError("Mixed storage modes across parts are not supported.")

                imgs_arr = np.load(images_npy_path, mmap_mode="r")  # shape (N, H, W, 3)
                msk_arr = np.load(masks_npy_path, mmap_mode="r")   # shape (N, H, W, 6)

                if imgs_arr.shape[0] != msk_arr.shape[0]:
                    raise ValueError(
                        f"Image/mask count mismatch in part '{p}': {imgs_arr.shape[0]} vs {msk_arr.shape[0]}"
                    )

                self._parts_images.append(imgs_arr)
                self._parts_masks.append(msk_arr)

                start_idx = len(self._index_map)
                for local_idx in range(imgs_arr.shape[0]):
                    self._index_map.append((p_idx, local_idx))

                # record counts
                setattr(self, f"_count_part_{p_idx}", imgs_arr.shape[0])

            else:
                # Fallback to per-file mode
                image_glob = os.path.join(part_dir, image_dirname, "*.png")
                mask_glob = os.path.join(part_dir, mask_dirname, "*.npy")

                part_images = sorted(glob(image_glob))
                part_masks = sorted(glob(mask_glob))

                if len(part_images) == 0:
                    raise RuntimeError(
                        f"No images found in part '{p}'. Expected either '*.png' files in '{image_dirname}' or aggregated 'images.npy'."
                    )

                if len(part_images) != len(part_masks):
                    raise ValueError(
                        f"Image/mask count mismatch in part '{p}': {len(part_images)} vs {len(part_masks)}"
                    )

                if self.storage_mode is None:
                    self.storage_mode = "files"
                elif self.storage_mode != "files":
                    raise RuntimeError("Mixed storage modes across parts are not supported.")

                # Extend paths
                self.image_paths.extend(part_images)
                self.mask_paths.extend(part_masks)

                setattr(self, f"_count_part_{p_idx}", len(part_images))

        if self.storage_mode is None:
            raise RuntimeError("Dataset initialisation failed – no data found.")

        # Validate dataset integrity if requested
        if validate_dataset:
            self._validate_dataset()

        print(f"[PanNukeDataset] Loaded {len(self)} samples using {self.storage_mode} storage mode.")
        print(f"[PanNukeDataset] Built-in transforms: CLAHE + Z-score + ToTensor")

    def _validate_dataset(self):
        """Perform basic validation checks on the dataset."""
        print(f"[PanNukeDataset] Validating dataset integrity...")
        
        # Check we have data
        if len(self) == 0:
            raise ValueError("Dataset is empty. Check your data paths.")
        
        # Sample a few items to check format
        sample_indices = [0, len(self) // 2, len(self) - 1] if len(self) > 2 else [0]
        
        for idx in sample_indices:
            try:
                sample = self[idx]
                img, mask = sample['image'], sample['mask']
                
                # Check image format (now tensors)
                if not isinstance(img, torch.Tensor) or img.ndim != 3:
                    raise ValueError(f"Invalid image format at index {idx}. Expected 3D tensor, got {type(img)} with shape {getattr(img, 'shape', 'N/A')}")
                
                # Check mask format  
                if not isinstance(mask, torch.Tensor) or mask.ndim != 2:
                    raise ValueError(f"Invalid mask format at index {idx}. Expected 2D tensor, got {type(mask)} with shape {getattr(mask, 'shape', 'N/A')}")
                
                # Check class values in mask
                unique_classes = torch.unique(mask).tolist()
                invalid_classes = [c for c in unique_classes if c < 0 or c > 5]
                if invalid_classes:
                    raise ValueError(f"Invalid class values {invalid_classes} found in mask at index {idx}. Expected values in [0, 5].")
                    
            except Exception as e:
                raise RuntimeError(f"Dataset validation failed at index {idx}: {str(e)}")
        
        print(f"[PanNukeDataset] Validation passed. Dataset appears healthy.")

    # -------------------------------------------------- #
    def __len__(self) -> int:
        if self.storage_mode == "files":
            return len(self.image_paths)
        else:  # aggregated
            return len(self._index_map)

    # -------------------------------------------------- #
    def _mask_to_class_map(self, mask: np.ndarray) -> np.ndarray:
        """Convert a 6-channel instance‐level mask to a single-channel semantic label map.

        Channel layout (PanNuke *X-SIGHT convention*):
            0 – Neoplastic
            1 – Inflammatory
            2 – Connective
            3 – Dead
            4 – Epithelial
            5 – Background (binary mask; 1 means *background*, 0 otherwise)

        Important — difference from the previous implementation
        --------------------------------------------------------
        • **Channel-5 is now treated as an *explicit* background indicator.**
          There is *no* separate "tissue" channel.
        • We therefore **do NOT** infer background from "all nucleus channels are zero".

        Conversion procedure
        --------------------
        1. Determine pixels that belong to background using channel-5.
        2. For the remaining pixels consider only the first five channels; any non-zero
           value indicates presence of that nucleus class.
        3. The semantic class for a foreground pixel is the `argmax` across the first five
           binary channels **plus 1** (to map 0-based indices → label IDs 1-5).

        Returns
        -------
        uint8 ndarray of shape (H, W) with labels in the range 0-5, where 0 denotes
        background and 1-5 are the five nucleus classes.
        """
        if mask.ndim != 3 or mask.shape[2] != 6:
            raise ValueError("PanNuke masks must be (H,W,6).")

        # Binary background mask from channel-5
        background_mask = mask[:, :, 5] > 0

        # Binary presence maps for the five nucleus channels
        nuc_bin = (mask[:, :, :5] > 0).astype(np.uint8)

        # Resolve class for foreground pixels (those not marked as background)
        # The argmax of a zero vector is 0. We add 1 to map to labels 1-5.
        class_map = np.argmax(nuc_bin, axis=2).astype(np.uint8) + 1

        # Correctly handle pixels that are not background but have no nucleus activity.
        # These should be background (class 0), not class 1 (Neoplastic).
        no_nucleus_mask = (nuc_bin.sum(axis=2) == 0)
        class_map[no_nucleus_mask] = 0

        # Apply explicit background where indicated
        class_map[background_mask] = 0

        return class_map

    # -------------------------------------------------- #
    def __getitem__(self, idx: int):
        if self.storage_mode == "files":
            # ---------------- File-per-sample mode ----------------
            img_path = self.image_paths[idx]
            mask_path = self.mask_paths[idx]

            # Load image (BGR -> RGB using cv2)
            image = cv2.imread(img_path)
            if image is None:
                raise IOError(f"Failed to read image file: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load mask – stored as *.npy arrays of shape (H, W, 6)
            mask = np.load(mask_path)
            if mask.ndim == 2:  # occasionally masks might be single-channel uint16 label maps
                # Convert to one-hot channels (0 is background)
                mask_onehot = np.zeros((*mask.shape, 6), dtype=np.uint8)
                for c in range(1, 7):
                    mask_onehot[:, :, c - 1] = (mask == c).astype(np.uint8)
                mask = mask_onehot

        else:
            # ---------------- Aggregated numpy arrays ----------------
            part_idx, local_idx = self._index_map[idx]
            imgs_arr = self._parts_images[part_idx]
            masks_arr = self._parts_masks[part_idx]

            image = imgs_arr[local_idx]
            mask = masks_arr[local_idx]

        # Ensure contiguous memory layout and proper data types
        image = np.ascontiguousarray(image).astype(np.uint8)
        mask = np.ascontiguousarray(mask)

        # Convert masks to class map
        class_map = self._mask_to_class_map(mask)

        # Apply optional augmentations first (if provided)
        if self.augmentations is not None:
            augmented = self.augmentations(image=image, mask=class_map)
            image, class_map = augmented["image"], augmented["mask"]

        # Apply built-in preprocessing transforms (CLAHE + Z-score + ToTensor)
        transformed = self.base_transforms(image=image, mask=class_map)
        
        # Convert mask to LongTensor for F.one_hot compatibility
        transformed["mask"] = transformed["mask"].long()
        
        return {"image": transformed["image"], "mask": transformed["mask"]}

    # -------------------------------------------------- #
    def visualise(self, idx: int):
        """Render the raw image and its corresponding mask for quick visual inspection."""
        sample = self[idx]
        img = sample["image"]
        mask = sample["mask"]

        # Handle both numpy arrays and tensors
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))  # C, H, W -> H, W, C
            # Handle normalized images by rescaling to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-7)
            img = np.clip(img, 0, 1)
        else:
            # Raw numpy array - normalize to [0, 1]
            img = img.astype(np.float32) / 255.0

        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        # Assign distinct colours to classes (0 background)
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
        mask_rgb = palette[mask]

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img)
        axs[0].set_title("Image")
        axs[0].axis("off")
        axs[1].imshow(mask_rgb)
        axs[1].set_title("Mask (labels)")
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------- #
    def dataset_summary(self):
        """Return statistics: per-part counts, total after concat."""
        part_counts = {}
        # counts stored as attributes _count_part_{i}
        i = 0
        while hasattr(self, f"_count_part_{i}"):
            part_counts[i] = getattr(self, f"_count_part_{i}")
            i += 1

        total_concat = sum(part_counts.values())

        return part_counts, total_concat
