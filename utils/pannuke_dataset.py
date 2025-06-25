import os
from glob import glob
from typing import List, Tuple, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    # Albumentations is optional – create a minimal stub so that type annotations and runtime
    # isinstance checks do not fail when the library is absent.

    class _AlbumentationsStub:
        class Compose:  # dummy type placeholder
            def __init__(self, *args, **kwargs):
                pass

        class NoOp:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, **kwargs):
                return kwargs

    A = _AlbumentationsStub()
    ToTensorV2 = None  # type: ignore

import matplotlib.pyplot as plt

# Optional staintools import (not required anymore)
try:
    import staintools  # type: ignore
except ImportError:
    staintools = None

# Only define StainNormalize if a real Albumentations module is available (i.e., has ImageOnlyTransform)
if A is not None and hasattr(A, "ImageOnlyTransform"):
    class SimpleStainNormalize(A.ImageOnlyTransform):
        """Performs simple colour normalisation by matching per-channel mean and std to a reference.

        This is not as sophisticated as Macenko but provides an approximation without heavy deps.
        """

        def __init__(self, reference: np.ndarray, always_apply: bool = False, p: float = 0.5):
            super().__init__(always_apply=always_apply, p=p)

            # compute ref stats in float domain
            ref = reference.astype(np.float32) / 255.0
            self.ref_mean = ref.reshape(-1, 3).mean(0)
            self.ref_std = ref.reshape(-1, 3).std(0) + 1e-6  # avoid div0

        def apply(self, img: np.ndarray, **params):  # type: ignore[override]
            x = img.astype(np.float32) / 255.0
            mean = x.reshape(-1, 3).mean(0)
            std = x.reshape(-1, 3).std(0) + 1e-6
            x = (x - mean) / std * self.ref_std + self.ref_mean
            x = np.clip(x, 0, 1) * 255.0
            return x.astype(np.uint8)

class PanNukeDataset(Dataset):
    """PyTorch Dataset for the PanNuke nucleus segmentation dataset.

    The official PanNuke release (v1.0) is divided into three folds (part0, part1, part2)
    each containing image patches (.png) and their corresponding 6-channel masks (.npy).

    This dataset implementation concatenates the three parts automatically so that they can
    be treated as a single dataset. It also performs common preprocessing steps such as
    resizing, normalisation and (optional) data augmentation.
    """

    # ImageNet statistics for RGB normalisation (used by default)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: str,
        parts: Sequence[str] = ("part1", "part2", "part3"),
        image_dirname: str = "images",
        mask_dirname: str = "masks",
        img_size: Tuple[int, int] = (256, 256),
        normalise: bool = True,
        transforms: Optional[Union[T.Compose, A.Compose]] = None,
    ) -> None:
        """Parameters
        ----------
        root : str
            Root directory that contains the PanNuke parts (e.g. ``/path/to/PanNuke``).
        parts : sequence of str, default ("part1", "part2", "part3")
            The sub-directories to concatenate. You can override this to use a subset.
        image_dirname : str, default "images"
            Name of the directory inside each part that holds RGB image patches.
        mask_dirname : str, default "masks"
            Name of the directory inside each part that holds corresponding *6-channel* masks.
        img_size : (int, int), default (256, 256)
            Spatial size ``(height, width)`` to which images and masks will be resized.
        normalise : bool, default True
            If ``True``, images are normalised using ImageNet statistics.
        transforms : torchvision.transforms.Compose | albumentations.Compose | None, optional
            Custom transform pipeline. If supplied, the internal resize/normalise/augment is
            *skipped* and responsibility is handed to the caller.
        """
        super().__init__()

        self.img_size = img_size
        self.normalise = normalise
        self.has_albu = (ToTensorV2 is not None)
        if not self.has_albu:
            raise ImportError(
                "Mandatory augmentations require Albumentations. Please install with: pip install albumentations[imgaug]==1.3.1"
            )

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

        # Build a simple default transformation pipeline if none supplied.
        if transforms is None:
            if self.has_albu:
                # choose a reference image for stain normalisation (first image found)
                ref_img_path = None
                if self.storage_mode == "files":
                    ref_img_path = self.image_paths[0]
                    ref_img = cv2.cvtColor(cv2.imread(ref_img_path), cv2.COLOR_BGR2RGB)
                else:
                    # aggregated – access first memmap entry
                    ref_img = self._parts_images[0][0]

                stain_norm_transform = SimpleStainNormalize(ref_img, p=0.5) if A is not None else A.NoOp()

                self.transforms = A.Compose(
                    [
                        A.Resize(*img_size, interpolation=cv2.INTER_CUBIC),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.7),
                        stain_norm_transform,
                        A.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD) if normalise else A.NoOp(),
                        ToTensorV2(transpose_mask=True),
                    ],
                    additional_targets={"mask": "mask"},
                )
            else:
                # Fallback to torchvision transforms (deterministic, no augmentations)
                t_list = [
                    T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                ]
                if normalise:
                    t_list.append(T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD))
                self.transforms = T.Compose(t_list)
        else:
            self.transforms = transforms

    # -------------------------------------------------- #
    def __len__(self) -> int:
        if self.storage_mode == "files":
            return len(self.image_paths)
        else:  # aggregated
            return len(self._index_map)

    # -------------------------------------------------- #
    def _mask_to_class_map(self, mask: np.ndarray) -> np.ndarray:
        """Convert a 6-channel binary mask to a single-channel class label map.

        Each channel encodes one class. The background (no nucleus) is encoded where all
        channels equal 0. We use ``argmax`` to obtain the *dominant* class per pixel, then
        shift class indices by +1 so that 0 can be reserved for background.
        """
        if mask.ndim != 3 or mask.shape[2] != 6:
            raise ValueError("Expected mask shape (H, W, 6)")

        class_map = np.argmax(mask, axis=2).astype(np.uint8)  # (H, W)
        # Zero everywhere but >0 where any channel is active
        background = (mask.sum(axis=2) == 0)
        class_map += 1  # shift so that background would be class 1, temporarily
        class_map[background] = 0  # set background
        return class_map  # dtype uint8, values 0-6

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

        # Ensure contiguous memory layout which some transforms expect
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        # Convert masks to class map
        class_map = self._mask_to_class_map(mask)

        # Apply transforms
        if self.has_albu and isinstance(self.transforms, A.Compose):
            transformed = self.transforms(image=image, mask=class_map)
            image_t = transformed["image"]  # already tensor
            mask_t = transformed["mask"].long()  # ToTensorV2 handles uint8->tensor
        else:  # torchvision transforms
            image_pil = F.to_pil_image(image)
            image_t = self.transforms(image_pil)
            # Resize mask via nearest
            mask_resized = cv2.resize(
                class_map,
                self.img_size[::-1],
                interpolation=cv2.INTER_NEAREST,
            )
            mask_t = torch.from_numpy(mask_resized).long()

        return {"image": image_t, "mask": mask_t}

    # -------------------------------------------------- #
    def visualise(self, idx: int):
        """Render the image and its corresponding mask for quick visual inspection."""
        sample = self[idx]
        img = sample["image"].cpu()
        mask = sample["mask"].cpu()

        if img.ndim == 3 and img.shape[0] in (1, 3):
            img_np = img.numpy()
            img_np = np.transpose(img_np, (1, 2, 0))  # C, H, W -> H, W, C
            if self.normalise:
                img_np = img_np * np.array(self.IMAGENET_STD) + np.array(self.IMAGENET_MEAN)
                img_np = np.clip(img_np, 0, 1)
        else:
            raise RuntimeError("Unexpected image tensor shape for visualisation")

        mask_np = mask.numpy()
        # Assign distinct colours to classes (0 background)
        palette = np.array(
            [
                [0, 0, 0],  # background – black
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [255, 0, 255],
                [0, 255, 255],
            ],
            dtype=np.uint8,
        )
        mask_rgb = palette[mask_np]

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img_np)
        axs[0].set_title("Image")
        axs[0].axis("off")
        axs[1].imshow(mask_rgb)
        axs[1].set_title("Mask (labels)")
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------- #
    def dataset_summary(self):
        """Return statistics: per-part counts, total after concat (original), total effective (same due to on-the-fly augmentation)."""
        part_counts = {}
        # counts stored as attributes _count_part_{i}
        i = 0
        while hasattr(self, f"_count_part_{i}"):
            part_counts[i] = getattr(self, f"_count_part_{i}")
            i += 1

        total_concat = sum(part_counts.values())
        total_after_aug = total_concat  # on-the-fly augmentation keeps count unchanged

        return part_counts, total_concat, total_after_aug 