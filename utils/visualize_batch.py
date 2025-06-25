import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.pannuke_dataset import PanNukeDataset

# Dataset & loader
print("Initialising dataset …")
ds = PanNukeDataset(
    root='Dataset',
    parts=['Part 1', 'Part 2', 'Part 3'],
    image_dirname='Images',
    mask_dirname='Masks',
    img_size=(256, 256),
)
print(f"Dataset length: {len(ds)}")

loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

batch = next(iter(loader))
images = batch['image']  # (B,3,256,256)
masks = batch['mask']    # (B,256,256)

img = images[0]
mask = masks[0]

# Unique values in mask
print('Unique labels in this mask:', torch.unique(mask))

# Shape sanity check
assert img.shape == (3, 256, 256), f"Unexpected image shape {img.shape}"
assert mask.shape == (256, 256), f"Unexpected mask shape {mask.shape}"
print('Shape checks passed.')

# Denormalise image for display
img_np = img.permute(1, 2, 0).cpu().numpy()
img_np = img_np * ds.IMAGENET_STD + ds.IMAGENET_MEAN
img_np = img_np.clip(0, 1)

# Colour mask
palette = torch.tensor(
    [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]],
    dtype=torch.uint8,
)
mask_rgb = palette[mask.cpu()].numpy()

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(img_np)
axs[0].set_title('Image')
axs[0].axis('off')
axs[1].imshow(mask_rgb)
axs[1].set_title('Mask')
axs[1].axis('off')
plt.tight_layout()
plt.show() 