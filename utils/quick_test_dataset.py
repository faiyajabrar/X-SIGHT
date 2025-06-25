import torch
from utils.pannuke_dataset import PanNukeDataset

ds = PanNukeDataset(root='Dataset', parts=['Part 1', 'Part 2', 'Part 3'], image_dirname='Images', mask_dirname='Masks', img_size=(256, 256))
print('Total samples:', len(ds))

sample = ds[0]
print('Image tensor:', sample['image'].shape, sample['image'].dtype)
print('Mask tensor:', sample['mask'].shape, sample['mask'].dtype, 'unique labels:', torch.unique(sample['mask']))

part_counts, total_concat, total_after_aug = ds.dataset_summary()
print('Per-part counts:', part_counts)
print('Total after concatenation:', total_concat)
print('Total effective samples after on-the-fly augmentation:', total_after_aug) 