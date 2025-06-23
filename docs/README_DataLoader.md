# PanNuke Dataset DataLoader for PyTorch

## Overview

This repository contains a comprehensive PyTorch DataLoader for the PanNuke dataset, designed for multi-class nuclei segmentation with GPU training support. The implementation includes memory optimization, data augmentation, and stain normalization capabilities.

## Features

### ✅ **Multi-class Nuclei Support**
- 6 nuclei types: Background, Neoplastic, Inflammatory, Connective, Dead, Epithelial
- 19 tissue types: Adrenal gland, Bile-duct, Bladder, Breast, Cervix, Colon, Esophagus, HeadNeck, Kidney, Liver, Lung, Ovarian, Pancreatic, Prostate, Skin, Stomach, Testis, Thyroid, Uterus

### ✅ **GPU Training Ready**
- Automatic GPU detection and utilization
- Efficient data transfer with pinned memory
- CUDA-optimized tensor operations
- Tested on NVIDIA GeForce GTX 1660 SUPER

### ✅ **Memory Optimization**
- Memory-mapped file access for large datasets (11.7GB total)
- On-demand data loading to prevent memory overflow
- Efficient batch processing

### ✅ **Data Augmentation**
- Horizontal and vertical flipping
- 90-degree rotations
- Small angle rotations
- Stain normalization for histopathological images

### ✅ **Flexible Data Splits**
- Train: 70% (1,859 samples)
- Validation: 20% (531 samples)  
- Test: 10% (266 samples)

## Files

1. **`pannuke_dataloader.py`** - Original comprehensive data loader with full features
2. **`pannuke_optimized.py`** - Memory-optimized version for large datasets ⭐
3. **`check_pytorch_gpu.py`** - GPU capability testing script
4. **`example_usage.py`** - Usage examples and training demonstrations

## Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision opencv-python pillow numpy
```

### 2. Basic Usage
```python
from pannuke_optimized import create_optimized_dataloaders

# Create data loaders
train_loader, val_loader, test_loader = create_optimized_dataloaders(
    dataset_path="./Dataset",
    batch_size=8,
    num_workers=2,
    image_size=(256, 256),
    augment=True,
    pin_memory=True
)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for batch in train_loader:
    images = batch['image'].to(device)      # Shape: (batch, 3, 256, 256)
    masks = batch['mask'].to(device)        # Shape: (batch, 6, 256, 256)
    binary_mask = batch['binary_mask']      # Shape: (batch, 256, 256)
    tissue_types = batch['tissue_type']     # Shape: (batch,)
    
    # Your training code here...
```

### 3. Test GPU Availability
```bash
python check_pytorch_gpu.py
```

### 4. Run Examples
```bash
python example_usage.py
```

## Data Structure

The DataLoader returns batches with the following structure:

```python
batch = {
    'image': torch.Tensor,        # RGB images (batch, 3, H, W)
    'mask': torch.Tensor,         # Multi-class masks (batch, 6, H, W)
    'binary_mask': torch.Tensor,  # Binary nuclei mask (batch, H, W)
    'instance_mask': torch.Tensor,# Instance segmentation (batch, H, W)
    'tissue_type': torch.Tensor,  # Tissue type labels (batch,)
    'idx': torch.Tensor          # Sample indices (batch,)
}
```

## Performance Results

✅ **GPU Training Confirmed**
- PyTorch version: 2.6.0+cu124
- CUDA version: 12.4
- cuDNN enabled: True
- Successfully loads and processes batches on GPU

✅ **Memory Efficiency**
- Handles 11.7GB dataset without memory overflow
- Memory-mapped file access
- Efficient batch processing (2-8 samples per batch tested)

✅ **Data Loading Speed**
- ~0.05-0.1 seconds per batch (GPU transfer included)
- Optimized for both CPU and GPU workflows

## Dataset Information

**PanNuke Dataset Statistics:**
- Total samples: 2,656
- Image size: 256x256x3
- Mask size: 256x256x6
- File sizes: 
  - Images: 3.9GB
  - Masks: 7.8GB
  - Types: 135KB

**Tissue Types Distribution:**
19 different tissue types from various organs and anatomical sites.

**Nuclei Classes:**
- Class 0: Background
- Class 1: Neoplastic cells
- Class 2: Inflammatory cells
- Class 3: Connective tissue cells
- Class 4: Dead cells
- Class 5: Epithelial cells

## Memory Optimization Features

The optimized data loader (`pannuke_optimized.py`) includes:

1. **Memory Mapping**: Uses `np.load(mmap_mode='r')` for efficient file access
2. **On-demand Loading**: Loads only required samples for each batch
3. **Efficient Copying**: Ensures tensor compatibility with proper array copying
4. **Reduced Workers**: Optimized worker count to prevent memory conflicts

## GPU Requirements

**Minimum Requirements:**
- NVIDIA GPU with CUDA support
- 4GB+ GPU memory (recommended 6GB+)
- CUDA toolkit installed
- PyTorch with CUDA support

**Tested Configuration:**
- GPU: NVIDIA GeForce GTX 1660 SUPER (6GB)
- CUDA: 12.4
- PyTorch: 2.6.0+cu124
- Driver: 576.52

## Troubleshooting

### Memory Issues
If you encounter memory errors:
1. Reduce `batch_size` (try 2-4)
2. Reduce `num_workers` (try 0-1)
3. Use the optimized data loader (`pannuke_optimized.py`)

### GPU Issues
If GPU is not detected:
1. Run `python check_pytorch_gpu.py`
2. Verify CUDA installation
3. Check PyTorch CUDA compatibility

### Performance Optimization
For better performance:
1. Use `pin_memory=True` for GPU training
2. Adjust `num_workers` based on CPU cores
3. Use appropriate batch size for your GPU memory

## Contributing

To extend this data loader:

1. **Add new augmentations** in `_apply_augmentations()` method
2. **Implement custom normalization** by modifying the normalization section
3. **Add new output formats** by extending the `__getitem__()` return dictionary
4. **Support other datasets** by adapting the file loading logic

## License

This implementation is provided for research and educational purposes. Please cite the original PanNuke dataset paper when using this code.

---

**Note**: This data loader is optimized for the PanNuke dataset structure. Adapt the file paths and loading logic if using different dataset formats. 