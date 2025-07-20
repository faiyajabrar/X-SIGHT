#!/usr/bin/env python3
"""
Evaluation script for Attention U-Net training results.
Shows original images, ground truth masks, and predicted masks with performance metrics.
"""

import sys
import os
import platform
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from utils.pannuke_dataset import PanNukeDataset
from models.attention_unet import AttentionUNet
from training.train import AdvancedAttentionModel, _calculate_class_wise_dice, _frequency_weighted_dice_score

# Fix for Windows multiprocessing
if platform.system() == 'Windows':
    torch.multiprocessing.set_start_method('spawn', force=True)

# Store original torch.load function before any modifications
_original_torch_load = torch.load

def safe_torch_load(path, map_location=None, **kwargs):
    """Safely load PyTorch checkpoint with compatibility for different PyTorch versions."""
    # Always use weights_only=False for PyTorch 2.6+ compatibility
    kwargs['weights_only'] = False
    try:
        return _original_torch_load(path, map_location=map_location, **kwargs)
    except TypeError as e:
        if 'weights_only' in str(e):
            # Remove weights_only for older PyTorch versions
            kwargs.pop('weights_only', None)
            return _original_torch_load(path, map_location=map_location, **kwargs)
        else:
            raise e


class ModelEvaluator:
    """Comprehensive model evaluation with visualization and metrics."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """Initialize evaluator with checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.checkpoint_path = checkpoint_path
        
        # Class names and colors for visualization
        self.class_names = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
        self.class_colors = np.array([
            [0, 0, 0],        # 0: Background - black
            [255, 0, 0],      # 1: Neoplastic - red
            [0, 255, 0],      # 2: Inflammatory - green  
            [0, 0, 255],      # 3: Connective - blue
            [255, 255, 0],    # 4: Dead - yellow
            [255, 0, 255],    # 5: Epithelial - magenta
        ], dtype=np.uint8)
        
        print(f"🎯 Initializing evaluator on device: {self.device}")
        self.load_model()
    
    def load_model(self):
        """Load the trained model from checkpoint."""
        print(f"📁 Loading checkpoint: {self.checkpoint_path}")
        
        # Load checkpoint with compatibility fix
        checkpoint = safe_torch_load(self.checkpoint_path, map_location=self.device)
        
        # Extract hyperparameters from checkpoint
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            print(f"📋 Model hyperparameters: {hparams}")
        else:
            print("⚠️  No hyperparameters found in checkpoint")
        
        # Create model from checkpoint - we need to temporarily patch torch.load for lightning
        original_torch_load = torch.load
        torch.load = safe_torch_load
        
        try:
            self.model = AdvancedAttentionModel.load_from_checkpoint(
                self.checkpoint_path,
                map_location=self.device
            )
        finally:
            # Always restore original torch.load
            torch.load = original_torch_load
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully!")
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def load_dataset(self, validation_only: bool = True) -> PanNukeDataset:
        """Load the evaluation dataset.
        
        Args:
            validation_only: If True, load only validation indices from saved split
            
        Returns:
            PanNukeDataset: Dataset for evaluation
        """
        print("📂 Loading dataset...")
        
        # Create base dataset (no augmentations for evaluation)
        dataset = PanNukeDataset(
            root='Dataset',
            augmentations=None,
            validate_dataset=False,
            base_size=256  # Use final training size
        )
        
        if validation_only:
            # Load validation indices from saved split
            data_split_path = Path('lightning_logs/data_split.json')
            if data_split_path.exists():
                with open(data_split_path, 'r') as f:
                    split_data = json.load(f)
                val_indices = [int(idx) for idx in split_data['val_indices']]
                
                # Create subset with validation indices
                eval_dataset = torch.utils.data.Subset(dataset, val_indices)
                print(f"📊 Loaded validation subset: {len(val_indices)} samples")
                return eval_dataset
            else:
                print("⚠️  No saved data split found, using full dataset")
        
        print(f"📊 Loaded full dataset: {len(dataset)} samples")
        return dataset
    
    def predict_sample(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference on a single image.
        
        Args:
            image: Input image tensor [C, H, W]
            
        Returns:
            Tuple of (logits, predicted_classes)
        """
        # Add batch dimension and move to device
        image_batch = image.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            logits = self.model(image_batch)
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
        return logits.squeeze(0), predictions.squeeze(0)
    
    def calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate comprehensive metrics for predictions.
        
        Args:
            predictions: Predicted class tensor [H, W]
            targets: Ground truth class tensor [H, W]
            
        Returns:
            Dictionary of metrics
        """
        # Move to CPU for metric calculation
        preds_cpu = predictions.cpu()
        targets_cpu = targets.cpu()
        
        # Overall metrics
        accuracy = (preds_cpu == targets_cpu).float().mean().item()
        dice_score = _frequency_weighted_dice_score(preds_cpu, targets_cpu, num_classes=6).item()
        
        # Class-wise Dice scores
        class_dice = _calculate_class_wise_dice(preds_cpu, targets_cpu, num_classes=6)
        
        # Convert class-wise scores to regular dict
        metrics = {
            'accuracy': accuracy,
            'dice_score': dice_score,
        }
        
        # Add class-wise metrics
        for key, value in class_dice.items():
            metrics[key] = value.item()
        
        return metrics
    
    def visualize_prediction(self, 
                           original_image: np.ndarray,
                           ground_truth: np.ndarray, 
                           prediction: np.ndarray,
                           metrics: Dict[str, float],
                           sample_idx: int,
                           save_path: str = None,
                           show_plot: bool = True) -> plt.Figure:
        """Create visualization of original image, ground truth, and prediction.
        
        Args:
            original_image: Original image [H, W, 3]
            ground_truth: Ground truth mask [H, W]  
            prediction: Predicted mask [H, W]
            metrics: Dictionary of calculated metrics
            sample_idx: Sample index for title
            save_path: Optional path to save the figure
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Sample {sample_idx} - Accuracy: {metrics["accuracy"]:.3f}, Dice: {metrics["dice_score"]:.3f}', 
                     fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Ground truth mask
        gt_colored = self.class_colors[ground_truth]
        axes[0, 1].imshow(gt_colored)
        axes[0, 1].set_title('Ground Truth', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Predicted mask
        pred_colored = self.class_colors[prediction]
        axes[0, 2].imshow(pred_colored)
        axes[0, 2].set_title('Prediction', fontweight='bold')
        axes[0, 2].axis('off')
        
        # Difference map (errors)
        diff_map = (ground_truth != prediction).astype(np.uint8)
        axes[1, 0].imshow(diff_map, cmap='Reds', vmin=0, vmax=1)
        axes[1, 0].set_title('Errors (Red)', fontweight='bold')
        axes[1, 0].axis('off')
        
        # Class distribution comparison
        gt_unique, gt_counts = np.unique(ground_truth, return_counts=True)
        pred_unique, pred_counts = np.unique(prediction, return_counts=True)
        
        # Create bar plot comparing class distributions
        x_pos = np.arange(len(self.class_names))
        gt_dist = np.zeros(len(self.class_names))
        pred_dist = np.zeros(len(self.class_names))
        
        for cls, count in zip(gt_unique, gt_counts):
            gt_dist[cls] = count / ground_truth.size * 100
        for cls, count in zip(pred_unique, pred_counts):
            pred_dist[cls] = count / prediction.size * 100
        
        width = 0.35
        axes[1, 1].bar(x_pos - width/2, gt_dist, width, label='Ground Truth', alpha=0.7)
        axes[1, 1].bar(x_pos + width/2, pred_dist, width, label='Prediction', alpha=0.7)
        axes[1, 1].set_title('Class Distribution (%)', fontweight='bold')
        axes[1, 1].set_xlabel('Classes')
        axes[1, 1].set_ylabel('Percentage')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels([name[:4] for name in self.class_names], rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Class-wise Dice scores
        dice_scores = []
        for i in range(6):
            key = f'dice_class_{i}_{self.class_names[i]}'
            dice_scores.append(metrics.get(key, 0.0))
        
        bars = axes[1, 2].bar(x_pos, dice_scores, color=[self.class_colors[i]/255.0 for i in range(6)], alpha=0.8)
        axes[1, 2].set_title('Class-wise Dice Scores', fontweight='bold')
        axes[1, 2].set_xlabel('Classes')
        axes[1, 2].set_ylabel('Dice Score')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels([name[:4] for name in self.class_names], rotation=45)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, dice_scores):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved visualization to: {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def evaluate_samples(self, 
                        dataset,
                        num_samples: int = 5, 
                        save_dir: str = None,
                        show_plots: bool = True) -> List[Dict]:
        """Evaluate multiple samples and create visualizations.
        
        Args:
            dataset: Dataset to evaluate
            num_samples: Number of samples to evaluate
            save_dir: Directory to save visualizations
            show_plots: Whether to display plots
            
        Returns:
            List of evaluation results
        """
        print(f"🎯 Evaluating {num_samples} samples...")
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
        
        results = []
        all_metrics = []
        
        # Select samples (spread across dataset)
        indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
        
        for i, idx in enumerate(indices):
            print(f"📊 Processing sample {i+1}/{num_samples} (dataset index {idx})...")
            
            # Get sample
            sample = dataset[idx]
            image = sample['image']  # [C, H, W] tensor
            ground_truth = sample['mask']  # [H, W] tensor
            
            # Run inference
            logits, prediction = self.predict_sample(image)
            
            # Calculate metrics
            metrics = self.calculate_metrics(prediction, ground_truth)
            all_metrics.append(metrics)
            
            # Convert tensors for visualization
            # Handle normalized images (convert back to displayable format)
            if isinstance(image, torch.Tensor):
                img_np = image.cpu().numpy()
                if img_np.ndim == 3 and img_np.shape[0] == 3:
                    img_np = np.transpose(img_np, (1, 2, 0))  # C,H,W -> H,W,C
                
                # Denormalize if needed (Z-score normalized)
                img_min, img_max = img_np.min(), img_np.max()
                img_np = (img_np - img_min) / (img_max - img_min + 1e-7)
                img_np = np.clip(img_np, 0, 1)
            else:
                img_np = image
            
            gt_np = ground_truth.cpu().numpy()
            pred_np = prediction.cpu().numpy()
            
            # Create visualization
            save_path = None
            if save_dir:
                save_path = save_dir / f'evaluation_sample_{i+1}_idx_{idx}.png'
            
            fig = self.visualize_prediction(
                original_image=img_np,
                ground_truth=gt_np,
                prediction=pred_np,
                metrics=metrics,
                sample_idx=idx,
                save_path=str(save_path) if save_path else None,
                show_plot=show_plots
            )
            
            # Store results
            result = {
                'sample_idx': idx,
                'metrics': metrics,
                'figure': fig
            }
            results.append(result)
            
            # Print sample metrics
            print(f"  ✅ Accuracy: {metrics['accuracy']:.3f}, Dice: {metrics['dice_score']:.3f}")
            
            # Print class-wise dice scores
            class_dice_str = []
            for j, name in enumerate(self.class_names):
                score = metrics.get(f'dice_class_{j}_{name}', 0.0)
                class_dice_str.append(f"{name[:4]}: {score:.3f}")
            print(f"     Class Dice: {', '.join(class_dice_str)}")
        
        # Calculate and print overall statistics
        print("\n" + "="*70)
        print("📈 OVERALL EVALUATION RESULTS")
        print("="*70)
        
        # Average metrics
        avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
        avg_dice = np.mean([m['dice_score'] for m in all_metrics])
        
        print(f"Average Accuracy: {avg_accuracy:.4f} ± {np.std([m['accuracy'] for m in all_metrics]):.4f}")
        print(f"Average Dice Score: {avg_dice:.4f} ± {np.std([m['dice_score'] for m in all_metrics]):.4f}")
        
        # Class-wise averages
        print("\nClass-wise Dice Scores:")
        for i, name in enumerate(self.class_names):
            key = f'dice_class_{i}_{name}'
            class_scores = [m.get(key, 0.0) for m in all_metrics]
            avg_score = np.mean(class_scores)
            std_score = np.std(class_scores)
            print(f"  {name:12}: {avg_score:.4f} ± {std_score:.4f}")
        
        print("="*70)
        
        return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Attention U-Net model')
    parser.add_argument('--checkpoint', type=str, 
                       default='lightning_logs/version_1/checkpoints/advanced-epoch=112-val_dice=0.656.ckpt',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to run evaluation on')
    parser.add_argument('--num_samples', type=int, default=8,
                       help='Number of samples to evaluate')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--validation_only', action='store_true', default=True,
                       help='Evaluate only on validation set')
    parser.add_argument('--no_display', action='store_true',
                       help='Do not display plots (useful for headless mode)')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        for version_dir in Path('lightning_logs').glob('version_*'):
            if version_dir.is_dir():
                checkpoint_dir = version_dir / 'checkpoints'
                if checkpoint_dir.exists():
                    print(f"  {version_dir.name}:")
                    for ckpt in checkpoint_dir.glob('*.ckpt'):
                        print(f"    {ckpt}")
        return
    
    print("🚀 ATTENTION U-NET MODEL EVALUATION")
    print("="*50)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {args.device}")
    print(f"Samples: {args.num_samples}")
    print(f"Save directory: {args.save_dir}")
    print("="*50)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(str(checkpoint_path), device=args.device)
    
    # Load dataset
    dataset = evaluator.load_dataset(validation_only=args.validation_only)
    
    # Run evaluation
    results = evaluator.evaluate_samples(
        dataset=dataset,
        num_samples=args.num_samples,
        save_dir=args.save_dir,
        show_plots=not args.no_display
    )
    
    print(f"\n✅ Evaluation complete! Results saved to: {args.save_dir}")
    print(f"📊 Evaluated {len(results)} samples with visualizations")


if __name__ == '__main__':
    # Essential for Windows multiprocessing
    if platform.system() == 'Windows':
        torch.multiprocessing.freeze_support()
    
    main() 