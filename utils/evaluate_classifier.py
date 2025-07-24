#!/usr/bin/env python3
"""
Evaluation script for State-of-the-Art Nucleus Classifier training results.
Shows nucleus images, ground truth labels, and predicted labels with performance metrics.
"""

import sys
import os
import platform
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
warnings.filterwarnings('ignore')

# Import project modules
from utils.nuclei_dataset import load_nuclei_dataset, NucleiClassificationDataset, get_classification_augmentations
from training.train_classifier import StateOfTheArtClassifierLightning
from utils.nuclei_dataset import NucleiDataModule

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

# Patch torch.load globally for compatibility
torch.load = safe_torch_load


class NucleusClassifierEvaluator:
    """Evaluator for nucleus classification model with comprehensive visualizations."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize the classifier evaluator.
        
        Args:
            model_path: Path to the trained classifier checkpoint
            device: Device to run evaluation on
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"🎯 Initializing Classifier Evaluator on device: {self.device}")
        
        # Class names and colors for visualization
        self.class_names = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
        self.class_colors = np.array([
            [255, 0, 0],      # Neoplastic - red
            [0, 255, 0],      # Inflammatory - green  
            [0, 0, 255],      # Connective - blue
            [255, 255, 0],    # Dead - yellow
            [255, 0, 255],    # Epithelial - magenta
        ], dtype=np.uint8)
        
        # Load model
        self.load_model(model_path)
        
        print("✅ Classifier Evaluator initialized successfully!")
    
    def load_model(self, model_path: str):
        """Load the trained classifier model."""
        print(f"📁 Loading classifier model from: {model_path}")
        
        # Load the Lightning checkpoint
        self.lightning_model = StateOfTheArtClassifierLightning.load_from_checkpoint(
            model_path,
            map_location=self.device
        )
        self.lightning_model.eval()
        
        # Extract the actual model
        self.model = self.lightning_model.model
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Architecture: {getattr(self.lightning_model, 'hparams', {}).get('architecture', 'Unknown')}")
        print(f"   Classes: {len(self.class_names)}")
    
    def predict_sample(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict classification for a single nucleus image.
        
        Args:
            image: Input tensor [C, H, W] or [1, C, H, W]
            
        Returns:
            Tuple of (logits, probabilities, prediction)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        image = image.to(self.device)
        
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(image)
            logits = outputs['logits']
            
            # Apply temperature scaling if available
            if hasattr(self.lightning_model, 'temperature'):
                logits = logits / self.lightning_model.temperature
            
            # Calculate probabilities and prediction
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1)
        
        return logits.squeeze(0), probabilities.squeeze(0), prediction.squeeze(0)
    
    def calculate_metrics(self, all_preds: List[int], all_labels: List[int]) -> Dict[str, float]:
        """Calculate comprehensive classification metrics."""
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        
        # Overall accuracy
        accuracy = (y_true == y_pred).mean()
        
        # Detailed classification report
        report = classification_report(
            y_true, y_pred,
            labels=list(range(len(self.class_names))),
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'weighted_f1': report['weighted avg']['f1-score'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_precision': report['weighted avg']['precision'],
            'weighted_recall': report['weighted avg']['recall']
        }
        
        # Per-class metrics - use class names as keys in the report
        for i, class_name in enumerate(self.class_names):
            # Try both numeric string key and class name key
            class_key = None
            if str(i) in report:
                class_key = str(i)
            elif class_name in report:
                class_key = class_name
            
            if class_key:
                metrics[f'f1_{class_name}'] = report[class_key]['f1-score']
                metrics[f'precision_{class_name}'] = report[class_key]['precision'] 
                metrics[f'recall_{class_name}'] = report[class_key]['recall']
            else:
                # Set to 0 if class not found in report (no samples of this class)
                metrics[f'f1_{class_name}'] = 0.0
                metrics[f'precision_{class_name}'] = 0.0
                metrics[f'recall_{class_name}'] = 0.0
        
        return metrics
    
    def visualize_samples(self, 
                         images: List[np.ndarray],
                         true_labels: List[int], 
                         pred_labels: List[int],
                         probabilities: List[np.ndarray],
                         sample_indices: List[int],
                         save_path: str = None,
                         show_plot: bool = True) -> plt.Figure:
        """Create visualization of nucleus samples with predictions."""
        n_samples = len(images)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, min(n_samples, 4), figsize=(16, 8))
        if n_samples == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('Nucleus Classification Results', fontsize=16, fontweight='bold')
        
        for i in range(min(n_samples, 4)):
            # Top row: Original images with predictions
            img = images[i]
            true_label = true_labels[i]
            pred_label = pred_labels[i]
            probs = probabilities[i]
            sample_idx = sample_indices[i]
            
            # Denormalize image for display
            if img.max() <= 1.0:
                img_display = (img * 255).astype(np.uint8)
            else:
                img_display = img.astype(np.uint8)
            
            # Ensure RGB format
            if img_display.shape[0] == 3:  # CHW format
                img_display = np.transpose(img_display, (1, 2, 0))
            
            # Show image with border color indicating correctness
            border_color = 'green' if true_label == pred_label else 'red'
            axes[0, i].imshow(img_display)
            axes[0, i].set_title(f'Sample {sample_idx}\nTrue: {self.class_names[true_label]}\nPred: {self.class_names[pred_label]}',
                               fontsize=10, color=border_color, fontweight='bold')
            axes[0, i].axis('off')
            
            # Add colored border
            for spine in axes[0, i].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
            
            # Bottom row: Probability distributions
            bars = axes[1, i].bar(range(len(self.class_names)), probs, 
                                color=[c/255.0 for c in self.class_colors], alpha=0.7)
            
            # Highlight true and predicted classes
            bars[true_label].set_edgecolor('green')
            bars[true_label].set_linewidth(2)
            bars[pred_label].set_edgecolor('red')
            bars[pred_label].set_linewidth(2)
            
            axes[1, i].set_title(f'Confidence: {probs[pred_label]:.3f}', fontsize=10)
            axes[1, i].set_ylim(0, 1)
            axes[1, i].set_xticks(range(len(self.class_names)))
            axes[1, i].set_xticklabels([name[:4] for name in self.class_names], rotation=45)
            axes[1, i].grid(True, alpha=0.3)
            
            # Add probability text
            for j, (bar, prob) in enumerate(zip(bars, probs)):
                if prob > 0.1:  # Only show significant probabilities
                    axes[1, i].text(bar.get_x() + bar.get_width()/2., prob + 0.01,
                                   f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Hide unused subplots
        for i in range(n_samples, 4):
            axes[0, i].axis('off')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Visualization saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_confusion_matrix(self, all_preds: List[int], all_labels: List[int], 
                              save_path: str = None, show_plot: bool = True) -> plt.Figure:
        """Create and display confusion matrix."""
        cm = confusion_matrix(all_labels, all_preds, labels=range(len(self.class_names)))
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)', fontweight='bold')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax2)
        ax2.set_title('Confusion Matrix (Percentages)', fontweight='bold')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Confusion matrix saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def evaluate(self, 
                dataset_path: str,
                num_samples: int = 8,
                save_dir: str = None,
                show_plots: bool = True,
                use_validation_split: bool = True) -> List[Dict]:
        """
        Evaluate the classifier on a dataset.
        
        Args:
            dataset_path: Path to the nuclei dataset
            num_samples: Number of samples to visualize
            save_dir: Directory to save results
            show_plots: Whether to display plots
            use_validation_split: Whether to use validation split
            
        Returns:
            List of evaluation results
        """
        print("🔬 Starting Classifier Evaluation")
        print("="*50)
        
        # Create save directory
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            print(f"📁 Results will be saved to: {save_dir}")
        
        # Load dataset
        print(f"📊 Loading dataset: {dataset_path}")
        nuclei_instances = load_nuclei_dataset(dataset_path)
        
        # Load dataset split if using validation
        if use_validation_split:
            split_path = Path('lightning_logs/dataset_split.json')
            if split_path.exists():
                with open(split_path, 'r') as f:
                    split_data = json.load(f)
                val_indices = split_data.get('val_indices', [])
                print(f"📊 Using validation split: {len(val_indices)} samples")
                
                # Filter to validation samples
                sample_groups = {}
                for i, nucleus in enumerate(nuclei_instances):
                    sample_idx = nucleus.get('sample_idx', nucleus.get('global_instance_id', i // 10))
                    if sample_idx not in sample_groups:
                        sample_groups[sample_idx] = []
                    sample_groups[sample_idx].append(i)
                
                val_nuclei_indices = []
                for sample_idx in val_indices:
                    if sample_idx in sample_groups:
                        val_nuclei_indices.extend(sample_groups[sample_idx])
                
                nuclei_instances = [nuclei_instances[i] for i in val_nuclei_indices]
                print(f"📊 Validation set: {len(nuclei_instances)} nuclei")
        
        # Create dataset without augmentations for evaluation
        dataset = NucleiClassificationDataset(
            nuclei_instances=nuclei_instances,
            augmentations=None,  # No augmentations for evaluation
            normalize=True
        )
        
        # Select random samples for visualization
        np.random.seed(42)  # For reproducible results
        total_samples = len(dataset)
        if num_samples > total_samples:
            num_samples = total_samples
        
        sample_indices = np.random.choice(total_samples, size=num_samples, replace=False)
        sample_indices = sorted(sample_indices)  # Sort for consistent ordering
        
        print(f"🎯 Evaluating {num_samples} samples from {total_samples} total")
        
        # Collect all predictions and labels
        all_preds = []
        all_labels = []
        all_probs = []
        
        # Visualization samples
        vis_images = []
        vis_labels = []
        vis_preds = []
        vis_probs = []
        vis_indices = []
        
        print("\n🔍 Running inference...")
        for i, idx in enumerate(sample_indices):
            sample = dataset[idx]
            image = sample['image']
            true_label = sample['label'].item()
            
            # Run inference
            logits, probabilities, prediction = self.predict_sample(image)
            pred_label = prediction.item()
            probs = probabilities.cpu().numpy()
            
            # Store for metrics
            all_preds.append(pred_label)
            all_labels.append(true_label)
            all_probs.append(probs)
            
            # Store for visualization
            img_np = image.cpu().numpy()
            vis_images.append(img_np)
            vis_labels.append(true_label)
            vis_preds.append(pred_label)
            vis_probs.append(probs)
            vis_indices.append(idx)
            
            # Print sample result
            status = "✅" if true_label == pred_label else "❌"
            confidence = probs[pred_label]
            print(f"  {status} Sample {idx}: True={self.class_names[true_label]}, "
                  f"Pred={self.class_names[pred_label]} (conf={confidence:.3f})")
        
        # Calculate overall metrics
        metrics = self.calculate_metrics(all_preds, all_labels)
        
        # Print results
        print("\n" + "="*50)
        print("📈 EVALUATION RESULTS")
        print("="*50)
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Weighted F1 Score: {metrics['weighted_f1']:.4f}")
        print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
        print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
        print(f"Weighted Recall: {metrics['weighted_recall']:.4f}")
        
        print("\nPer-Class Results:")
        for class_name in self.class_names:
            f1 = metrics.get(f'f1_{class_name}', 0)
            precision = metrics.get(f'precision_{class_name}', 0)
            recall = metrics.get(f'recall_{class_name}', 0)
            print(f"  {class_name:12}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
        
        # Create visualizations
        print("\n🎨 Creating visualizations...")
        
        # Sample predictions visualization
        samples_save_path = None
        if save_dir:
            samples_save_path = save_dir / 'classifier_samples.png'
        
        samples_fig = self.visualize_samples(
            vis_images, vis_labels, vis_preds, vis_probs, vis_indices,
            save_path=samples_save_path, show_plot=show_plots
        )
        
        # Confusion matrix
        cm_save_path = None
        if save_dir:
            cm_save_path = save_dir / 'confusion_matrix.png'
        
        cm_fig = self.create_confusion_matrix(
            all_preds, all_labels,
            save_path=cm_save_path, show_plot=show_plots
        )
        
        # Save detailed results
        if save_dir:
            results_path = save_dir / 'detailed_results.json'
            
            # Convert metrics to JSON-serializable format
            json_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    json_metrics[key] = float(value)
                elif isinstance(value, np.ndarray):
                    json_metrics[key] = value.tolist()
                else:
                    json_metrics[key] = value
            
            detailed_results = {
                'model_path': str(getattr(self.lightning_model, 'training_state', {}).get('resume_path', 'Unknown')),
                'total_samples_evaluated': len(all_preds),
                'metrics': json_metrics,
                'per_sample_results': [
                    {
                        'sample_idx': int(idx),
                        'true_label': int(true_label),
                        'true_class': self.class_names[true_label],
                        'pred_label': int(pred_label),
                        'pred_class': self.class_names[pred_label],
                        'confidence': float(probs[pred_label]),
                        'correct': bool(true_label == pred_label),
                        'probabilities': {
                            self.class_names[i]: float(prob) 
                            for i, prob in enumerate(probs)
                        }
                    }
                    for idx, true_label, pred_label, probs in zip(vis_indices, vis_labels, vis_preds, vis_probs)
                ]
            }
            
            with open(results_path, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            print(f"💾 Detailed results saved to: {results_path}")
        
        print("="*50)
        
        return {
            'metrics': metrics,
            'samples_figure': samples_fig,
            'confusion_matrix_figure': cm_fig,
            'sample_results': list(zip(vis_indices, vis_labels, vis_preds, vis_probs))
        }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate State-of-the-Art Nucleus Classifier')
    parser.add_argument('--checkpoint', type=str, 
                       default='lightning_logs/classifier/classifier_efficientnet_b3_20250724_002315/version_0/checkpoints/classifier-epoch=04-val_f1=0.789.ckpt',
                       help='Path to classifier checkpoint')
    parser.add_argument('--dataset', type=str, default='nuclei_dataset.pkl',
                       help='Path to nuclei dataset')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to run evaluation on')
    parser.add_argument('--num_samples', type=int, default=8,
                       help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='evaluation_results/classifier',
                       help='Directory to save evaluation results')
    parser.add_argument('--validation_only', action='store_true', default=True,
                       help='Evaluate only on validation set')
    parser.add_argument('--no_display', action='store_true',
                       help='Do not display plots (useful for headless mode)')
    
    args = parser.parse_args()
    
    print("🚀 STATE-OF-THE-ART NUCLEUS CLASSIFIER EVALUATION")
    print("="*60)
    print(f"Model: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Samples: {args.num_samples}")
    print(f"Validation only: {args.validation_only}")
    print("="*60)
    
    # Create evaluator
    evaluator = NucleusClassifierEvaluator(
        model_path=args.checkpoint,
        device=args.device
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        dataset_path=args.dataset,
        num_samples=args.num_samples,
        save_dir=args.save_dir,
        show_plots=not args.no_display,
        use_validation_split=args.validation_only
    )
    
    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main() 