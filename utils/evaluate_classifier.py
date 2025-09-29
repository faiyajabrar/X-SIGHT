#!/usr/bin/env python3
"""
Evaluation script for Nucleus Classifier training results.
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
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

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
        
        # Grad-CAM buffers
        self._gc_activations = None
        self._gc_gradients = None
        self._gc_layer_idx = -1  # default last
        self._gc_blur_ksize = 0
        self._gc_blur_sigma = 0.0
        self._register_gradcam_hooks()
    
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
        
        # Simple wrappers for SHAP: logits and probabilities
        class _LogitsWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
            def forward(self, x):
                out = self.base_model(x)
                return out['logits']
        class _ProbsWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
            def forward(self, x):
                out = self.base_model(x)
                return torch.softmax(out['logits'], dim=1)
        self.model_logits = _LogitsWrapper(self.model)
        self.model_probs = _ProbsWrapper(self.model)
    
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

    # ----------------------
    # Grad-CAM
    # ----------------------
    def _register_gradcam_hooks(self):
        """Register hooks on the final convolutional layer to capture activations."""
        if not hasattr(self, 'model') or not hasattr(self.model, 'backbone'):
            return
        
        # Helper to resolve a submodule path like "blocks.6"
        def _get_submodule(root: torch.nn.Module, path: str):
            if hasattr(root, 'get_submodule'):
                try:
                    return root.get_submodule(path)
                except Exception:
                    pass
            module = root
            for name in path.split('.'):
                module = getattr(module, name)
            return module
        
        # Try using timm feature_info to identify the last feature-producing module
        target_module = None
        try:
            fi = getattr(self.model.backbone, 'feature_info', None)
            if fi is not None:
                # FeatureInfo supports indexing to dicts with 'module' or 'name'
                # Select by configured layer index (supports negatives)
                idx = self._gc_layer_idx if isinstance(self._gc_layer_idx, int) else -1
                nfi = len(fi)
                if idx < 0:
                    idx = nfi + idx
                idx = max(0, min(idx, nfi - 1))
                last_info = fi[idx]
                module_path = last_info.get('module', None) or last_info.get('name', None)
                if isinstance(module_path, str):
                    # Search on backbone and backbone.body (timm FeatureListNet)
                    if hasattr(self.model.backbone, 'body'):
                        try:
                            target_module = _get_submodule(self.model.backbone.body, module_path)
                        except Exception:
                            target_module = None
                    if target_module is None:
                        try:
                            target_module = _get_submodule(self.model.backbone, module_path)
                        except Exception:
                            target_module = None
        except Exception:
            target_module = None
        
        # Fallback: pick the last Conv2d in the backbone
        if target_module is None:
            conv_layers = [m for m in self.model.backbone.modules() if isinstance(m, torch.nn.Conv2d)]
            if len(conv_layers) > 0:
                target_module = conv_layers[-1]
        
        if target_module is None:
            # As a last resort, keep the old backbone-level hook
            def backbone_forward_hook(module, inp, out):
                feat = out[-1] if isinstance(out, (list, tuple)) else out
                self._gc_activations = feat
            try:
                if hasattr(self, '_gc_forward_handle'):
                    self._gc_forward_handle.remove()
            except Exception:
                pass
            self._gc_forward_handle = self.model.backbone.register_forward_hook(backbone_forward_hook)
            self._gc_target_module_name = 'backbone_output'
            return
        
        def forward_hook(module, inp, out):
            # out should be a tensor [B,C,H,W]
            self._gc_activations = out
        
        # Remove previous hooks if any
        try:
            if hasattr(self, '_gc_forward_handle'):
                self._gc_forward_handle.remove()
        except Exception:
            pass
        
        self._gc_forward_handle = target_module.register_forward_hook(forward_hook)
        self._gc_target_module_name = type(target_module).__name__

    def generate_gradcam(self, image: torch.Tensor, target_class: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM heatmap and overlay for a single image tensor [C,H,W].
        Returns (heatmap_uint8, overlay_rgb_uint8).
        """
        was_training = self.model.training
        self.model.eval()
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        
        # Forward pass with gradients enabled
        for p in self.model.parameters():
            p.requires_grad_(True)
        
        self.model.zero_grad(set_to_none=True)
        self._gc_activations = None
        
        logits = self.model(image)['logits']
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())
        score = logits[:, target_class].sum()
        
        # Prefer autograd.grad to get gradients w.r.t. captured activations
        gradients = None
        activations = self._gc_activations
        if activations is not None and isinstance(activations, torch.Tensor):
            try:
                gradients = torch.autograd.grad(score, activations, retain_graph=True, allow_unused=True)[0]
            except Exception:
                gradients = None
        
        if gradients is None or activations is None:
            # Fallback: re-run and use .retain_grad + backward
            feats = None
            try:
                # Try to recompute the hooked module output
                logits = self.model(image)['logits']
                score = logits[:, target_class].sum()
                if self._gc_activations is not None and isinstance(self._gc_activations, torch.Tensor):
                    self._gc_activations.retain_grad()
                    score.backward(retain_graph=True)
                    gradients = self._gc_activations.grad
                    activations = self._gc_activations
                else:
                    feats = self.model.backbone(image)
                    feats = feats[-1] if isinstance(feats, (list, tuple)) else feats
                    feats.retain_grad()
                    self.model.zero_grad(set_to_none=True)
                    logits = self.model(image)['logits']
                    score = logits[:, target_class].sum()
                    score.backward(retain_graph=True)
                    activations = feats
                    gradients = feats.grad
            except Exception:
                pass
        
        if activations is None or gradients is None:
            # As a last guard, return a zero heatmap of input size
            _, _, H, W = image.shape
            heatmap_uint8 = np.zeros((H, W), dtype=np.uint8)
            img = image.squeeze(0).detach().cpu().numpy()
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            img_vis = (img * 255.0).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 + 1e-3 else img.clip(0, 255).astype(np.uint8)
            overlay_bgr = cv2.addWeighted(cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR), 0.5, cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET), 0.5, 0)
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            if was_training:
                self.model.train()
            return heatmap_uint8, overlay_rgb
        
        activations = activations.detach()
        gradients = gradients.detach()
        
        # Compute weights: global average pooling over spatial dims
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = torch.relu(cam)
        
        # Fallback to LayerCAM-style if CAM is (near) constant or zero
        cam_min = cam.amin(dim=(2,3), keepdim=True)
        cam_max = cam.amax(dim=(2,3), keepdim=True)
        if torch.all((cam_max - cam_min) < 1e-6):
            cam = torch.relu((gradients * activations).sum(dim=1, keepdim=True))
            cam_min = cam.amin(dim=(2,3), keepdim=True)
            cam_max = cam.amax(dim=(2,3), keepdim=True)
        
        # Normalize each CAM to 0-1
        cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam_np = cam_norm.squeeze().cpu().numpy()  # [H,W]
        
        # Resize to input size (224x224 expected)
        _, _, H, W = image.shape
        heatmap = cv2.resize(cam_np, (W, H))
        # Optional smoothing
        if isinstance(self._gc_blur_ksize, int) and self._gc_blur_ksize > 0:
            ks = self._gc_blur_ksize
            if ks % 2 == 0:
                ks += 1
            heatmap = cv2.GaussianBlur(heatmap, (ks, ks), self._gc_blur_sigma if self._gc_blur_sigma > 0 else 0)
            # Renormalize after blur
            minv, maxv = float(heatmap.min()), float(heatmap.max())
            if maxv - minv > 1e-8:
                heatmap = (heatmap - minv) / (maxv - minv)
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
        
        # Prepare original image for overlay (denormalize if needed)
        img = image.squeeze(0).detach().cpu().numpy()  # [C,H,W]
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        # If normalized [0,1] or ImageNet, just clamp to 0-1 for visualization
        img_vis = img.copy()
        if img_vis.max() <= 1.0 + 1e-3:
            img_vis = (img_vis * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img_vis = img_vis.clip(0, 255).astype(np.uint8)
        
        overlay_bgr = cv2.addWeighted(cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR), 0.5, heatmap_color, 0.5, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        
        if was_training:
            self.model.train()
        return heatmap_uint8, overlay_rgb

    # ----------------------
    # SHAP explanations
    # ----------------------
    def _create_shap_explainer(self, background: torch.Tensor):
        """Create a SHAP GradientExplainer with a small background batch."""
        if not HAS_SHAP:
            return None
        background = background.to(self.device).float()
        self.model_probs.eval()
        explainer = shap.GradientExplainer(self.model_probs, background)
        return explainer
    
    def generate_shap_overlay(self, image: torch.Tensor, pred_class: int, explainer, nsamples: int = 20) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
        """
        Compute SHAP values for a single image [C,H,W], return overlay RGB uint8, and simple stats.
        """
        if not HAS_SHAP or explainer is None:
            return None, {}, None
        if image.dim() == 3:
            image_b = image.unsqueeze(0)
        else:
            image_b = image
        # Use float32 and enable grad on inputs for gradient-based SHAP
        image_b = image_b.to(self.device).float().detach()
        image_b.requires_grad_(True)
        
        # Prefer ranked output=1 (fast and avoids indexing issues); fallback to full outputs
        explained_class = int(pred_class)
        try:
            result = explainer.shap_values(image_b, nsamples=nsamples, ranked_outputs=1, output_rank_order='max')
            if isinstance(result, tuple) and len(result) == 2:
                shap_values, indexes = result
                try:
                    explained_class = int(np.array(indexes)[0][0])
                except Exception:
                    explained_class = int(pred_class)
            else:
                shap_values = result
        except Exception:
            shap_values = explainer.shap_values(image_b, nsamples=nsamples, ranked_outputs=None)
        
        # Extract SHAP attributions for the (single) sample
        sv = None
        if isinstance(shap_values, list):
            if len(shap_values) == 1:
                sv_arr = shap_values[0]
                if isinstance(sv_arr, torch.Tensor):
                    sv_arr = sv_arr.detach().cpu().numpy()
                sv = sv_arr[0]
            else:
                idx = explained_class if 0 <= explained_class < len(shap_values) else int(pred_class)
                idx = max(0, min(idx, len(shap_values) - 1))
                sv_arr = shap_values[idx]
                if isinstance(sv_arr, torch.Tensor):
                    sv_arr = sv_arr.detach().cpu().numpy()
                sv = sv_arr[0]
        elif isinstance(shap_values, np.ndarray):
            sv = shap_values[0]
        elif isinstance(shap_values, torch.Tensor):
            sv = shap_values[0].detach().cpu().numpy()
        
        if sv is None:
            return None, {}, None
        
        # Ensure channel-first [C,H,W]
        if sv.ndim == 4:
            # Try to squeeze singleton batch/chan dims
            sv = np.squeeze(sv)
        if sv.ndim == 3 and sv.shape[0] in (1, 3):
            sv_chw = sv
        elif sv.ndim == 3 and sv.shape[-1] in (1, 3):
            sv_chw = np.transpose(sv, (2, 0, 1))
        elif sv.ndim == 2:
            # Single-channel 2D map
            sv_chw = np.expand_dims(sv, 0)
        else:
            return None, {}, None
        
        # Aggregate across channels and normalize
        sv_abs = np.abs(sv_chw).sum(axis=0).astype(np.float32)  # [H,W]
        H, W = image_b.shape[-2], image_b.shape[-1]
        if sv_abs.shape != (H, W):
            sv_abs = cv2.resize(sv_abs, (W, H))
        if sv_abs.max() > 0:
            sv_abs = sv_abs / (sv_abs.max() + 1e-8)
        heatmap_uint8 = np.uint8(255 * sv_abs)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
        
        # Prepare original image
        img = image_b[0].detach().cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img_vis = img.copy()
        if img_vis.max() <= 1.0 + 1e-3:
            img_vis = (img_vis * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img_vis = img_vis.clip(0, 255).astype(np.uint8)
        if heatmap_color.shape[:2] != img_vis.shape[:2]:
            heatmap_color = cv2.resize(heatmap_color, (img_vis.shape[1], img_vis.shape[0]))
        
        overlay_bgr = cv2.addWeighted(cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        
        # Simple text attribution summary
        pos_contrib = float(np.mean(sv_chw[sv_chw > 0])) if np.any(sv_chw > 0) else 0.0
        neg_contrib = float(np.mean(sv_chw[sv_chw < 0])) if np.any(sv_chw < 0) else 0.0
        stats = {
            'mean_positive_contribution': pos_contrib,
            'mean_negative_contribution': neg_contrib,
            'mean_abs_contribution': float(np.mean(np.abs(sv_chw))),
            'explained_class': int(explained_class)
        }
        return overlay_rgb, stats, heatmap_uint8
    
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
                use_validation_split: bool = True,
                enable_gradcam: bool = True,
                enable_shap: bool = True,
                shap_bg: int = 8,
                shap_nsamples: int = 20,
                gradcam_layer_idx: int = -2,
                gradcam_blur_ksize: int = 3,
                gradcam_blur_sigma: float = 0.0) -> List[Dict]:
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
            explanations_dir = save_dir / 'explanations'
            explanations_dir.mkdir(exist_ok=True, parents=True)
        
        # Load dataset
        print(f"📊 Loading dataset: {dataset_path}")
        nuclei_instances = load_nuclei_dataset(dataset_path)
        
        # Load dataset split if using validation
        if use_validation_split:
            split_path = Path('lightning_logs/classifier/dataset_split.json')
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
        # Configure Grad-CAM layer and smoothing
        self._gc_layer_idx = gradcam_layer_idx
        self._gc_blur_ksize = gradcam_blur_ksize
        self._gc_blur_sigma = gradcam_blur_sigma
        self._register_gradcam_hooks()
        # Prepare SHAP explainer (small background set)
        explainer = None
        if enable_shap and HAS_SHAP:
            bg_count = min(max(1, shap_bg), total_samples)
            bg_indices = np.random.choice(total_samples, size=bg_count, replace=False)
            bg_tensors = []
            for j in bg_indices:
                bg_sample = dataset[int(j)]
                bg_tensors.append(bg_sample['image'])
            background = torch.stack(bg_tensors)
            explainer = self._create_shap_explainer(background)
            if explainer is None:
                print("⚠️  SHAP not available; skipping SHAP explanations.")
                enable_shap = False
        elif enable_shap and not HAS_SHAP:
            print("⚠️  'shap' package not installed; skipping SHAP explanations.")
            enable_shap = False
        
        shap_summaries = []
        gradcam_saved = 0
        shap_saved = 0
        quality_records = []
        
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
            
            # Grad-CAM overlay
            if save_dir and enable_gradcam:
                try:
                    heatmap_uint8, overlay_rgb = self.generate_gradcam(image, pred_label)
                    out_path = explanations_dir / f'gradcam_sample_{idx}.png'
                    cv2.imwrite(str(out_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
                    # Also save raw heatmap for verification
                    cv2.imwrite(str(explanations_dir / f'gradcam_heatmap_{idx}.png'), heatmap_uint8)
                    gradcam_saved += 1
                    # Quality metric: center-mass concentration for Grad-CAM
                    ghm = heatmap_uint8.astype(np.float32) / 255.0
                    Hc, Wc = ghm.shape
                    cy, cx = Hc // 2, Wc // 2
                    hy, hx = int(Hc * 0.3), int(Wc * 0.3)  # central 60% box
                    y1, y2 = max(0, cy - hy), min(Hc, cy + hy)
                    x1, x2 = max(0, cx - hx), min(Wc, cx + hx)
                    total = float(ghm.sum()) + 1e-8
                    gradcam_center_mass = float(ghm[y1:y2, x1:x2].sum() / total)
                except Exception as e:
                    print(f"⚠️  Grad-CAM failed for sample {idx}: {e}")
                    gradcam_center_mass = None
                except Exception as e:
                    print(f"⚠️  Grad-CAM failed for sample {idx}: {e}")
            
            # SHAP overlay and summary
            if save_dir and enable_shap and explainer is not None:
                try:
                    shap_overlay, shap_stats, shap_heat = self.generate_shap_overlay(image, pred_label, explainer, nsamples=shap_nsamples)
                    if shap_overlay is not None:
                        out_path = explanations_dir / f'shap_overlay_sample_{idx}.png'
                        cv2.imwrite(str(out_path), cv2.cvtColor(shap_overlay, cv2.COLOR_RGB2BGR))
                        if shap_heat is not None:
                            cv2.imwrite(str(explanations_dir / f'shap_heatmap_{idx}.png'), shap_heat)
                        shap_summaries.append({
                            'sample_idx': int(idx),
                            'true_label': int(true_label),
                            'true_class': self.class_names[true_label],
                            'pred_label': int(pred_label),
                            'pred_class': self.class_names[pred_label],
                            'confidence': float(confidence),
                            **shap_stats
                        })
                        shap_saved += 1
                        # Quality metric: center-mass for SHAP and correlation with Grad-CAM
                        shap_map = shap_heat.astype(np.float32) / 255.0 if shap_heat is not None else None
                        if shap_map is not None:
                            Hs, Ws = shap_map.shape
                            cy, cx = Hs // 2, Ws // 2
                            hy, hx = int(Hs * 0.3), int(Ws * 0.3)
                            y1, y2 = max(0, cy - hy), min(Hs, cy + hy)
                            x1, x2 = max(0, cx - hx), min(Ws, cx + hx)
                            total_s = float(shap_map.sum()) + 1e-8
                            shap_center_mass = float(shap_map[y1:y2, x1:x2].sum() / total_s)
                        else:
                            shap_center_mass = None
                        # Correlation SHAP vs Grad-CAM (if both available)
                        shap_gradcam_corr = None
                        try:
                            if shap_map is not None and 'ghm' in locals() and ghm is not None:
                                a = shap_map.flatten()
                                b = ghm.flatten()
                                if a.std() > 1e-6 and b.std() > 1e-6:
                                    shap_gradcam_corr = float(np.corrcoef(a, b)[0, 1])
                        except Exception:
                            shap_gradcam_corr = None
                    else:
                        shap_center_mass = None
                except Exception as e:
                    print(f"⚠️  SHAP failed for sample {idx}: {e}")
                    shap_center_mass = None
                    shap_gradcam_corr = None

            # Record quality metrics for this sample
            quality_records.append({
                'sample_idx': int(idx),
                'pred_class': self.class_names[pred_label],
                'confidence': float(confidence),
                'gradcam_center_mass': gradcam_center_mass if 'gradcam_center_mass' in locals() else None,
                'shap_center_mass': shap_center_mass if 'shap_center_mass' in locals() else None,
                'shap_gradcam_corr': shap_gradcam_corr if 'shap_gradcam_corr' in locals() else None
            })
        
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
            
            # Save SHAP text attribution summary if available
            if len([1 for _ in shap_summaries]) > 0:
                shap_path = save_dir / 'shap_text_summary.json'
                with open(shap_path, 'w') as f:
                    json.dump(shap_summaries, f, indent=2)
                print(f"💾 SHAP text attribution summary saved to: {shap_path}")
            # Final explanation summary
            print(f"🖼️  Saved Grad-CAM overlays: {gradcam_saved} → {explanations_dir}")
            if enable_shap:
                if shap_saved > 0:
                    print(f"🖼️  Saved SHAP overlays: {shap_saved} → {explanations_dir}")
                else:
                    print("ℹ️  SHAP enabled but no overlays were produced. Ensure 'shap' is installed and consider lowering --shap_bg/--shap_nsamples.")
            # Save and print explanation quality metrics
            quality_path = save_dir / 'explanations_quality.json'
            with open(quality_path, 'w') as f:
                json.dump(quality_records, f, indent=2)
            # Summary stats
            try:
                gcm_vals = [q['gradcam_center_mass'] for q in quality_records if isinstance(q['gradcam_center_mass'], (float, int))]
                scm_vals = [q['shap_center_mass'] for q in quality_records if isinstance(q['shap_center_mass'], (float, int))]
                corr_vals = [q['shap_gradcam_corr'] for q in quality_records if isinstance(q['shap_gradcam_corr'], (float, int))]
                if gcm_vals:
                    print(f"📊 Grad-CAM center-mass avg: {np.mean(gcm_vals):.3f}")
                if scm_vals:
                    print(f"📊 SHAP center-mass avg: {np.mean(scm_vals):.3f}")
                if corr_vals:
                    print(f"📊 SHAP–GradCAM correlation avg: {np.mean(corr_vals):.3f}")
            except Exception:
                pass
        
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
                       default='lightning_logs/classifier/classifier_efficientnet_b3_20250727_002713/version_0/checkpoints/classifier-epoch=19-val_f1=0.806.ckpt',
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
    parser.add_argument('--enable_gradcam', action='store_true', default=True,
                        help='Enable Grad-CAM overlays for samples')
    parser.add_argument('--enable_shap', action='store_true', default=True,
                        help='Enable SHAP explanations for samples')
    parser.add_argument('--shap_bg', type=int, default=8,
                        help='Background batch size for SHAP explainer')
    parser.add_argument('--shap_nsamples', type=int, default=20,
                        help='Number of samples for SHAP GradientExplainer')
    parser.add_argument('--gradcam_layer_idx', type=int, default=-2,
                        help='Feature layer index from timm feature_info to compute Grad-CAM (e.g., -2, -1)')
    parser.add_argument('--gradcam_blur_ksize', type=int, default=3,
                        help='Gaussian blur kernel size for Grad-CAM heatmap (0 to disable)')
    parser.add_argument('--gradcam_blur_sigma', type=float, default=0.0,
                        help='Gaussian blur sigma for Grad-CAM heatmap')
    
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
        use_validation_split=args.validation_only,
        enable_gradcam=args.enable_gradcam,
        enable_shap=args.enable_shap,
        shap_bg=args.shap_bg,
        shap_nsamples=args.shap_nsamples,
        gradcam_layer_idx=args.gradcam_layer_idx,
        gradcam_blur_ksize=args.gradcam_blur_ksize,
        gradcam_blur_sigma=args.gradcam_blur_sigma
    )
    
    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main() 