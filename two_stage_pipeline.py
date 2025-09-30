"""
Two-Stage Nuclei Analysis Pipeline

This script implements the complete two-stage pipeline for nuclei analysis:
1. Stage 1: Segmentation - Attention U-Net locates nuclei and gives rough type prediction
2. Stage 2: Classification - EfficientNet classifier provides fine-grained nucleus classification

Usage:
    python two_stage_pipeline.py --segmentation_model path/to/segmentation.ckpt --classifier_model path/to/classifier.ckpt --input_image image.png
"""

import sys
import os
import platform
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Import project modules
from training.train_segmentation import AdvancedAttentionModel
from training.train_classifier import StateOfTheArtClassifierLightning
from utils.nuclei_extraction import extract_nuclei_from_prediction, visualize_extracted_nuclei, nucleus_statistics
from utils.pannuke_dataset import PanNukeDataset, CLAHETransform, ZScoreTransform
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Fix for Windows multiprocessing
if platform.system() == 'Windows':
    torch.multiprocessing.set_start_method('spawn', force=True)


class TwoStageNucleiPipeline:
    """
    Complete two-stage pipeline for nuclei analysis combining segmentation and classification.
    """
    
    def __init__(
        self,
        segmentation_model_path: str,
        classifier_model_path: str,
        device: str = 'cuda',
        segmentation_threshold: float = 0.5,
        min_nucleus_area: int = 50,
        max_nucleus_area: int = 5000,
        context_padding: int = 32,
        # Explainability
        enable_gradcam: bool = True,
        enable_shap: bool = True,
        gradcam_layer_idx: int = -2,
        gradcam_blur_ksize: int = 3,
        gradcam_blur_sigma: float = 0.0,
        shap_bg: int = 8,
        shap_nsamples: int = 20,
        # SHAP visualization controls
        shap_overlay_mode: str = 'masked',
        shap_top_percent: float = 0.15,
        shap_alpha: float = 0.6
    ):
        """
        Initialize the two-stage pipeline.
        
        Args:
            segmentation_model_path: Path to trained segmentation model checkpoint
            classifier_model_path: Path to trained nucleus classifier checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            segmentation_threshold: Confidence threshold for segmentation
            min_nucleus_area: Minimum nucleus area in pixels
            max_nucleus_area: Maximum nucleus area in pixels
            context_padding: Padding around nucleus for classification
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.segmentation_threshold = segmentation_threshold
        self.min_nucleus_area = min_nucleus_area
        self.max_nucleus_area = max_nucleus_area
        self.context_padding = context_padding
        
        # Explainability settings
        self.enable_gradcam = enable_gradcam
        self.enable_shap = enable_shap and HAS_SHAP
        self.gradcam_layer_idx = gradcam_layer_idx
        self.gradcam_blur_ksize = gradcam_blur_ksize
        self.gradcam_blur_sigma = gradcam_blur_sigma
        self.shap_bg = shap_bg
        self.shap_nsamples = shap_nsamples
        # SHAP overlay controls
        self.shap_overlay_mode = shap_overlay_mode
        self.shap_top_percent = shap_top_percent
        self.shap_alpha = shap_alpha
        
        # Class names and colors (Background at index 0, excluded from classification)
        self.class_names = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
        self.classification_classes = self.class_names[1:]  # 5 classes for classification (excluding background)
        
        self.class_colors = np.array([
            [0, 0, 0],        # 0: Background - black (excluded from classification)
            [255, 0, 0],      # 1: Neoplastic - red
            [0, 255, 0],      # 2: Inflammatory - green  
            [0, 0, 255],      # 3: Connective - blue
            [255, 255, 0],    # 4: Dead - yellow
            [255, 0, 255],    # 5: Epithelial - magenta
        ], dtype=np.uint8)
        
        print(f"[INFO] Initializing Two-Stage Nuclei Pipeline on device: {self.device}")
        print(f"[INFO] Classification: {len(self.classification_classes)} nucleus types (Background excluded)")
        print(f"   Classes: {', '.join(self.classification_classes)}")
        
        # Load models
        self.load_segmentation_model(segmentation_model_path)
        self.load_classifier_model(classifier_model_path)
        
        # Set up preprocessing
        self.setup_preprocessing()
        
        # Grad-CAM buffers
        self._gc_activations = None
        self._gc_forward_handle = None
        self._register_gradcam_hooks()
        
        # Placeholder output dir for explanations (set during analyze)
        self._current_output_dir: Optional[Path] = None
        self._shap_summaries: List[Dict] = []
        self._explain_quality_records: List[Dict] = []
        
        print("[INFO] Two-Stage Pipeline initialized successfully!")
    
    def load_segmentation_model(self, model_path: str):
        """Load the segmentation model (Stage 1)."""
        print(f"[INFO] Loading segmentation model from: {model_path}")
        
        # Load the trained Attention U-Net model
        self.segmentation_model = AdvancedAttentionModel.load_from_checkpoint(
            model_path,
            map_location=self.device,
            resume_path=None
        )
        self.segmentation_model = self.segmentation_model.to(self.device)
        self.segmentation_model.eval()
        
        print("[INFO] Segmentation model loaded successfully!")
    
    def load_classifier_model(self, model_path: str):
        """Load the state-of-the-art nucleus classifier model (Stage 2)."""
        print(f"[INFO] Loading classifier model from: {model_path}")
        
        # Load the trained state-of-the-art nucleus classifier
        # This loads the Lightning checkpoint and extracts the actual model
        lightning_model = StateOfTheArtClassifierLightning.load_from_checkpoint(
            model_path,
            map_location=self.device,
            resume_path=None
        )
        
        # Extract the actual classifier model from the Lightning wrapper
        self.classifier_model = lightning_model.model
        self.classifier_model.eval()
        
        # Verify model is configured for 5 classes (excluding background)
        expected_classes = len(self.classification_classes)
        if hasattr(lightning_model, 'num_classes'):
            assert lightning_model.num_classes == expected_classes, f"Classifier should have {expected_classes} classes, got {lightning_model.num_classes}"
            print(f"[INFO] Verified: Classifier configured for {lightning_model.num_classes} classes (excluding background)")
            print(f"   Classification classes: {', '.join(self.classification_classes)}")
        
        print("[INFO] classifier model loaded successfully!")
        # Try to load training_state.json path hint if present
        try:
            ts_path = Path('lightning_logs/classifier/training_state.json')
            if ts_path.exists():
                print(f"[INFO] Training state file found: {ts_path}")
        except Exception:
            pass
        
        # SHAP wrappers for probabilities
        class _ProbsWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
            def forward(self, x):
                out = self.base_model(x)
                return torch.softmax(out['logits'], dim=1)
        self._model_probs = _ProbsWrapper(self.classifier_model)
        
        # End of load_classifier_model

    # ----------------------
    # SHAP helper methods (aligned with evaluator)
    # ----------------------
    def _create_shap_explainer(self, background: torch.Tensor):
        if not HAS_SHAP:
            return None
        try:
            background = background.to(self.device).float()
            self._model_probs.eval()
            return shap.GradientExplainer(self._model_probs, background)
        except Exception:
            return None

    def _ensure_chw(self, image: torch.Tensor) -> torch.Tensor:
        return image if image.dim() == 4 else image.unsqueeze(0)

    def _denormalize_for_display(self, img_chw: np.ndarray) -> np.ndarray:
        """Best-effort convert possibly ImageNet-normalized CHW float to HWC uint8 RGB."""
        arr = img_chw.astype(np.float32)
        # CHW -> HWC if needed
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr_hwc = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
            arr_hwc = arr
        else:
            arr_hwc = arr
        # ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        # If appears to be in [0,1], scale; else denormalize using ImageNet
        if (arr_hwc.min() >= -1e-3) and (arr_hwc.max() <= 1.0 + 1e-3):
            vis = arr_hwc * 255.0
        else:
            if arr_hwc.shape[-1] == 1:
                vis = arr_hwc * 255.0
            else:
                vis = ((arr_hwc * std) + mean) * 255.0
        vis = np.clip(vis, 0, 255).astype(np.uint8)
        if vis.ndim == 3 and vis.shape[-1] == 1:
            vis = np.repeat(vis, 3, axis=-1)
        return vis

    def _generate_shap_overlay(self, image: torch.Tensor, pred_class: int, explainer, nsamples: int = 20):
        if not HAS_SHAP or explainer is None:
            return None, {}, None
        image_b = self._ensure_chw(image).to(self.device).float().detach()
        image_b.requires_grad_(True)
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
        sv = None
        if isinstance(shap_values, list):
            if len(shap_values) == 1:
                arr = shap_values[0]
                if isinstance(arr, torch.Tensor):
                    arr = arr.detach().cpu().numpy()
                sv = arr[0]
            else:
                idx = explained_class if 0 <= explained_class < len(shap_values) else int(pred_class)
                idx = max(0, min(idx, len(shap_values) - 1))
                arr = shap_values[idx]
                if isinstance(arr, torch.Tensor):
                    arr = arr.detach().cpu().numpy()
                sv = arr[0]
        elif isinstance(shap_values, np.ndarray):
            sv = shap_values[0]
        elif isinstance(shap_values, torch.Tensor):
            sv = shap_values[0].detach().cpu().numpy()
        if sv is None:
            return None, {}, None
        # Normalize/aggregate
        if sv.ndim == 4:
            sv = np.squeeze(sv)
        if sv.ndim == 3 and sv.shape[-1] in (1, 3):
            sv = np.transpose(sv, (2, 0, 1))
        if sv.ndim == 2:
            sv = np.expand_dims(sv, 0)
        sv_abs = np.abs(sv).sum(axis=0).astype(np.float32)
        H, W = image_b.shape[-2], image_b.shape[-1]
        if sv_abs.shape != (H, W):
            sv_abs = cv2.resize(sv_abs, (W, H))
        if sv_abs.max() > 0:
            sv_abs = sv_abs / (sv_abs.max() + 1e-8)
        heatmap_uint8 = np.uint8(255 * sv_abs)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
        img = image_b[0].detach().cpu().numpy()
        img_vis = self._denormalize_for_display(img)
        if heatmap_color.shape[:2] != img_vis.shape[:2]:
            heatmap_color = cv2.resize(heatmap_color, (img_vis.shape[1], img_vis.shape[0]))
        # Render overlay with optional masked top-percent to avoid full wash
        mode = str(getattr(self, 'shap_overlay_mode', 'masked')).lower()
        if mode == 'masked':
            heat = sv_abs
            top = float(max(0.0, min(1.0, getattr(self, 'shap_top_percent', 0.15))))
            thr = float(np.quantile(heat, 1.0 - top)) if np.any(heat > 0) else 1.0
            mask = (heat >= thr).astype(np.float32)
            mask3 = np.repeat(mask[:, :, None], 3, axis=2)
            heat_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            a = float(max(0.0, min(1.0, getattr(self, 'shap_alpha', 0.6))))
            overlay_rgb = (img_vis * (1.0 - a * mask3) + heat_rgb * (a * mask3)).clip(0, 255).astype(np.uint8)
        else:
            overlay_bgr = cv2.addWeighted(cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        stats = {
            'mean_positive_contribution': float(np.mean(sv[sv > 0])) if np.any(sv > 0) else 0.0,
            'mean_negative_contribution': float(np.mean(sv[sv < 0])) if np.any(sv < 0) else 0.0,
            'mean_abs_contribution': float(np.mean(np.abs(sv))),
            'explained_class': int(explained_class),
        }
        return overlay_rgb, stats, heatmap_uint8
    
    def setup_preprocessing(self):
        """Set up image preprocessing transforms."""
        # Preprocessing for segmentation (same as training)
        self.segmentation_transforms = A.Compose([
            A.Resize(256, 256),  # Resize to segmentation model input size
            CLAHETransform(),
            ZScoreTransform(),
            ToTensorV2(transpose_mask=True)
        ])
        
        # Preprocessing for classification (ImageNet normalization)
        self.classification_transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def preprocess_image_for_segmentation(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for segmentation model.
        
        Args:
            image: Input RGB image [H, W, 3]
            
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Apply segmentation preprocessing
        transformed = self.segmentation_transforms(image=image)
        image_tensor = transformed['image']
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def run_segmentation(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run Stage 1: Segmentation to locate and roughly classify nuclei.
        
        Args:
            image: Input RGB image [H, W, 3]
            
        Returns:
            Tuple of (logits, prediction_mask)
        """
        print("[INFO] Stage 1: Running segmentation...")
        
        # Preprocess image
        image_tensor = self.preprocess_image_for_segmentation(image)
        
        # Run segmentation
        with torch.no_grad():
            logits = self.segmentation_model(image_tensor)
            
            # Get prediction mask
            prediction_mask = torch.argmax(logits, dim=1).squeeze(0)  # [H, W]
        
        print(f"[INFO] Segmentation complete. Detected classes: {torch.unique(prediction_mask).cpu().numpy()}")
        
        return logits.squeeze(0), prediction_mask
    
    def extract_nuclei_instances(
        self, 
        image: np.ndarray, 
        segmentation_logits: torch.Tensor
    ) -> List[Dict]:
        """
        Extract individual nucleus instances from segmentation results.
        
        Args:
            image: Original RGB image [H, W, 3]
            segmentation_logits: Segmentation model logits [6, H, W]
            
        Returns:
            List of extracted nucleus instances
        """
        print("[INFO] Extracting individual nucleus instances...")
        
        # Resize image to match segmentation output if needed
        seg_height, seg_width = segmentation_logits.shape[-2:]
        if image.shape[:2] != (seg_height, seg_width):
            image_resized = cv2.resize(image, (seg_width, seg_height), interpolation=cv2.INTER_LINEAR)
        else:
            image_resized = image
        
        # Convert image to tensor format for extraction function
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        
        # Extract nuclei instances
        nuclei_instances = extract_nuclei_from_prediction(
            image=image_tensor,
            prediction_logits=segmentation_logits,
            min_area=self.min_nucleus_area,
            max_area=self.max_nucleus_area,
            context_padding=self.context_padding,
            target_size=224
        )
        
        print(f"[INFO] Extracted {len(nuclei_instances)} nucleus instances")
        return nuclei_instances
    
    def classify_nuclei(self, nuclei_instances: List[Dict]) -> List[Dict]:
        """
        Run Stage 2: Fine-grained classification of individual nucleus patches.
        
        Args:
            nuclei_instances: List of extracted nucleus instances
            
        Returns:
            List of nucleus instances with refined classifications
        """
        if len(nuclei_instances) == 0:
            print("⚠️  No nuclei instances to classify")
            return nuclei_instances
        
        print(f"🧬 Stage 2: Running fine-grained classification on {len(nuclei_instances)} nuclei...")
        
        # Prepare batch of nucleus patches
        patches = []
        patch_tensors = []
        for nucleus in nuclei_instances:
            patch = nucleus['patch']  # Already 224x224x3
            
            # Apply classification preprocessing
            transformed = self.classification_transforms(image=patch)
            patch_tensor = transformed['image']
            patches.append(patch_tensor)
            patch_tensors.append(patch_tensor)
        
        # Create batch
        batch = torch.stack(patches).to(self.device)
        
        # Run classification
        with torch.no_grad():
            outputs = self.classifier_model(batch)
            logits = outputs['logits']  # Extract logits from dictionary
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(logits, dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]
        
        # Update nucleus instances with refined classifications
        # Prepare SHAP explainer (use first N patches as background)
        explainer = None
        if self.enable_shap:
            try:
                bg_count = min(max(1, self.shap_bg), len(patch_tensors))
                background = torch.stack(patch_tensors[:bg_count])
                # Create explainer using the class method we added to the instance
                explainer = self._create_shap_explainer(background)
                if explainer is None:
                    print("⚠️  SHAP explainer init failed: falling back to disabled SHAP.")
                    self.enable_shap = False
            except Exception as e:
                print(f"⚠️  SHAP explainer init failed: {e}")
                explainer = None
                self.enable_shap = False
        
        explanations_dir = None
        if self._current_output_dir is not None:
            explanations_dir = self._current_output_dir / 'explanations'
            explanations_dir.mkdir(exist_ok=True, parents=True)
        
        for i, nucleus in enumerate(nuclei_instances):
            # Map from classifier output (0-4) back to original class IDs (1-5)
            # Classifier outputs: 0=Neoplastic, 1=Inflammatory, 2=Connective, 3=Dead, 4=Epithelial
            # Class IDs: 1=Neoplastic, 2=Inflammatory, 3=Connective, 4=Dead, 5=Epithelial (Background=0 excluded)
            refined_class_id = predicted_classes[i].item() + 1
            confidence = confidence_scores[i].item()
            class_probs = probabilities[i].cpu().numpy()
            
            # Validate class mapping (should be 1-5, excluding background=0)
            max_class_id = len(self.classification_classes)
            assert 1 <= refined_class_id <= max_class_id, f"Invalid class ID {refined_class_id}, should be 1-{max_class_id} (excluding background)"
            
            # Store original segmentation prediction
            nucleus['segmentation_class_id'] = nucleus['class_id']
            nucleus['segmentation_class_name'] = self.class_names[nucleus['class_id']]
            
            # Update with refined classification
            nucleus['class_id'] = refined_class_id
            nucleus['class_name'] = self.class_names[refined_class_id]
            nucleus['confidence'] = confidence
            nucleus['class_probabilities'] = class_probs
            
            # Agreement between stages
            nucleus['stage_agreement'] = (nucleus['segmentation_class_id'] == refined_class_id)
            
            # Explanations per nucleus
            if explanations_dir is not None:
                # Grad-CAM
                if self.enable_gradcam:
                    try:
                        heatmap_uint8, overlay_rgb = self.generate_gradcam(patch_tensors[i].unsqueeze(0), predicted_classes[i].item())
                        cv2.imwrite(str(explanations_dir / f'gradcam_nucleus_{nucleus["instance_id"]}.png'), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
                        nucleus['gradcam_path'] = str(explanations_dir / f'gradcam_nucleus_{nucleus["instance_id"]}.png')
                    except Exception as e:
                        print(f"⚠️  Grad-CAM failed for nucleus {nucleus['instance_id']}: {e}")
                        heatmap_uint8 = None
                else:
                    heatmap_uint8 = None
                # SHAP
                if self.enable_shap and explainer is not None:
                    try:
                        ov_rgb, shap_stats, heat_uint8 = self._generate_shap_overlay(patch_tensors[i], int(predicted_classes[i].item()), explainer, nsamples=self.shap_nsamples)
                        if ov_rgb is not None:
                            cv2.imwrite(str(explanations_dir / f'shap_nucleus_{nucleus["instance_id"]}.png'), cv2.cvtColor(ov_rgb, cv2.COLOR_RGB2BGR))
                            nucleus['shap_path'] = str(explanations_dir / f'shap_nucleus_{nucleus["instance_id"]}.png')
                            self._shap_summaries.append({
                                'nucleus_id': int(nucleus['instance_id']),
                                'predicted_class': int(predicted_classes[i].item()),
                                'predicted_class_name': self.classification_classes[predicted_classes[i].item()],
                                'confidence': float(confidence_scores[i].item()),
                                **shap_stats,
                            })
                        else:
                            heat_uint8 = None
                    except Exception as e:
                        print(f"⚠️  SHAP failed for nucleus {nucleus['instance_id']}: {e}")
                        heat_uint8 = None
                else:
                    heat_uint8 = None
                
                # Quality metrics per nucleus
                try:
                    gcm = None
                    scm = None
                    corr = None
                    if isinstance(heatmap_uint8, np.ndarray):
                        hm = heatmap_uint8.astype(np.float32) / 255.0
                        Hc, Wc = hm.shape
                        cy, cx = Hc // 2, Wc // 2
                        hy, hx = int(Hc * 0.3), int(Wc * 0.3)
                        y1, y2 = max(0, cy - hy), min(Hc, cy + hy)
                        x1, x2 = max(0, cx - hx), min(Wc, cx + hx)
                        total = float(hm.sum()) + 1e-8
                        gcm = float(hm[y1:y2, x1:x2].sum() / total) if total > 0 else 0.0
                    if isinstance(heat_uint8, np.ndarray):
                        sm = heat_uint8.astype(np.float32) / 255.0
                        Hs, Ws = sm.shape
                        cy, cx = Hs // 2, Ws // 2
                        hy, hx = int(Hs * 0.3), int(Ws * 0.3)
                        y1, y2 = max(0, cy - hy), min(Hs, cy + hy)
                        x1, x2 = max(0, cx - hx), min(Ws, cx + hx)
                        total_s = float(sm.sum()) + 1e-8
                        scm = float(sm[y1:y2, x1:x2].sum() / total_s) if total_s > 0 else 0.0
                    if isinstance(heatmap_uint8, np.ndarray) and isinstance(heat_uint8, np.ndarray):
                        a = (heatmap_uint8.astype(np.float32) / 255.0).flatten()
                        b = (heat_uint8.astype(np.float32) / 255.0).flatten()
                        if a.std() > 1e-6 and b.std() > 1e-6:
                            corr = float(np.corrcoef(a, b)[0, 1])
                    self._explain_quality_records.append({
                        'nucleus_id': int(nucleus['instance_id']),
                        'predicted_class': int(predicted_classes[i].item()),
                        'predicted_class_name': self.classification_classes[predicted_classes[i].item()],
                        'confidence': float(confidence_scores[i].item()),
                        'gradcam_center_mass': gcm,
                        'shap_center_mass': scm,
                        'shap_gradcam_corr': corr
                    })
                except Exception:
                    pass
        
        print("✅ Classification complete!")
        return nuclei_instances
    
    def create_fused_prediction_mask(
        self,
        segmentation_mask: np.ndarray,
        nuclei_instances: List[Dict],
        confidence_threshold: float = 0.6
    ) -> np.ndarray:
        """
        Create final prediction mask using confidence-based fusion of segmentation and classification.
        
        This implements the same fusion strategy as the evaluation script for consistency.
        
        Args:
            segmentation_mask: Segmentation model prediction [H, W]
            nuclei_instances: List of classified nucleus instances
            confidence_threshold: Minimum confidence to override segmentation (default: 0.6)
            
        Returns:
            Fused prediction mask [H, W] with class IDs 0-5
        """
        from scipy import ndimage
        
        # Start with segmentation as base
        fused_mask = segmentation_mask.copy()
        H, W = segmentation_mask.shape
        num_classes = 6  # Background + 5 nucleus types
        
        # Precompute connected components per class
        cc_by_class = {}
        for cls in range(1, num_classes):
            cls_mask = (segmentation_mask == cls).astype(np.uint8)
            if cls_mask.any():
                num_cc, cc_labels = cv2.connectedComponents(cls_mask)
                if num_cc > 1:
                    dist_transform = ndimage.distance_transform_edt(cls_mask)
                else:
                    dist_transform = None
            else:
                num_cc, cc_labels, dist_transform = 0, None, None
            cc_by_class[cls] = (num_cc, cc_labels, dist_transform)
        
        # Apply classifications with confidence-based fusion
        for nucleus in nuclei_instances:
            pred_class_id = nucleus['class_id']  # 1-5
            confidence = nucleus.get('confidence', 0.0)
            seg_class_id = nucleus.get('segmentation_class_id', 0)
            
            centroid = nucleus.get('centroid', (H // 2, W // 2))
            cy, cx = int(centroid[0]), int(centroid[1])
            cy = max(0, min(H - 1, cy))
            cx = max(0, min(W - 1, cx))
            
            # Get segmentation prediction at centroid
            seg_at_centroid = int(segmentation_mask[cy, cx])
            
            # CONFIDENCE-BASED DECISION:
            # 1. If agreement, keep as is (no change needed)
            # 2. If segmentation is background but we detected nucleus, override
            # 3. If disagreement and high confidence, override
            # 4. If disagreement and low confidence, keep segmentation
            
            should_override = False
            if seg_at_centroid == pred_class_id:
                # Already agrees, no change needed
                should_override = False
            elif seg_at_centroid == 0:
                # Segmentation missed this nucleus, add it
                should_override = True
            elif confidence >= confidence_threshold:
                # Confident disagreement, trust classifier
                should_override = True
            else:
                # Low confidence disagreement, keep segmentation
                should_override = False
            
            if not should_override:
                continue
            
            # Find region to override
            region_mask = None
            
            # 1) Try connected component at centroid
            if seg_at_centroid > 0 and cc_by_class[seg_at_centroid][1] is not None:
                num_cc, cc_labels, dist_transform = cc_by_class[seg_at_centroid]
                cc_id = int(cc_labels[cy, cx])
                if cc_id > 0:
                    region_mask = (cc_labels == cc_id)
            
            # 2) Fallback to explicit mask if available
            if region_mask is None or not region_mask.any():
                m = nucleus.get('mask', None)
                if isinstance(m, np.ndarray) and m.dtype == bool:
                    if m.shape == (H, W):
                        region_mask = m
                    else:
                        # Try to resize/place mask
                        bbox = nucleus.get('bbox', None)
                        if bbox is not None:
                            y1, x1, y2, x2 = bbox
                            region_mask = np.zeros((H, W), dtype=bool)
                            bbox_h, bbox_w = y2 - y1, x2 - x1
                            if bbox_h > 0 and bbox_w > 0:
                                try:
                                    mask_resized = cv2.resize(m.astype(np.uint8), (bbox_w, bbox_h), 
                                                             interpolation=cv2.INTER_NEAREST)
                                    region_mask[y1:y2, x1:x2] = mask_resized.astype(bool)
                                except:
                                    region_mask = None
            
            # 3) Final fallback: conservative circle for high confidence only
            if (region_mask is None or not region_mask.any()) and confidence >= 0.8:
                area = int(nucleus.get('area', 100))
                rad = max(2, int(np.sqrt(max(1, area) / np.pi) * 0.9))
                y, x = np.ogrid[:H, :W]
                circle = (x - cx) ** 2 + (y - cy) ** 2 <= rad ** 2
                region_mask = np.logical_and(circle, segmentation_mask > 0)
                if not region_mask.any():
                    region_mask = circle
            
            if region_mask is not None and region_mask.any():
                fused_mask[region_mask] = pred_class_id
        
        return fused_mask

    # ---------------------- Explainability: Grad-CAM ----------------------
    def _register_gradcam_hooks(self):
        if not hasattr(self, 'classifier_model'):
            return
        backbone = getattr(self.classifier_model, 'backbone', None)
        if backbone is None:
            return
        # Resolve module by feature_info index
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
        target_module = None
        try:
            fi = getattr(backbone, 'feature_info', None)
            if fi is not None and len(fi) > 0:
                idx = self.gradcam_layer_idx
                nfi = len(fi)
                if idx < 0:
                    idx = nfi + idx
                idx = max(0, min(idx, nfi - 1))
                info = fi[idx]
                module_path = info.get('module', None) or info.get('name', None)
                if isinstance(module_path, str):
                    if hasattr(backbone, 'body'):
                        try:
                            target_module = _get_submodule(backbone.body, module_path)
                        except Exception:
                            target_module = None
                    if target_module is None:
                        target_module = _get_submodule(backbone, module_path)
        except Exception:
            target_module = None
        if target_module is None:
            conv_layers = [m for m in backbone.modules() if isinstance(m, torch.nn.Conv2d)]
            if conv_layers:
                target_module = conv_layers[-1]
        if target_module is None:
            return
        if self._gc_forward_handle is not None:
            try:
                self._gc_forward_handle.remove()
            except Exception:
                pass
        def fwd_hook(module, inp, out):
            self._gc_activations = out
        self._gc_forward_handle = target_module.register_forward_hook(fwd_hook)

    def generate_gradcam(self, image: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        was_training = self.classifier_model.training
        self.classifier_model.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        for p in self.classifier_model.parameters():
            p.requires_grad_(True)
        self.classifier_model.zero_grad(set_to_none=True)
        self._gc_activations = None
        logits = self.classifier_model(image)['logits']
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())
        score = logits[:, target_class].sum()
        gradients = None
        activations = self._gc_activations
        if activations is not None and isinstance(activations, torch.Tensor):
            try:
                gradients = torch.autograd.grad(score, activations, retain_graph=True, allow_unused=True)[0]
            except Exception:
                gradients = None
        if gradients is None or activations is None:
            logits = self.classifier_model(image)['logits']
            score = logits[:, target_class].sum()
            if self._gc_activations is not None and isinstance(self._gc_activations, torch.Tensor):
                self._gc_activations.retain_grad()
                score.backward(retain_graph=True)
                gradients = self._gc_activations.grad
                activations = self._gc_activations
        if activations is None or gradients is None:
            _, _, H, W = image.shape
            heatmap_uint8 = np.zeros((W, H), dtype=np.uint8)
            img = image.squeeze(0).detach().cpu().numpy()
            if img.shape[0] == 3:
                img = np.transpose(img, (1,2,0))
            img_vis = (img * 255.0).clip(0,255).astype(np.uint8) if img.max() <= 1.0 + 1e-3 else img.clip(0,255).astype(np.uint8)
            overlay_bgr = cv2.addWeighted(cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR), 0.5, cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET), 0.5, 0)
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            if was_training:
                self.classifier_model.train()
            return heatmap_uint8, overlay_rgb
        activations = activations.detach()
        gradients = gradients.detach()
        weights = gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam_min = cam.amin(dim=(2,3), keepdim=True)
        cam_max = cam.amax(dim=(2,3), keepdim=True)
        if torch.all((cam_max - cam_min) < 1e-6):
            cam = torch.relu((gradients * activations).sum(dim=1, keepdim=True))
            cam_min = cam.amin(dim=(2,3), keepdim=True)
            cam_max = cam.amax(dim=(2,3), keepdim=True)
        cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam_np = cam_norm.squeeze().cpu().numpy()
        _, _, H, W = image.shape
        heatmap = cv2.resize(cam_np, (W, H))
        ks = int(self.gradcam_blur_ksize) if self.gradcam_blur_ksize else 0
        if ks > 0:
            if ks % 2 == 0:
                ks += 1
            heatmap = cv2.GaussianBlur(heatmap, (ks, ks), self.gradcam_blur_sigma if self.gradcam_blur_sigma > 0 else 0)
            minv, maxv = float(heatmap.min()), float(heatmap.max())
            if maxv - minv > 1e-8:
                heatmap = (heatmap - minv) / (maxv - minv)
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        img = image.squeeze(0).detach().cpu().numpy()
        img_vis = self._denormalize_for_display(img)
        overlay_bgr = cv2.addWeighted(cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR), 0.5, heatmap_color, 0.5, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        if was_training:
            self.classifier_model.train()
        return heatmap_uint8, overlay_rgb
    
    def analyze_image(
        self, 
        image_path: str,
        save_results: bool = True,
        output_dir: str = None,
        visualize: bool = True
    ) -> Dict:
        """
        Run complete two-stage analysis on an input image.
        
        Args:
            image_path: Path to input image
            save_results: Whether to save results to disk
            output_dir: Directory to save results
            visualize: Whether to show visualizations
            
        Returns:
            Complete analysis results
        """
        print(f"\n🚀 Starting Two-Stage Nuclei Analysis")
        print(f"📁 Input image: {image_path}")
        print("="*60)
        
        # Reset explainability accumulators
        self._shap_summaries = []
        self._explain_quality_records = []
        
        # Prepare output dir for explanations if saving
        if save_results:
            if output_dir is None:
                output_dir = f"pipeline_results_{Path(image_path).stem}"
            self._current_output_dir = Path(output_dir)
        else:
            self._current_output_dir = None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        print(f"📸 Image loaded: {image.shape}")
        
        # Stage 1: Segmentation
        segmentation_logits, segmentation_mask = self.run_segmentation(image)
        
        # Extract nucleus instances
        nuclei_instances = self.extract_nuclei_instances(image, segmentation_logits)
        
        if len(nuclei_instances) == 0:
            print("⚠️  No nuclei detected in the image")
            return {
                'image_path': image_path,
                'original_size': original_size,
                'nuclei_count': 0,
                'nuclei_instances': [],
                'statistics': {'total_count': 0}
            }
        
        # Stage 2: Fine-grained classification
        classified_nuclei = self.classify_nuclei(nuclei_instances)
        
        # Generate statistics
        stats = self.generate_statistics(classified_nuclei)
        
        # Create results dictionary
        results = {
            'image_path': image_path,
            'original_size': original_size,
            'nuclei_count': len(classified_nuclei),
            'nuclei_instances': classified_nuclei,
            'statistics': stats,
            'segmentation_mask': segmentation_mask.cpu().numpy(),
            'processing_info': {
                'min_nucleus_area': self.min_nucleus_area,
                'max_nucleus_area': self.max_nucleus_area,
                'context_padding': self.context_padding,
                'device': self.device
            }
        }
        
        # Save results if requested
        if save_results:
            self.save_results(results, output_dir, image)
            # Save SHAP text summary if any
            if self.enable_shap and len(self._shap_summaries) > 0:
                shap_path = Path(output_dir) / 'shap_text_summary.json'
                with open(shap_path, 'w') as f:
                    json.dump(self._shap_summaries, f, indent=2)
            # Save explanation quality metrics
            if len(self._explain_quality_records) > 0:
                quality_path = Path(output_dir) / 'explanations_quality.json'
                with open(quality_path, 'w') as f:
                    json.dump(self._explain_quality_records, f, indent=2)
            # Print brief summary
            try:
                gcm_vals = [q.get('gradcam_center_mass') for q in self._explain_quality_records if isinstance(q.get('gradcam_center_mass'), (float, int))]
                scm_vals = [q.get('shap_center_mass') for q in self._explain_quality_records if isinstance(q.get('shap_center_mass'), (float, int))]
                corr_vals = [q.get('shap_gradcam_corr') for q in self._explain_quality_records if isinstance(q.get('shap_gradcam_corr'), (float, int))]
                explanations_dir = Path(output_dir) / 'explanations'
                print(f"🖼️  Saved per-nucleus Grad-CAM/SHAP overlays → {explanations_dir}")
                if gcm_vals:
                    print(f"📊 Grad-CAM center-mass avg: {float(np.mean(gcm_vals)):.3f}")
                if scm_vals:
                    print(f"📊 SHAP center-mass avg: {float(np.mean(scm_vals)):.3f}")
                if corr_vals:
                    print(f"📊 SHAP–GradCAM correlation avg: {float(np.mean(corr_vals)):.3f}")
            except Exception:
                pass
        
        # Visualize results if requested
        if visualize:
            self.visualize_results(results, image)
        
        print("\n✅ Two-Stage Analysis Complete!")
        print(f"📊 Found {len(classified_nuclei)} nuclei instances")
        print(f"📈 Stage agreement: {stats['stage_agreement']:.1%}")
        
        return results
    
    def analyze_image_array(
        self, 
        image: np.ndarray,
        save_results: bool = True,
        output_dir: str = None,
        visualize: bool = True
    ) -> Dict:
        """
        Run complete two-stage analysis on an input image array.
        
        Args:
            image: Input image as numpy array [H, W, 3]
            save_results: Whether to save results to disk
            output_dir: Directory to save results
            visualize: Whether to show visualizations
            
        Returns:
            Complete analysis results
        """
        print(f"\n🚀 Starting Two-Stage Nuclei Analysis on Image Array")
        print(f"📸 Image shape: {image.shape}")
        print("="*60)
        
        # Reset explainability accumulators
        self._shap_summaries = []
        self._explain_quality_records = []
        
        # Prepare output dir for explanations if saving
        if save_results:
            if output_dir is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = f"pipeline_results_{timestamp}"
            self._current_output_dir = Path(output_dir)
        else:
            self._current_output_dir = None
        
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        original_size = image.shape[:2]
        
        # Stage 1: Segmentation
        segmentation_logits, segmentation_mask = self.run_segmentation(image)
        
        # Extract nucleus instances
        nuclei_instances = self.extract_nuclei_instances(image, segmentation_logits)
        
        if len(nuclei_instances) == 0:
            print("⚠️  No nuclei detected in the image")
            return {
                'image_path': 'array_input',
                'original_size': original_size,
                'nuclei_count': 0,
                'nuclei_instances': [],
                'statistics': {'total_count': 0},
                'segmentation': {
                    'logits': segmentation_logits,
                    'prediction_mask': segmentation_mask,
                },
                'extracted_nuclei': [],
                'classifications': {}
            }
        
        # Stage 2: Fine-grained classification
        classified_nuclei = self.classify_nuclei(nuclei_instances)
        
        # Generate statistics
        stats = self.generate_statistics(classified_nuclei)
        
        # Create fused prediction mask using confidence-based fusion
        seg_mask_np = segmentation_mask.cpu().numpy()
        fused_mask = self.create_fused_prediction_mask(
            segmentation_mask=seg_mask_np,
            nuclei_instances=classified_nuclei,
            confidence_threshold=0.6
        )
        
        # Convert to format expected by demo script
        extracted_nuclei = []
        classifications = {}
        
        for i, nucleus in enumerate(classified_nuclei):
            nucleus_id = nucleus['instance_id']
            
            # Create extracted nuclei info with better mask handling
            nucleus_mask = nucleus.get('mask', np.zeros(original_size, dtype=bool))
            # Ensure mask is properly sized
            if isinstance(nucleus_mask, np.ndarray) and nucleus_mask.shape != original_size:
                nucleus_mask = np.zeros(original_size, dtype=bool)
            
            extracted_nuclei.append({
                'nucleus_id': nucleus_id,
                'mask': nucleus_mask,
                'mask_patch': nucleus.get('mask_patch', None),  # Keep the patch-level mask too
                'contour': nucleus.get('contour', []),
                'area': nucleus['area'],
                'centroid': nucleus['centroid'],
                'bbox': nucleus.get('bbox', (0, 0, original_size[0], original_size[1])),
                'patch': nucleus.get('patch', np.zeros((224, 224, 3), dtype=np.uint8))
            })
            
            # Create classification info
            classifications[nucleus_id] = {
                'predicted_class': nucleus['class_id'] - 1,  # Convert 1-5 to 0-4
                'predicted_class_name': nucleus['class_name'],
                'confidence': nucleus['confidence'],
                'class_probabilities': nucleus['class_probabilities']
            }
        
        # Create results dictionary
        results = {
            'image_path': 'array_input',
            'original_size': original_size,
            'nuclei_count': len(classified_nuclei),
            'nuclei_instances': classified_nuclei,
            'statistics': stats,
            'segmentation_mask': seg_mask_np,
            'fused_prediction_mask': fused_mask,  # Add the confidence-based fused mask
            'segmentation': {
                'logits': segmentation_logits,
                'prediction_mask': segmentation_mask,
            },
            'extracted_nuclei': extracted_nuclei,
            'classifications': classifications,
            'processing_info': {
                'min_nucleus_area': self.min_nucleus_area,
                'max_nucleus_area': self.max_nucleus_area,
                'context_padding': self.context_padding,
                'device': self.device,
                'confidence_threshold': 0.6
            }
        }
        
        # Save results if requested
        if save_results:
            self.save_results(results, output_dir, image)
            # Save SHAP text summary if any
            if self.enable_shap and len(self._shap_summaries) > 0:
                shap_path = Path(output_dir) / 'shap_text_summary.json'
                with open(shap_path, 'w') as f:
                    json.dump(self._shap_summaries, f, indent=2)
            # Save explanation quality metrics
            if len(self._explain_quality_records) > 0:
                quality_path = Path(output_dir) / 'explanations_quality.json'
                with open(quality_path, 'w') as f:
                    json.dump(self._explain_quality_records, f, indent=2)
            # Print brief summary
            try:
                gcm_vals = [q.get('gradcam_center_mass') for q in self._explain_quality_records if isinstance(q.get('gradcam_center_mass'), (float, int))]
                scm_vals = [q.get('shap_center_mass') for q in self._explain_quality_records if isinstance(q.get('shap_center_mass'), (float, int))]
                corr_vals = [q.get('shap_gradcam_corr') for q in self._explain_quality_records if isinstance(q.get('shap_gradcam_corr'), (float, int))]
                explanations_dir = Path(output_dir) / 'explanations'
                print(f"🖼️  Saved per-nucleus Grad-CAM/SHAP overlays → {explanations_dir}")
                if gcm_vals:
                    print(f"📊 Grad-CAM center-mass avg: {float(np.mean(gcm_vals)):.3f}")
                if scm_vals:
                    print(f"📊 SHAP center-mass avg: {float(np.mean(scm_vals)):.3f}")
                if corr_vals:
                    print(f"📊 SHAP–GradCAM correlation avg: {float(np.mean(corr_vals)):.3f}")
            except Exception:
                pass
        
        # Visualize results if requested
        if visualize:
            self.visualize_results(results, image)
        
        print("\n✅ Two-Stage Analysis Complete!")
        print(f"📊 Found {len(classified_nuclei)} nuclei instances")
        print(f"📈 Stage agreement: {stats['stage_agreement']:.1%}")
        
        return results
    
    def generate_statistics(self, nuclei_instances: List[Dict]) -> Dict:
        """Generate comprehensive statistics about the analysis results."""
        if len(nuclei_instances) == 0:
            return {'total_count': 0}
        
        # Basic statistics
        stats = nucleus_statistics(nuclei_instances)
        
        # Two-stage specific statistics
        stage_agreements = [nucleus['stage_agreement'] for nucleus in nuclei_instances]
        stats['stage_agreement'] = np.mean(stage_agreements)
        
        # Confidence statistics
        confidences = [nucleus['confidence'] for nucleus in nuclei_instances]
        stats['confidence_stats'] = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'median': np.median(confidences)
        }
        
        # Class-wise confidence
        class_confidences = {}
        for nucleus in nuclei_instances:
            class_name = nucleus['class_name']
            if class_name not in class_confidences:
                class_confidences[class_name] = []
            class_confidences[class_name].append(nucleus['confidence'])
        
        stats['class_confidence'] = {
            class_name: {
                'mean': np.mean(confs),
                'count': len(confs)
            }
            for class_name, confs in class_confidences.items()
        }
        
        return stats
    
    def save_results(self, results: Dict, output_dir: str, original_image: np.ndarray):
        """Save analysis results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"💾 Saving results to: {output_path}")
        
        # Save JSON results (without numpy arrays)
        json_results = {
            'image_path': results['image_path'],
            'original_size': results['original_size'],
            'nuclei_count': results['nuclei_count'],
            'statistics': results['statistics'],
            'processing_info': results['processing_info']
        }
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(output_path / 'analysis_results.json', 'w') as f:
            json.dump(json_results, f, default=convert_numpy, indent=2)
        
        # Save detailed nuclei information
        nuclei_data = []
        for nucleus in results['nuclei_instances']:
            nucleus_info = {
                'instance_id': nucleus['instance_id'],
                'class_id': nucleus['class_id'],
                'class_name': nucleus['class_name'],
                'confidence': nucleus['confidence'],
                'area': nucleus['area'],
                'bbox': nucleus['bbox'],
                'centroid': nucleus['centroid'],
                'segmentation_class_id': nucleus['segmentation_class_id'],
                'segmentation_class_name': nucleus['segmentation_class_name'],
                'stage_agreement': nucleus['stage_agreement'],
                'class_probabilities': nucleus['class_probabilities'].tolist()
            }
            nuclei_data.append(nucleus_info)
        
        with open(output_path / 'nuclei_details.json', 'w') as f:
            json.dump(nuclei_data, f, default=convert_numpy, indent=2)
        
        # Save segmentation mask
        seg_mask_colored = self.class_colors[results['segmentation_mask']]
        cv2.imwrite(str(output_path / 'segmentation_mask.png'), cv2.cvtColor(seg_mask_colored, cv2.COLOR_RGB2BGR))
        
        # Save fused prediction mask if available
        if 'fused_prediction_mask' in results:
            fused_mask_colored = self.class_colors[results['fused_prediction_mask']]
            cv2.imwrite(str(output_path / 'fused_prediction_mask.png'), cv2.cvtColor(fused_mask_colored, cv2.COLOR_RGB2BGR))
        
        # Save visualization
        self.create_summary_visualization(results, original_image, output_path / 'analysis_summary.png')
        
        print(f"✅ Results saved to: {output_path}")
    
    def visualize_results(self, results: Dict, original_image: np.ndarray):
        """Display comprehensive visualization of results."""
        self.create_summary_visualization(results, original_image, save_path=None, show=True)
    
    def create_summary_visualization(
        self, 
        results: Dict, 
        original_image: np.ndarray, 
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """Create comprehensive summary visualization."""
        nuclei_instances = results['nuclei_instances']
        stats = results['statistics']
        seg_mask = results['segmentation_mask']
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Two-Stage Nuclei Analysis - {Path(results["image_path"]).name}', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Segmentation result
        seg_colored = self.class_colors[seg_mask]
        axes[0, 1].imshow(seg_colored)
        axes[0, 1].set_title('Stage 1: Segmentation', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Fused prediction
        if 'fused_prediction_mask' in results:
            fused_colored = self.class_colors[results['fused_prediction_mask']]
            axes[0, 2].imshow(fused_colored)
            axes[0, 2].set_title('Final: Confidence-Based Fusion', fontweight='bold')
        else:
            # Fallback: Show classifications on original image
            axes[0, 2].imshow(original_image)
            for nucleus in nuclei_instances:
                y1, x1, y2, x2 = nucleus['bbox']
                color = self.class_colors[nucleus['class_id']] / 255.0
                
                # Draw bounding box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, edgecolor=color, linewidth=2)
                axes[0, 2].add_patch(rect)
                
                # Add label
                axes[0, 2].text(x1, y1-5, f"{nucleus['class_name'][:4]}\n{nucleus['confidence']:.2f}",
                              fontsize=8, color=color, fontweight='bold',
                              bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
            axes[0, 2].set_title('Stage 2: Classification Results', fontweight='bold')
        
        axes[0, 2].axis('off')
        
        # Class distribution
        if 'class_distribution' in stats:
            class_names = list(stats['class_distribution'].keys())
            class_counts = list(stats['class_distribution'].values())
            
            bars = axes[1, 0].bar(range(len(class_names)), class_counts, 
                                color=[self.class_colors[self.class_names.index(name)]/255.0 for name in class_names])
            axes[1, 0].set_title('Class Distribution', fontweight='bold')
            axes[1, 0].set_xlabel('Nucleus Type')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_xticks(range(len(class_names)))
            axes[1, 0].set_xticklabels([name[:4] for name in class_names], rotation=45)
            
            # Add count labels
            for bar, count in zip(bars, class_counts):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                              f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Confidence distribution
        if len(nuclei_instances) > 0:
            confidences = [nucleus['confidence'] for nucleus in nuclei_instances]
            axes[1, 1].hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 1].axvline(np.mean(confidences), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(confidences):.3f}')
            axes[1, 1].set_title('Classification Confidence', fontweight='bold')
            axes[1, 1].set_xlabel('Confidence Score')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Summary statistics
        axes[1, 2].axis('off')
        summary_text = f"""Summary Statistics:
        
Total Nuclei: {results['nuclei_count']}
Stage Agreement: {stats.get('stage_agreement', 0):.1%}

Mean Confidence: {stats.get('confidence_stats', {}).get('mean', 0):.3f}
Std Confidence: {stats.get('confidence_stats', {}).get('std', 0):.3f}

Area Stats:
  Mean: {stats.get('area_stats', {}).get('mean', 0):.1f} px
  Median: {stats.get('area_stats', {}).get('median', 0):.1f} px

Processing:
  Min area: {results['processing_info']['min_nucleus_area']} px
  Max area: {results['processing_info']['max_nucleus_area']} px
  Device: {results['processing_info']['device']}
        """
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=12, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        axes[1, 2].set_title('Analysis Summary', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Summary visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def find_best_models():
    """Auto-detect the best available models if defaults don't exist."""
    best_models = {}
    
    # Find best segmentation model
    seg_paths = [
        'lightning_logs/segmentation/version_1/checkpoints/advanced-epoch=112-val_dice=0.656.ckpt',
    ]
    
    # Search for any segmentation checkpoints
    lightning_logs = Path('lightning_logs')
    if lightning_logs.exists():
        for pattern in ['**/advanced-*val_dice*.ckpt', '**/segmentation*/**/*.ckpt']:
            seg_candidates = list(lightning_logs.glob(pattern))
            seg_paths.extend([str(p) for p in seg_candidates])
    
    for path in seg_paths:
        if Path(path).exists():
            best_models['segmentation'] = path
            break
    
    # Find best classifier model  
    clf_paths = [
        'lightning_logs/classifier/classifier_efficientnet_b3_20250727_002713/version_0/checkpoints/classifier-epoch=19-val_f1=0.806.ckpt',
    ]
    
    # Search for any classifier checkpoints
    if lightning_logs.exists():
        for pattern in ['**/classifier-*val_f1*.ckpt', '**/classifier*/**/*.ckpt']:
            clf_candidates = list(lightning_logs.glob(pattern))
            clf_paths.extend([str(p) for p in clf_candidates])
    
    for path in clf_paths:
        if Path(path).exists():
            best_models['classifier'] = path
            break
    
    return best_models


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Two-Stage Nuclei Analysis Pipeline')
    
    # Model paths (with auto-detection of best models)
    parser.add_argument('--segmentation_model', type=str, 
                       default='lightning_logs/segmentation/version_1/checkpoints/advanced-epoch=112-val_dice=0.656.ckpt',
                       help='Path to trained segmentation model checkpoint')
    parser.add_argument('--classifier_model', type=str,
                       default='lightning_logs/classifier/classifier_efficientnet_b3_20250727_002713/version_0/checkpoints/classifier-epoch=19-val_f1=0.806.ckpt',
                       help='Path to trained state-of-the-art classifier checkpoint (.ckpt file from Lightning)')
    
    # Input/output
    parser.add_argument('--input_image', type=str, required=True,
                       help='Path to input image for analysis')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save results (default: auto-generated)')
    
    # Processing parameters
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--min_nucleus_area', type=int, default=50,
                       help='Minimum nucleus area in pixels')
    parser.add_argument('--max_nucleus_area', type=int, default=5000,
                       help='Maximum nucleus area in pixels')
    parser.add_argument('--context_padding', type=int, default=32,
                       help='Context padding around nucleus for classification')
    
    # Output options
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to disk')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Do not show visualizations')
    
    args = parser.parse_args()
    
    # Auto-detect best models if defaults don't exist
    if not Path(args.segmentation_model).exists() or not Path(args.classifier_model).exists():
        print("🔍 Auto-detecting best available models...")
        best_models = find_best_models()
        
        if not Path(args.segmentation_model).exists() and 'segmentation' in best_models:
            args.segmentation_model = best_models['segmentation']
            print(f"   ✅ Found segmentation model: {args.segmentation_model}")
        
        if not Path(args.classifier_model).exists() and 'classifier' in best_models:
            args.classifier_model = best_models['classifier'] 
            print(f"   ✅ Found classifier model: {args.classifier_model}")
    
    print("🧬 TWO-STAGE NUCLEI ANALYSIS PIPELINE (STATE-OF-THE-ART)")
    print("="*60)
    print(f"🔍 Segmentation model: {args.segmentation_model}")
    print(f"🧠 Classifier model: {args.classifier_model}")
    print(f"📸 Input image: {args.input_image}")
    print(f"⚡ Device: {args.device}")
    print("="*60)
    
    # Check if files exist
    if not Path(args.segmentation_model).exists():
        raise FileNotFoundError(f"Segmentation model not found: {args.segmentation_model}")
    
    if not Path(args.classifier_model).exists():
        raise FileNotFoundError(f"Classifier model not found: {args.classifier_model}")
    
    if not Path(args.input_image).exists():
        raise FileNotFoundError(f"Input image not found: {args.input_image}")
    
    # Initialize pipeline
    pipeline = TwoStageNucleiPipeline(
        segmentation_model_path=args.segmentation_model,
        classifier_model_path=args.classifier_model,
        device=args.device,
        min_nucleus_area=args.min_nucleus_area,
        max_nucleus_area=args.max_nucleus_area,
        context_padding=args.context_padding
    )
    
    # Run analysis
    results = pipeline.analyze_image(
        image_path=args.input_image,
        save_results=not args.no_save,
        output_dir=args.output_dir,
        visualize=not args.no_visualize
    )
    
    print(f"\n🎉 Analysis complete! Processed {results['nuclei_count']} nuclei.")


if __name__ == '__main__':
    main() 