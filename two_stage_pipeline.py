"""
Two-Stage Nuclei Analysis Pipeline

This script implements the complete two-stage pipeline for nuclei analysis:
1. Stage 1: Segmentation - Attention U-Net locates nuclei and gives rough type prediction
2. Stage 2: Classification - State-of-the-art EfficientNet classifier provides fine-grained nucleus classification

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
        context_padding: int = 32
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
        
        print(f"🎯 Initializing Two-Stage Nuclei Pipeline on device: {self.device}")
        print(f"🧬 Classification: {len(self.classification_classes)} nucleus types (Background excluded)")
        print(f"   Classes: {', '.join(self.classification_classes)}")
        
        # Load models
        self.load_segmentation_model(segmentation_model_path)
        self.load_classifier_model(classifier_model_path)
        
        # Set up preprocessing
        self.setup_preprocessing()
        
        print("✅ Two-Stage Pipeline initialized successfully!")
    
    def load_segmentation_model(self, model_path: str):
        """Load the segmentation model (Stage 1)."""
        print(f"📁 Loading segmentation model from: {model_path}")
        
        # Load the trained Attention U-Net model
        self.segmentation_model = AdvancedAttentionModel.load_from_checkpoint(
            model_path,
            map_location=self.device
        )
        self.segmentation_model = self.segmentation_model.to(self.device)
        self.segmentation_model.eval()
        
        print("✅ Segmentation model loaded successfully!")
    
    def load_classifier_model(self, model_path: str):
        """Load the state-of-the-art nucleus classifier model (Stage 2)."""
        print(f"📁 Loading SOTA classifier model from: {model_path}")
        
        # Load the trained state-of-the-art nucleus classifier
        # This loads the Lightning checkpoint and extracts the actual model
        lightning_model = StateOfTheArtClassifierLightning.load_from_checkpoint(
            model_path,
            map_location=self.device
        )
        
        # Extract the actual classifier model from the Lightning wrapper
        self.classifier_model = lightning_model.model
        self.classifier_model.eval()
        
        # Verify model is configured for 5 classes (excluding background)
        expected_classes = len(self.classification_classes)
        if hasattr(lightning_model, 'num_classes'):
            assert lightning_model.num_classes == expected_classes, f"Classifier should have {expected_classes} classes, got {lightning_model.num_classes}"
            print(f"✅ Verified: Classifier configured for {lightning_model.num_classes} classes (excluding background)")
            print(f"   Classification classes: {', '.join(self.classification_classes)}")
        
        print("✅ State-of-the-art classifier model loaded successfully!")
    
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
        print("🔍 Stage 1: Running segmentation...")
        
        # Preprocess image
        image_tensor = self.preprocess_image_for_segmentation(image)
        
        # Run segmentation
        with torch.no_grad():
            logits = self.segmentation_model(image_tensor)
            
            # Get prediction mask
            prediction_mask = torch.argmax(logits, dim=1).squeeze(0)  # [H, W]
        
        print(f"✅ Segmentation complete. Detected classes: {torch.unique(prediction_mask).cpu().numpy()}")
        
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
        print("✂️  Extracting individual nucleus instances...")
        
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
        
        print(f"✅ Extracted {len(nuclei_instances)} nucleus instances")
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
        for nucleus in nuclei_instances:
            patch = nucleus['patch']  # Already 224x224x3
            
            # Apply classification preprocessing
            transformed = self.classification_transforms(image=patch)
            patch_tensor = transformed['image']
            patches.append(patch_tensor)
        
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
        
        print("✅ Classification complete!")
        return nuclei_instances
    
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
            if output_dir is None:
                output_dir = f"pipeline_results_{Path(image_path).stem}"
            
            self.save_results(results, output_dir, image)
        
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
        
        # Convert to format expected by demo script
        extracted_nuclei = []
        classifications = {}
        
        for i, nucleus in enumerate(classified_nuclei):
            nucleus_id = nucleus['instance_id']
            
            # Create extracted nuclei info
            extracted_nuclei.append({
                'nucleus_id': nucleus_id,
                'mask': nucleus.get('mask', np.zeros(original_size, dtype=bool)),
                'contour': nucleus.get('contour', []),
                'area': nucleus['area'],
                'centroid': nucleus['centroid'],
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
            'segmentation_mask': segmentation_mask.cpu().numpy(),
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
                'device': self.device
            }
        }
        
        # Save results if requested
        if save_results:
            if output_dir is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = f"pipeline_results_{timestamp}"
            
            self.save_results(results, output_dir, image)
        
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
        
        # Nuclei instances with classifications
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
        'lightning_logs/segmentation/version_1/checkpoints/advanced-epoch=12-val_dice=0.656.ckpt',
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
        'lightning_logs/classifier/classifier_efficientnet_b3_20250724_002315/version_0/checkpoints/classifier-epoch=04-val_f1=0.789.ckpt',
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
                       default='lightning_logs/segmentation/version_1/checkpoints/advanced-epoch=12-val_dice=0.656.ckpt',
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
    print(f"🧠 SOTA Classifier model: {args.classifier_model}")
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