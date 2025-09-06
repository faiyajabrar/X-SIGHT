## X-SIGHT: Two-Stage Nuclei Analysis for Histopathology

X-SIGHT is a two-stage, explainable pipeline for nuclei analysis in histopathology images. It performs:

- Segmentation: Locates nuclei and produces a coarse cell-type map using an Attention U-Net–style model.
- Classification: Refines each detected nucleus into a fine-grained class using a state-of-the-art EfficientNet classifier.
- Explainability: Generates per-nucleus Grad-CAM and SHAP visualizations to highlight the image regions that influenced the prediction.


### Why this matters (medical context)
Histopathology is the gold standard for cancer diagnosis and grading. Nuclei morphology, distribution, and types carry essential diagnostic information. Automated nuclei analysis can help pathologists by:
- Quantifying nuclei counts by type (e.g., neoplastic vs inflammatory)
- Revealing spatial patterns
- Providing model explanations to assess trust and failure modes

The five nucleus classes used here follow the PanNuke dataset convention:
- Neoplastic: Tumor cells (neoplastic epithelial). Abnormal morphology and proliferation.
- Inflammatory: Immune cells (e.g., lymphocytes). Indicative of inflammation/immune response.
- Connective: Stromal/connective tissue cells (e.g., fibroblasts). Provide structure/support.
- Dead: Necrotic/pyknotic cells. Indicate tissue damage, hypoxia, or treatment effect.
- Epithelial: Benign epithelial cells. Normal or non-neoplastic epithelial structures.


### Project structure
- `two_stage_pipeline.py`: Core pipeline class `TwoStageNucleiPipeline` for segmentation → extraction → classification → explanations, with CLI.
- `training/`: Training scripts for segmentation (`train_segmentation.py`) and classification (`train_classifier.py`).
- `utils/`: Dataset loader (`pannuke_dataset.py`), extraction utilities, and evaluation helpers.
- `streamlit_app.py`: Interactive app to upload an image and view segmentation, labels, Grad-CAM, and SHAP.
- `evaluation_results/`: Example outputs and metrics.
- `streamlit_outputs/`: Outputs produced by the Streamlit app.


### Pipeline overview
1) Input: RGB histopathology image (PNG/JPG/TIFF) as path or in-memory array.
2) Segmentation (Stage 1): Attention U-Net predicts a 6-class semantic mask (background + 5 cell types).
3) Instance extraction: From the segmentation logits, extract per-nucleus instances, bounding boxes, and 224×224 patches with context padding.
4) Classification (Stage 2): EfficientNet-based classifier predicts one of the five classes for each nucleus patch with probabilities and confidence.
5) Explainability: For each nucleus patch, Grad-CAM and SHAP heatmaps are generated (when enabled) and saved.
6) Reporting: JSON results, per-nucleus details, segmentation mask image, explanation overlays, and an analysis summary visualization are produced.


### Explainability methods (what the heatmaps mean)
- Grad-CAM (Gradient-weighted Class Activation Mapping): Highlights spatial regions in a patch that most strongly influence the classifier for the predicted class. Hotter regions (reds/yellows) indicate higher contribution to the predicted label.
- SHAP (SHapley Additive exPlanations): Assigns pixel-level contributions (positive or negative) to the prediction based on game-theoretic Shapley values. The combined magnitude heatmap reflects where evidence supporting or opposing the prediction resides.

Together, these help users judge whether the model focuses on plausible nuclear morphology rather than artifacts.


### Installation
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Optional: Set `PANNUKE_ROOT` to your PanNuke dataset root for helper tools.


### Running the Streamlit app
```bash
streamlit run streamlit_app.py
```
In the sidebar:
- Provide segmentation and classifier checkpoint paths (auto-detected if found in `lightning_logs/`).
- Select device (cuda/cpu) and toggle Grad-CAM/SHAP.
- Upload an image; press "Run Analysis".

The app shows the original image, segmentation mask, classification overlay, a table of nuclei, and the saved Grad-CAM/SHAP overlays.


### CLI usage (batch or scriptable)
`two_stage_pipeline.py` also provides a command-line interface:
```bash
python two_stage_pipeline.py \
  --segmentation_model path/to/seg.ckpt \
  --classifier_model path/to/clf.ckpt \
  --input_image path/to/image.png \
  --output_dir outputs \
  --device cuda
```


### Outputs
When running either CLI or Streamlit:
- `analysis_results.json`: High-level metadata and processing info.
- `nuclei_details.json`: Per-nucleus class, confidence, bbox, centroid, agreement with segmentation.
- `segmentation_mask.png`: Colorized semantic mask.
- `analysis_summary.png`: Composite visualization (original, segmentation, boxes+labels, histograms, stats).
- `explanations/`: Per-nucleus `gradcam_*` and `shap_*` overlays and heatmaps.
- `shap_text_summary.json`, `explanations_quality.json`: Aggregate explanation metrics.


### Dataset
The pipeline is built around the PanNuke dataset (3 folds, each with `Images` and `Masks`). The dataset loader supports two storage modes:
- File-per-sample: `*.png` images and `*.npy` six-channel masks.
- Aggregated: `images.npy` and `masks.npy` memmap arrays under each `Part N`.

To export a sample image for testing the app, see the helper in `tools/` (if present) or use your own H&E image that roughly matches PanNuke patch sizes (e.g., 256×256).


### Design notes and constraints
- Segmentation predicts classes including background; classification refines nucleus class labels (background excluded).
- Instance extraction uses area thresholds and context padding; tune via `min_nucleus_area`, `max_nucleus_area`, and `context_padding`.
- Explainability defaults: Grad-CAM enabled; SHAP enabled if `shap` is installed. SHAP uses a small background set for efficiency.
- Models are loaded from Lightning checkpoints; best checkpoints can be auto-detected from `lightning_logs`.


### Troubleshooting
- CUDA/Memory: Switch device to `cpu` in Streamlit sidebar or CLI if you hit GPU memory limits.
- Checkpoint paths: Ensure the provided `.ckpt` files exist; the app will attempt auto-detection if defaults are missing.
- PanNuke data: For training/evaluation tools, ensure the folder contains `Part 1..3` with `Images` and `Masks` or aggregated `.npy` files.
- SHAP slow/failed: Disable SHAP in the UI to speed up inference or if CUDA autograd raises errors.


### References
- PanNuke dataset (multi-organ nuclei segmentation and classification)
- Grad-CAM: Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,” 2017.
- SHAP: Lundberg and Lee, “A Unified Approach to Interpreting Model Predictions,” 2017.


### License
Specify your license here (e.g., MIT). If using PanNuke, follow its data license and citation requirements.


