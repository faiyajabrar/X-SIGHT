import os
import io
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import cv2
import streamlit as st

from two_stage_pipeline import TwoStageNucleiPipeline, find_best_models


@st.cache_resource(show_spinner=False)
def load_pipeline(seg_model_path: str, clf_model_path: str, device: str,
                  enable_gradcam: bool, enable_shap: bool) -> TwoStageNucleiPipeline:
    return TwoStageNucleiPipeline(
        segmentation_model_path=seg_model_path,
        classifier_model_path=clf_model_path,
        device=device,
        enable_gradcam=enable_gradcam,
        enable_shap=enable_shap,
    )


def colorize_segmentation(segmentation_mask: np.ndarray) -> np.ndarray:
    # Colors must align with pipeline class colors
    # We'll import lazily via a dummy pipe to access the palette if needed
    from two_stage_pipeline import TwoStageNucleiPipeline as _Pipe
    dummy_colors = np.array([
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
    ], dtype=np.uint8)
    if segmentation_mask.ndim == 3:
        # already colored
        return segmentation_mask
    colored = dummy_colors[segmentation_mask]
    return colored


def overlay_classifications(image_rgb: np.ndarray, nuclei_instances: list, class_colors: np.ndarray) -> np.ndarray:
    vis = image_rgb.copy()
    for nucleus in nuclei_instances:
        y1, x1, y2, x2 = nucleus['bbox']
        class_id = int(nucleus['class_id'])
        color = class_colors[class_id].tolist()
        cv2.rectangle(vis, (x1, y1), (x2, y2), color=color, thickness=2)
        label = f"{nucleus['class_name']} {nucleus['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color=(255, 255, 255), thickness=-1)
        cv2.putText(vis, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def main():
    st.set_page_config(page_title="Two-Stage Nuclei Analysis", layout="wide")
    st.title("Two-Stage Nuclei Analysis Pipeline")

    with st.sidebar:
        st.header("Models & Settings")
        detected = find_best_models()
        seg_default = detected.get('segmentation', 'lightning_logs/segmentation/version_1/checkpoints/advanced-epoch=12-val_dice=0.656.ckpt')
        clf_default = detected.get('classifier', 'lightning_logs/classifier/classifier_efficientnet_b3_20250727_002713/version_0/checkpoints/classifier-epoch=19-val_f1=0.806.ckpt')

        seg_model = st.text_input("Segmentation model (.ckpt)", value=seg_default)
        clf_model = st.text_input("Classifier model (.ckpt)", value=clf_default)
        device = st.selectbox("Device", options=["cuda", "cpu"], index=0)
        enable_gradcam = st.checkbox("Enable Grad-CAM", value=True)
        enable_shap = st.checkbox("Enable SHAP", value=True)

        st.divider()
        st.subheader("Nucleus extraction")
        min_area = st.number_input("Min nucleus area (px)", min_value=0, max_value=100000, value=50, step=10)
        max_area = st.number_input("Max nucleus area (px)", min_value=0, max_value=500000, value=5000, step=100)
        context_padding = st.number_input("Context padding (px)", min_value=0, max_value=512, value=32, step=4)

        st.divider()
        output_dir = st.text_input("Output directory", value="streamlit_outputs")
        save_results = st.checkbox("Save results to disk", value=True)

    uploaded = st.file_uploader("Upload histopathology image", type=["png", "jpg", "jpeg", "tif", "tiff"])

    if uploaded is None:
        st.info("Upload an image to run the pipeline.")
        return

    # Read as RGB numpy array
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Failed to read the uploaded image.")
        return
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Initialize pipeline (cached)
    try:
        pipeline = load_pipeline(seg_model, clf_model, device, enable_gradcam, enable_shap)
        # Update extraction params that are not part of cache key
        pipeline.min_nucleus_area = int(min_area)
        pipeline.max_nucleus_area = int(max_area)
        pipeline.context_padding = int(context_padding)
    except Exception as e:
        st.exception(e)
        return

    run = st.button("Run Analysis", type="primary")
    if not run:
        st.stop()

    with st.spinner("Running two-stage analysis..."):
        try:
            results: Dict[str, Any] = pipeline.analyze_image_array(
                image=image_rgb,
                save_results=save_results,
                output_dir=output_dir,
                visualize=False,
            )
        except Exception as e:
            st.exception(e)
            return

    st.success(f"Found {results['nuclei_count']} nuclei")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image_rgb, channels="RGB", width='stretch')

    with col2:
        st.subheader("Segmentation Mask")
        seg_mask_np = results['segmentation_mask']
        seg_color = colorize_segmentation(seg_mask_np)
        st.image(seg_color, channels="RGB", width='stretch')

    # Overlay with classifications
    class_colors = np.array([
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
    ], dtype=np.uint8)
    overlay = overlay_classifications(image_rgb, results['nuclei_instances'], class_colors)
    st.subheader("Classification Overlay")
    st.image(overlay, channels="RGB", width='stretch')

    # Per-nucleus table
    if len(results['nuclei_instances']) > 0:
        import pandas as pd
        rows = []
        for n in results['nuclei_instances']:
            rows.append({
                'id': int(n['instance_id']),
                'class': n['class_name'],
                'confidence': float(n['confidence']),
                'bbox': n['bbox'],
                'area': int(n['area']),
            })
        df = pd.DataFrame(rows).sort_values(by=['confidence'], ascending=False)
        st.subheader("Nuclei")
        st.dataframe(df, use_container_width=True)

    # Explanations gallery
    if save_results:
        exp_dir = Path(output_dir) / 'explanations'
        if exp_dir.exists():
            st.subheader("Explanations (Grad-CAM / SHAP)")
            images = sorted(exp_dir.glob("*.png"))
            if images:
                thumbs = []
                for p in images:
                    img = cv2.imread(str(p))
                    if img is None:
                        continue
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    thumbs.append((p.name, rgb))
                # Display in grid
                cols = st.columns(4)
                for idx, (name, rgb) in enumerate(thumbs):
                    with cols[idx % 4]:
                        st.image(rgb, caption=name, width='stretch')


if __name__ == "__main__":
    main()


