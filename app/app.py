"""Streamlit Web Application: Multi-Label Chest X-Ray Abnormality Detection & Explainable AI.

Features:
- Radiograph upload (PNG, JPG, JPEG) or selection from sample test set cases.
- Multi-label disease probability predictions with threshold status.
- Interactive target-specific Grad-CAM visual explanation generation.
- Medical disclaimer banner.
"""

import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import torch
import matplotlib.pyplot as plt

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from dataset import load_selected_labels
from model import build_model
from thresholds import load_thresholds
from gradcam import GradCAM, overlay_gradcam_on_image
from dataset import IMAGENET_MEAN, IMAGENET_STD
from torchvision import transforms

# Page configuration
st.set_page_config(
    page_title="Chest X-Ray Abnormality Detection (XAI)",
    page_icon="🩻",
    layout="wide",
)

# Paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"
SELECTED_LABELS_PATH = DATA_DIR / "selected_labels.json"
PROCESSED_CSV = DATA_DIR / "processed" / "labels.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_app_model():
    """Load model, weights, selected abnormalities, and tuned thresholds."""
    if not BEST_MODEL_PATH.exists() or not THRESHOLDS_PATH.exists():
        return None, None, None

    labels = load_selected_labels(SELECTED_LABELS_PATH)
    thresholds = load_thresholds(THRESHOLDS_PATH)

    model = build_model(num_classes=len(labels), pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    return model, labels, thresholds


def main():
    st.title("🩻 Multi-Label Chest X-Ray Abnormality Detection")
    st.markdown(
        "**Deep Learning & Explainable AI (Grad-CAM) for Thoracic Pathology Screening**"
    )

    # Mandatory Medical Disclaimer
    st.warning(
        "⚠️ **Disclaimer:** This project is for educational and research demonstration purposes only. "
        "It is **not** a clinical diagnostic tool and must not be used for real-world medical decision making."
    )

    model, labels, thresholds = load_app_model()

    if model is None:
        st.error(
            "Model checkpoint or threshold configuration not found. "
            "Please run `python src/setup_dataset.py` and `python src/train.py` first."
        )
        return

    # Sidebar - Input Selection
    st.sidebar.header("📁 Input Radiograph")
    input_mode = st.sidebar.radio("Select Input Source", ["Choose Sample Radiograph", "Upload Custom Radiograph"])

    selected_image = None
    image_title = ""

    if input_mode == "Choose Sample Radiograph":
        if PROCESSED_CSV.exists():
            df = pd.read_csv(PROCESSED_CSV)
            test_df = df[df["split"] == "test"].reset_index(drop=True)

            sample_options = []
            for i, row in test_df.head(15).iterrows():
                fname = Path(row["image_path"]).name
                findings = row["Finding Labels"]
                sample_options.append(f"{fname} (Findings: {findings})")

            chosen = st.sidebar.selectbox("Select Test Image", sample_options)
            chosen_idx = sample_options.index(chosen)
            row = test_df.iloc[chosen_idx]
            fname = Path(row["image_path"]).name

            img_path = DATA_DIR / "chestxray14" / "images" / fname
            if not img_path.exists():
                img_path = DATA_DIR / row["image_path"]

            if img_path.exists():
                selected_image = Image.open(img_path).convert("RGB")
                image_title = f"{fname} (Ground Truth: {row['Finding Labels']})"
        else:
            st.sidebar.info("Dataset CSV not found. Please upload a radiograph below.")

    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload chest radiograph (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file).convert("RGB")
            image_title = uploaded_file.name

    if selected_image is None:
        st.info("👈 Please select a sample radiograph or upload an image from the sidebar to begin.")
        return

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    input_tensor = transform(selected_image).unsqueeze(0).to(DEVICE)

    # Model Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    # Layout: Two Columns
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("🖼️ Input Radiograph")
        st.image(selected_image, caption=image_title, use_container_width=True)

    with col2:
        st.subheader("📊 Multi-Label Prediction Results")

        pred_rows = []
        for idx, name in enumerate(labels):
            p = float(probs[idx])
            t = float(thresholds.get(name, 0.5))
            detected = p >= t
            pred_rows.append({
                "Abnormality": name,
                "Probability": p,
                "Threshold": t,
                "Status": "🚨 Detected" if detected else "✅ Negative",
            })

        pred_df = pd.DataFrame(pred_rows).sort_values(by="Probability", ascending=False)

        for _, r in pred_df.iterrows():
            abn = r["Abnormality"]
            prob = r["Probability"]
            thresh = r["Threshold"]
            status = r["Status"]

            st.write(f"**{abn}** — `{prob * 100:.1f}%` (Threshold: `{thresh:.2f}`) {status}")
            st.progress(min(max(prob, 0.0), 1.0))

    # Explainable AI Section
    st.markdown("---")
    st.subheader("🔍 Explainable AI: Target-Specific Grad-CAM")
    st.markdown(
        "Select a specific thoracic abnormality to visualize which anatomical regions "
        "of the radiograph contributed most strongly to the model's prediction."
    )

    selected_abnormality = st.selectbox("Select Abnormality to Explain", labels, index=0)
    target_idx = labels.index(selected_abnormality)

    # Compute Grad-CAM
    cam_gen = GradCAM(model)
    cam_heatmap = cam_gen.generate_heatmap(input_tensor, target_class_idx=target_idx)
    overlay_rgb, cam_resized = overlay_gradcam_on_image(selected_image, cam_heatmap, alpha=0.45)

    cam_col1, cam_col2, cam_col3 = st.columns(3)

    with cam_col1:
        st.image(selected_image, caption="Original Radiograph", use_container_width=True)

    with cam_col2:
        # Colormap Heatmap
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cam_resized, cmap="jet")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.caption(f"Grad-CAM Heatmap for **{selected_abnormality}**")

    with cam_col3:
        st.image(
            overlay_rgb,
            caption=f"Grad-CAM Overlay ({selected_abnormality}: {probs[target_idx]*100:.1f}%)",
            use_container_width=True
        )

    # Technical Details Accordion
    with st.expander("ℹ️ How does this system work?"):
        st.markdown("""
        - **Architecture:** Torchvision ResNet-18 pretrained on ImageNet, fine-tuned on NIH ChestX-ray14.
        - **Multi-Label Formulation:** Uses independent Sigmoid units with class-weighted Binary Cross-Entropy (`BCEWithLogitsLoss`).
        - **Threshold Tuning:** Decision thresholds are tuned specifically per abnormality on the validation set to maximize F1 score.
        - **Grad-CAM (Selvaraju et al., 2017):** Gradients of the target abnormality logit with respect to feature maps in `layer4` are globally pooled to weigh feature activations, highlighting regions of interest.
        """)


if __name__ == "__main__":
    main()
