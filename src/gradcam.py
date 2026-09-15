"""Target-Specific Grad-CAM (Gradient-Weighted Class Activation Mapping) for ResNet-18.

Explains individual thoracic abnormality predictions by computing the gradient
of a targeted class logit with respect to feature maps in the final convolutional
layer of ResNet-18 (layer4[-1].conv2).

Crucial requirement:
Grad-CAM must target a specific abnormality logit, generating unique heatmaps
per abnormality rather than one generic heatmap for the whole image.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from dataset import IMAGENET_MEAN, IMAGENET_STD, load_selected_labels
from model import build_model
from thresholds import load_thresholds

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"
SELECTED_LABELS_PATH = DATA_DIR / "selected_labels.json"
RESULTS_DIR = BASE_DIR / "results"
GRADCAM_DIR = RESULTS_DIR / "figures" / "gradcam"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradCAM:
    """Computes target-specific Grad-CAM heatmaps for ChestXrayResNet18."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # Default to the final convolutional layer of ResNet-18
        if target_layer is None:
            self.target_layer = self.model.backbone.layer4[-1].conv2
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int,
    ) -> np.ndarray:
        """Generate a 2D Grad-CAM heatmap for a specific class index.

        Args:
            input_tensor: Image tensor of shape (1, 3, H, W).
            target_class_idx: Integer index of the abnormality to explain.

        Returns:
            2D numpy array heatmap of shape (H, W) with values normalized in [0, 1].
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)

        # Target specific abnormality logit
        target_score = logits[0, target_class_idx]
        target_score.backward(retain_graph=True)

        # Global average pooling of gradients: weights alpha_k
        gradients = self.gradients[0]  # (C, h, w)
        activations = self.activations[0]  # (C, h, w)

        weights = torch.mean(gradients, dim=(1, 2))  # (C,)

        # Linear combination of feature maps weighted by alpha
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU to retain only features having positive influence on the target
        cam = torch.relu(cam)

        cam_np = cam.cpu().numpy()

        # Normalize to [0, 1]
        max_val = np.max(cam_np)
        if max_val > 0:
            cam_np = cam_np / max_val
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


def overlay_gradcam_on_image(
    original_pil: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize CAM to original image dimensions and blend with colormap overlay.

    Args:
        original_pil: Original PIL image.
        cam: 2D numpy heatmap in [0, 1].
        alpha: Overlay blending opacity.
        colormap_name: Matplotlib colormap ('jet', 'inferno', etc.)

    Returns:
        overlay_rgb: Blended uint8 image (H, W, 3).
        cam_resized: Scaled heatmap (H, W) in [0, 1].
    """
    img_w, img_h = original_pil.size
    pil_cam = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (img_w, img_h), resample=Image.BILINEAR
    )
    cam_resized = np.array(pil_cam, dtype=np.float32) / 255.0

    # Colorize heatmap
    colormap = plt.colormaps[colormap_name]
    colored_heatmap = colormap(cam_resized)[:, :, :3]  # (H, W, 3) in [0, 1]

    # Normalize original image to RGB float
    orig_np = np.array(original_pil.convert("RGB"), dtype=np.float32) / 255.0

    # Alpha blend
    overlay = (1.0 - alpha) * orig_np + alpha * colored_heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    overlay_rgb = (overlay * 255).astype(np.uint8)

    return overlay_rgb, cam_resized


def explain_image(
    image_path: Union[str, Path],
    target_abnormality: str,
    model: Optional[nn.Module] = None,
    save_path: Optional[Path] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """High-level function to predict and generate Grad-CAM for a given chest radiograph.

    Args:
        image_path: Path to chest radiograph.
        target_abnormality: Name of abnormality to explain (must be in selected_labels).
        model: Optional pre-loaded model.
        save_path: Optional path to save visual figure.

    Returns:
        prob_dict: All abnormality predicted probabilities.
        overlay_img: RGB overlay image.
        cam_resized: Normalized heatmap.
    """
    selected_labels = load_selected_labels(SELECTED_LABELS_PATH)
    if target_abnormality not in selected_labels:
        raise ValueError(f"Abnormality '{target_abnormality}' not in selected list: {selected_labels}")

    target_idx = selected_labels.index(target_abnormality)

    # Load model if not passed
    if model is None:
        model = build_model(num_classes=len(selected_labels), pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Preprocess image
    orig_pil = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    input_tensor = transform(orig_pil).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    prob_dict = {name: float(probs[i]) for i, name in enumerate(selected_labels)}

    # Compute Grad-CAM
    cam_generator = GradCAM(model)
    cam_raw = cam_generator.generate_heatmap(input_tensor, target_class_idx=target_idx)
    overlay_rgb, cam_resized = overlay_gradcam_on_image(orig_pil, cam_raw, alpha=0.45)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        axes[0].imshow(orig_pil, cmap="gray")
        axes[0].set_title(f"Original Chest Radiograph\n({Path(image_path).name})", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(cam_resized, cmap="jet")
        axes[1].set_title(f"Grad-CAM Heatmap\nTarget: {target_abnormality}", fontsize=10, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(overlay_rgb)
        axes[2].set_title(
            f"Explanation Overlay\n{target_abnormality} Prob: {prob_dict[target_abnormality]*100:.1f}%",
            fontsize=10,
            fontweight="bold",
        )
        axes[2].axis("off")

        plt.suptitle(
            f"Explainable AI: Target-Specific Grad-CAM for {target_abnormality}",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Grad-CAM explanation saved to {save_path}")

    return prob_dict, overlay_rgb, cam_resized


def generate_sample_explanations():
    """Generate demonstration Grad-CAM explanations for selected test images."""
    print("=" * 65)
    print("  GENERATING GRAD-CAM DEMONSTRATION EXPLANATIONS")
    print("=" * 65)

    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    selected_labels = load_selected_labels(SELECTED_LABELS_PATH)
    processed_csv = DATA_DIR / "processed" / "labels.csv"

    if not processed_csv.exists() or not BEST_MODEL_PATH.exists():
        print("Required model or dataset not found. Skipping sample generation.")
        return

    df = pd.read_csv(processed_csv)
    test_df = df[df["split"] == "test"]

    model = build_model(num_classes=len(selected_labels), pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

    # Pick 3 different positive test images for prominent abnormalities
    demonstration_targets = ["Cardiomegaly", "Effusion", "Atelectasis", "Infiltration"]

    for target in demonstration_targets:
        if target in selected_labels:
            pos_cases = test_df[test_df[target] == 1]
            if len(pos_cases) > 0:
                sample_row = pos_cases.iloc[0]
                img_path = DATA_DIR / sample_row["image_path"]
                if not img_path.exists():
                    fname = Path(sample_row["image_path"]).name
                    img_path = DATA_DIR / "chestxray14" / "images" / fname

                if img_path.exists():
                    out_path = GRADCAM_DIR / f"gradcam_{target.lower()}.png"
                    explain_image(img_path, target, model=model, save_path=out_path)


if __name__ == "__main__":
    generate_sample_explanations()
