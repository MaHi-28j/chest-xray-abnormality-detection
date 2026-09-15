"""Tests for target-specific Grad-CAM explainability."""

import torch
import numpy as np
import pytest
from PIL import Image

from src.model import build_model
from src.gradcam import GradCAM, overlay_gradcam_on_image


def test_gradcam_generation_and_class_specificity():
    """Verify that Grad-CAM produces a valid 2D heatmap in [0, 1] that differs by class."""
    num_classes = 8
    model = build_model(num_classes=num_classes, pretrained=False)
    model.eval()

    cam_gen = GradCAM(model)

    dummy_image = torch.randn(1, 3, 224, 224, requires_grad=True)

    # Heatmap for class 0
    cam_class0 = cam_gen.generate_heatmap(dummy_image, target_class_idx=0)
    # Heatmap for class 1
    cam_class1 = cam_gen.generate_heatmap(dummy_image, target_class_idx=1)

    assert isinstance(cam_class0, np.ndarray)
    assert cam_class0.ndim == 2
    assert (cam_class0 >= 0.0).all() and (cam_class0 <= 1.0).all()

    # The heatmaps for different classes must not be completely identical
    assert not np.allclose(cam_class0, cam_class1, atol=1e-3), "Grad-CAM is not class-specific!"


def test_gradcam_overlay_dimensions():
    """Verify that overlay produces matching RGB image with correct shape and range."""
    pil_img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    dummy_cam = np.random.uniform(0.0, 1.0, (7, 7))

    overlay, cam_resized = overlay_gradcam_on_image(pil_img, dummy_cam, alpha=0.5)

    assert overlay.shape == (300, 300, 3)
    assert overlay.dtype == np.uint8
    assert cam_resized.shape == (300, 300)
    assert (cam_resized >= 0.0).all() and (cam_resized <= 1.0).all()
