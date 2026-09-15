"""Tests for ResNet-18 model architecture, freezing logic, and probability bounds."""

import torch
import pytest
from src.model import build_model, ChestXrayResNet18


def test_model_forward_shape_and_sigmoid_range():
    """Verify model output shapes and that sigmoid probabilities lie strictly in [0, 1]."""
    num_classes = 8
    model = build_model(num_classes=num_classes, pretrained=False)
    model.eval()

    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    # Forward pass produces raw logits
    logits = model(dummy_input)
    assert logits.shape == (batch_size, num_classes)

    # Probabilities via Sigmoid
    probs = model.predict_probabilities(dummy_input)
    assert probs.shape == (batch_size, num_classes)
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities outside [0, 1]!"


def test_model_freezing_and_unfreezing():
    """Verify that freeze_backbone and unfreeze_final_blocks set requires_grad properly."""
    model = build_model(num_classes=8, pretrained=False)

    # Stage 1: Freeze backbone
    model.freeze_backbone()
    trainable_params_stage1 = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Only the fc layer should be trainable: 512 * 8 + 8 = 4104
    assert trainable_params_stage1 == 512 * 8 + 8

    # Stage 2: Unfreeze layer4 + fc
    model.unfreeze_final_blocks(["layer4", "fc"])
    trainable_params_stage2 = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert trainable_params_stage2 > trainable_params_stage1
