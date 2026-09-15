"""Tests for PyTorch Dataset, transforms, and multi-label tensor formatting."""

from pathlib import Path
import json
import torch
import pandas as pd
import pytest

from src.dataset import ChestXrayDataset, get_transforms, load_selected_labels

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_CSV = DATA_DIR / "processed" / "labels.csv"
SELECTED_LABELS_JSON = DATA_DIR / "selected_labels.json"


def test_dataset_loading_and_tensor_dimensions():
    """Verify that dataset loads images as 3x224x224 tensors with proper target shapes."""
    assert PROCESSED_CSV.exists()
    labels = load_selected_labels(SELECTED_LABELS_JSON)
    num_classes = len(labels)

    df = pd.read_csv(PROCESSED_CSV)
    train_df = df[df["split"] == "train"]

    dataset = ChestXrayDataset(
        df=train_df,
        data_root=DATA_DIR,
        label_columns=labels,
        transform=get_transforms("train", image_size=224),
    )

    assert len(dataset) > 0

    img_tensor, target_tensor, img_path = dataset[0]

    # Image tensor shape (3, 224, 224)
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 224, 224)
    assert img_tensor.dtype == torch.float32

    # Target tensor shape (num_classes,) and binary values
    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.shape == (num_classes,)
    assert target_tensor.dtype == torch.float32

    # All values in target vector must be 0.0 or 1.0
    unique_vals = set(target_tensor.numpy())
    assert unique_vals.issubset({0.0, 1.0}), f"Target contains non-binary values: {unique_vals}"


def test_deterministic_val_transforms():
    """Verify that validation/test transforms are deterministic."""
    val_transform = get_transforms("val", image_size=224)
    assert val_transform is not None

    df = pd.read_csv(PROCESSED_CSV)
    labels = load_selected_labels(SELECTED_LABELS_JSON)
    val_df = df[df["split"] == "val"]

    dataset = ChestXrayDataset(
        df=val_df,
        data_root=DATA_DIR,
        label_columns=labels,
        transform=val_transform,
    )

    img1, _, _ = dataset[0]
    img2, _, _ = dataset[0]

    # Exactly equal across repeated calls
    assert torch.allclose(img1, img2, atol=1e-5), "Validation transforms are not deterministic!"
