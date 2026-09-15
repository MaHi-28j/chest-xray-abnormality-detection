"""Tests for dataset split integrity, data leakage prevention, and manifest validity."""

from pathlib import Path
import json
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_CSV = DATA_DIR / "processed" / "labels.csv"
MANIFEST_JSON = DATA_DIR / "chestxray14" / "dataset_manifest.json"
SELECTED_LABELS_JSON = DATA_DIR / "selected_labels.json"


def test_manifest_exists_and_valid():
    """Verify that dataset manifest exists and has valid status."""
    assert MANIFEST_JSON.exists(), f"Manifest missing at {MANIFEST_JSON}"
    with open(MANIFEST_JSON, "r") as f:
        manifest = json.load(f)

    assert manifest["status"] == "READY"
    assert manifest["available_local_images"] > 0
    assert manifest["corrupted_images_detected"] == 0
    assert len(manifest["all_14_labels_present"]) == 14


def test_selected_labels_count_and_format():
    """Verify that 6-8 abnormalities are selected."""
    assert SELECTED_LABELS_JSON.exists(), f"Selected labels file missing at {SELECTED_LABELS_JSON}"
    with open(SELECTED_LABELS_JSON, "r") as f:
        labels = json.load(f)

    assert isinstance(labels, list)
    assert 6 <= len(labels) <= 8, f"Expected 6-8 selected labels, got {len(labels)}"
    for label in labels:
        assert isinstance(label, str) and len(label) > 0


def test_processed_splits_exist_and_no_duplicates():
    """Verify processed CSV exists and contains valid splits with no duplicate images."""
    assert PROCESSED_CSV.exists(), f"Processed CSV missing at {PROCESSED_CSV}"
    df = pd.read_csv(PROCESSED_CSV)

    assert "image_path" in df.columns
    assert "split" in df.columns
    assert "Patient ID" in df.columns

    # No duplicate images
    assert df["image_path"].duplicated().sum() == 0, "Duplicate images found in processed dataset!"

    # All three splits exist
    splits_present = set(df["split"].unique())
    assert {"train", "val", "test"}.issubset(splits_present)


def test_zero_patient_leakage():
    """CRITICAL TEST: Verify zero patient leakage across train, val, and test splits."""
    assert PROCESSED_CSV.exists()
    df = pd.read_csv(PROCESSED_CSV)

    train_pts = set(df[df["split"] == "train"]["Patient ID"].unique())
    val_pts = set(df[df["split"] == "val"]["Patient ID"].unique())
    test_pts = set(df[df["split"] == "test"]["Patient ID"].unique())

    # Intersection checks
    train_val_overlap = train_pts.intersection(val_pts)
    train_test_overlap = train_pts.intersection(test_pts)
    val_test_overlap = val_pts.intersection(test_pts)

    assert len(train_val_overlap) == 0, f"DATA LEAKAGE: Patients overlap between Train and Val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"DATA LEAKAGE: Patients overlap between Train and Test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"DATA LEAKAGE: Patients overlap between Val and Test: {val_test_overlap}"
