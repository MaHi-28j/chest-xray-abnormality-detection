"""Tests for class-specific threshold tuning and application."""

import numpy as np
import pytest
from src.thresholds import find_optimal_thresholds, apply_thresholds


def test_threshold_search_and_bounds():
    """Verify that optimal thresholds are found and bounded in [0.05, 0.95]."""
    np.random.seed(42)
    n_samples = 200
    classes = ["Atelectasis", "Cardiomegaly", "Effusion"]

    # Synthetic validation data with known correlations
    y_true = np.random.binomial(1, 0.25, (n_samples, len(classes)))
    # Simulated model probabilities correlated with ground truth
    noise = np.random.normal(0, 0.2, (n_samples, len(classes)))
    y_probs = np.clip(y_true * 0.7 + 0.15 + noise, 0.0, 1.0)

    thresholds, metrics = find_optimal_thresholds(y_true, y_probs, classes)

    assert len(thresholds) == len(classes)
    for c in classes:
        assert c in thresholds
        thresh = thresholds[c]
        assert 0.05 <= thresh <= 0.95, f"Threshold {thresh} outside expected range [0.05, 0.95]"
        assert "f1" in metrics[c]
        assert "precision" in metrics[c]
        assert "recall" in metrics[c]


def test_apply_thresholds_logic():
    """Verify that apply_thresholds correctly binarizes predictions based on custom thresholds."""
    classes = ["Atelectasis", "Effusion"]
    thresholds = {"Atelectasis": 0.40, "Effusion": 0.70}

    y_probs = np.array([
        [0.35, 0.65],  # below both -> [0, 0]
        [0.45, 0.65],  # Atelectasis above, Effusion below -> [1, 0]
        [0.55, 0.75],  # above both -> [1, 1]
    ])

    binary_preds = apply_thresholds(y_probs, thresholds, classes)
    expected = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
    ])

    np.testing.assert_array_equal(binary_preds, expected)
