"""Threshold tuning module for multi-label classification.

Optimizes decision thresholds per abnormality using ONLY the validation set.
In imbalanced multi-label tasks, the default threshold of 0.5 often leads
to low recall for less frequent abnormalities. Tuning per-class thresholds
maximizes the validation F1 score without touching the test set.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    min_thresh: float = 0.05,
    max_thresh: float = 0.95,
    num_steps: int = 46,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Search for the threshold maximizing F1 score for each class on validation data.

    Args:
        y_true: Binary ground truth array of shape (N, num_classes).
        y_probs: Predicted probabilities array of shape (N, num_classes).
        class_names: List of class names corresponding to columns.
        min_thresh: Minimum threshold to evaluate.
        max_thresh: Maximum threshold to evaluate.
        num_steps: Number of threshold steps between min_thresh and max_thresh.

    Returns:
        optimal_thresholds: Dict mapping class_name -> best threshold float.
        threshold_metrics: Dict mapping class_name -> {'f1', 'precision', 'recall', 'threshold'}.
    """
    candidate_thresholds = np.linspace(min_thresh, max_thresh, num_steps)
    optimal_thresholds = {}
    threshold_metrics = {}

    for col_idx, class_name in enumerate(class_names):
        y_col_true = y_true[:, col_idx]
        y_col_prob = y_probs[:, col_idx]

        best_f1 = -1.0
        best_thresh = 0.5
        best_prec = 0.0
        best_rec = 0.0

        # If class has 0 positive examples in validation set, default to 0.5
        if np.sum(y_col_true) == 0:
            optimal_thresholds[class_name] = 0.5
            threshold_metrics[class_name] = {
                "threshold": 0.5,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }
            continue

        for thresh in candidate_thresholds:
            y_col_pred = (y_col_prob >= thresh).astype(int)
            score = f1_score(y_col_true, y_col_pred, zero_division=0)

            if score > best_f1:
                best_f1 = score
                best_thresh = float(thresh)
                best_prec = float(precision_score(y_col_true, y_col_pred, zero_division=0))
                best_rec = float(recall_score(y_col_true, y_col_pred, zero_division=0))

        # Fallback if no threshold gave positive F1
        if best_f1 <= 0:
            best_thresh = 0.5

        optimal_thresholds[class_name] = round(best_thresh, 3)
        threshold_metrics[class_name] = {
            "threshold": round(best_thresh, 3),
            "f1": round(best_f1, 4),
            "precision": round(best_prec, 4),
            "recall": round(best_rec, 4),
        }

    return optimal_thresholds, threshold_metrics


def apply_thresholds(
    y_probs: np.ndarray,
    thresholds: Dict[str, float],
    class_names: List[str],
) -> np.ndarray:
    """Apply class-specific thresholds to predicted probabilities to produce binary predictions.

    Args:
        y_probs: Predicted probabilities of shape (N, num_classes).
        thresholds: Dict mapping class_name -> threshold value.
        class_names: List of class names.

    Returns:
        Binary predictions array of shape (N, num_classes).
    """
    y_pred = np.zeros_like(y_probs, dtype=int)
    for idx, name in enumerate(class_names):
        t = thresholds.get(name, 0.5)
        y_pred[:, idx] = (y_probs[:, idx] >= t).astype(int)
    return y_pred


def save_thresholds(thresholds: Dict[str, float], filepath: Path):
    """Save tuned thresholds to JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(thresholds, f, indent=2)


def load_thresholds(filepath: Path) -> Dict[str, float]:
    """Load tuned thresholds from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)
