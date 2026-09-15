"""Final held-out test evaluation module.

Evaluates the final best model exactly ONCE on the untouched test set.
Computes:
- Overall: Micro/Macro F1, Precision, Recall, Macro ROC-AUC
- Per Class: F1, Precision, Recall, ROC-AUC
- Comparison: Default Threshold (0.5) vs Validation-Tuned Thresholds
- Figures: ROC curves, Precision-Recall curves, Per-Class 2x2 Confusion Matrices
"""

import json
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt

from dataset import create_dataloaders, load_selected_labels
from model import build_model
from thresholds import load_thresholds, apply_thresholds

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_CSV = DATA_DIR / "processed" / "labels.csv"
SELECTED_LABELS_JSON = DATA_DIR / "selected_labels.json"
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"

RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_test_set():
    """Run thorough evaluation on the held-out test set."""
    print("=" * 65)
    print("  HELD-OUT TEST SET EVALUATION")
    print("=" * 65)
    print(f"Device: {DEVICE}")

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {BEST_MODEL_PATH}. Please run train.py first.")
    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(f"Thresholds not found at {THRESHOLDS_PATH}. Please run train.py first.")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    selected_labels = load_selected_labels(SELECTED_LABELS_JSON)
    thresholds = load_thresholds(THRESHOLDS_PATH)
    num_classes = len(selected_labels)

    # 1. Load Data
    _, _, test_loader = create_dataloaders(
        csv_path=PROCESSED_CSV,
        data_root=DATA_DIR,
        label_columns=selected_labels,
        batch_size=16,
    )
    print(f"Test samples: {len(test_loader.dataset)} from held-out test split.")

    # 2. Load Model
    model = build_model(num_classes=num_classes, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 3. Collect Predictions
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    y_true = np.vstack(all_targets)
    y_probs = np.vstack(all_probs)

    # 4. Predictions with Default (0.5) and Tuned Thresholds
    y_pred_default = (y_probs >= 0.5).astype(int)
    y_pred_tuned = apply_thresholds(y_probs, thresholds, selected_labels)

    # Compute overall metrics
    def calc_overall(preds):
        return {
            "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y_true, preds, average="micro", zero_division=0)),
            "macro_precision": float(precision_score(y_true, preds, average="macro", zero_division=0)),
            "micro_precision": float(precision_score(y_true, preds, average="micro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, preds, average="macro", zero_division=0)),
            "micro_recall": float(recall_score(y_true, preds, average="micro", zero_division=0)),
        }

    overall_default = calc_overall(y_pred_default)
    overall_tuned = calc_overall(y_pred_tuned)

    # Macro ROC-AUC
    auc_per_class = []
    ap_per_class = []
    for c in range(num_classes):
        if len(np.unique(y_true[:, c])) > 1:
            auc = roc_auc_score(y_true[:, c], y_probs[:, c])
            ap = average_precision_score(y_true[:, c], y_probs[:, c])
        else:
            auc = 0.5
            ap = 0.0
        auc_per_class.append(auc)
        ap_per_class.append(ap)

    macro_auc = float(np.mean(auc_per_class))
    overall_default["macro_roc_auc"] = macro_auc
    overall_tuned["macro_roc_auc"] = macro_auc

    print("\n" + "-" * 60)
    print("  OVERALL TEST PERFORMANCE")
    print("-" * 60)
    print(f"  Metric              | Default (0.5) | Tuned Thresholds")
    print("-" * 60)
    print(f"  Macro ROC-AUC       | {macro_auc:.4f}        | {macro_auc:.4f}")
    print(f"  Macro F1            | {overall_default['macro_f1']:.4f}        | {overall_tuned['macro_f1']:.4f}")
    print(f"  Micro F1            | {overall_default['micro_f1']:.4f}        | {overall_tuned['micro_f1']:.4f}")
    print(f"  Macro Precision     | {overall_default['macro_precision']:.4f}        | {overall_tuned['macro_precision']:.4f}")
    print(f"  Micro Precision     | {overall_default['micro_precision']:.4f}        | {overall_tuned['micro_precision']:.4f}")
    print(f"  Macro Recall        | {overall_default['macro_recall']:.4f}        | {overall_tuned['macro_recall']:.4f}")
    print(f"  Micro Recall        | {overall_default['micro_recall']:.4f}        | {overall_tuned['micro_recall']:.4f}")
    print("-" * 60)

    # 5. Per-Class Metrics Table
    per_class_rows = []
    for c, name in enumerate(selected_labels):
        thresh = thresholds.get(name, 0.5)
        p_tuned = precision_score(y_true[:, c], y_pred_tuned[:, c], zero_division=0)
        r_tuned = recall_score(y_true[:, c], y_pred_tuned[:, c], zero_division=0)
        f1_tuned = f1_score(y_true[:, c], y_pred_tuned[:, c], zero_division=0)

        p_def = precision_score(y_true[:, c], y_pred_default[:, c], zero_division=0)
        r_def = recall_score(y_true[:, c], y_pred_default[:, c], zero_division=0)
        f1_def = f1_score(y_true[:, c], y_pred_default[:, c], zero_division=0)

        per_class_rows.append({
            "abnormality": name,
            "tuned_threshold": round(thresh, 2),
            "test_positives": int(y_true[:, c].sum()),
            "roc_auc": round(auc_per_class[c], 4),
            "pr_auc_ap": round(ap_per_class[c], 4),
            "f1_tuned": round(f1_tuned, 4),
            "precision_tuned": round(p_tuned, 4),
            "recall_tuned": round(r_tuned, 4),
            "f1_default_0.5": round(f1_def, 4),
            "precision_default": round(p_def, 4),
            "recall_default": round(r_def, 4),
        })

    df_metrics = pd.DataFrame(per_class_rows)
    metrics_csv = METRICS_DIR / "final_metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"\nPer-class metrics saved to {metrics_csv}:\n")
    print(df_metrics[["abnormality", "test_positives", "roc_auc", "f1_tuned", "precision_tuned", "recall_tuned"]].to_string(index=False))

    # =========================================================================
    # PLOTTING: ROC CURVES
    # =========================================================================
    plt.figure(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for c, (name, color) in enumerate(zip(selected_labels, colors)):
        if len(np.unique(y_true[:, c])) > 1:
            fpr, tpr, _ = roc_curve(y_true[:, c], y_probs[:, c])
            plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc_per_class[c]:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    plt.title(f"Held-Out Test ROC Curves (Macro AUC = {macro_auc:.3f})", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.6)

    roc_fig_path = FIGURES_DIR / "roc_curves.png"
    plt.savefig(roc_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC curves saved to {roc_fig_path}")

    # =========================================================================
    # PLOTTING: PRECISION-RECALL CURVES
    # =========================================================================
    plt.figure(figsize=(9, 7))
    for c, (name, color) in enumerate(zip(selected_labels, colors)):
        if len(np.unique(y_true[:, c])) > 1:
            prec, rec, _ = precision_recall_curve(y_true[:, c], y_probs[:, c])
            plt.plot(rec, prec, color=color, lw=2, label=f"{name} (AP = {ap_per_class[c]:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11, fontweight="bold")
    plt.ylabel("Precision", fontsize=11, fontweight="bold")
    plt.title("Held-Out Test Precision-Recall Curves", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.6)

    pr_fig_path = FIGURES_DIR / "pr_curves.png"
    plt.savefig(pr_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PR curves saved to {pr_fig_path}")

    # =========================================================================
    # PLOTTING: PER-CLASS 2X2 CONFUSION MATRICES (Multi-label appropriate)
    # =========================================================================
    cols_grid = 4
    rows_grid = int(np.ceil(num_classes / cols_grid))
    fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(14, 3.2 * rows_grid))
    axes = axes.flatten()

    for c, name in enumerate(selected_labels):
        cm = confusion_matrix(y_true[:, c], y_pred_tuned[:, c], labels=[0, 1])
        ax = axes[c]
        cax = ax.matshow(cm, cmap="Blues", alpha=0.7)

        for i in range(2):
            for j in range(2):
                ax.text(x=j, y=i, s=f"{cm[i, j]}", va="center", ha="center", fontsize=11, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticklabels(["Neg", "Pos"])
        ax.set_title(f"{name}\n(thresh={thresholds.get(name, 0.5):.2f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)

    # Hide any unused subplots
    for k in range(num_classes, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    cm_fig_path = FIGURES_DIR / "per_class_confusion_matrices.png"
    plt.savefig(cm_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Per-class confusion matrices saved to {cm_fig_path}")

    print("\n" + "=" * 65)
    print("  TEST EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_test_set()
