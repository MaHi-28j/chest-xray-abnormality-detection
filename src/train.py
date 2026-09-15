"""Two-Stage Training Pipeline for Multi-Label Thoracic Abnormality Detection.

Implements:
1. Class-specific positive weights for BCEWithLogitsLoss.
2. Stage 1: Frozen backbone (train linear classification head).
3. Stage 2: Fine-tune final residual blocks (layer4 + head) with smaller learning rate.
4. Validation tracking: Loss, Micro F1, Macro F1, Macro ROC-AUC.
5. Best model checkpointing & Early Stopping.
6. Validation-based threshold tuning.
7. Experiment comparison: Frozen vs Fine-Tuned (saved to CSV and figure).
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt

from dataset import create_dataloaders, load_selected_labels
from model import build_model
from thresholds import find_optimal_thresholds, save_thresholds, apply_thresholds

# Reproducibility seeds
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_CSV = DATA_DIR / "processed" / "labels.csv"
SELECTED_LABELS_JSON = DATA_DIR / "selected_labels.json"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
EXPERIMENTS_DIR = RESULTS_DIR / "experiment_results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_positive_weights(train_df: pd.DataFrame, class_names: List[str]) -> torch.Tensor:
    """Calculate class-specific positive weights strictly from the training split.

    Formula:
        pos_weight[c] = (N_total_train - N_positive_train[c]) / N_positive_train[c]

    Rationale:
        In thoracic multi-label datasets, most findings are present in fewer than 10-20%
        of cases. Standard binary cross entropy is dominated by the prevalent negative class,
        incentivizing the network to predict near-zero probabilities for all conditions.
        Positive weighting scales up the loss contribution of true positives, balancing
        the gradient updates and enabling the network to learn subtle pathology features.
    """
    total_samples = len(train_df)
    weights = []
    weights_dict = {}

    print("\nTraining Split Class Weights (Weighted BCE):")
    print("-" * 55)
    for name in class_names:
        num_pos = train_df[name].sum()
        num_neg = total_samples - num_pos
        w = float(num_neg / max(num_pos, 1))
        weights.append(w)
        weights_dict[name] = round(w, 3)
        print(f"  {name:<18} | Pos: {int(num_pos):>4} | Neg: {int(num_neg):>4} | Weight: {w:.2f}")
    print("-" * 55)

    pos_weights_file = MODELS_DIR / "pos_weights.json"
    with open(pos_weights_file, "w") as f:
        json.dump(weights_dict, f, indent=2)
    print(f"Positive weights saved to {pos_weights_file}\n")

    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute multi-label classification metrics."""
    y_pred = (y_probs >= threshold).astype(int)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    # ROC-AUC per column, skipping any with only 1 class in ground truth
    auc_scores = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) > 1:
            try:
                score = roc_auc_score(y_true[:, c], y_probs[:, c])
                auc_scores.append(score)
            except Exception:
                pass

    macro_auc = float(np.mean(auc_scores)) if auc_scores else 0.5

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_auc": macro_auc,
    }


def run_epoch(model, loader, criterion, optimizer=None, is_training: bool = True):
    """Run one training or validation epoch."""
    if is_training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_targets = []
    all_probs = []

    for images, targets, _ in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, targets)

            if is_training:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    metrics = compute_metrics(all_targets, all_probs, threshold=0.5)
    metrics["loss"] = epoch_loss

    return epoch_loss, metrics, all_probs, all_targets


def train_pipeline(
    epochs_stage1: int = 4,
    epochs_stage2: int = 4,
    batch_size: int = 16,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-4,
):
    """Execute the full two-stage training workflow."""
    print("=" * 65)
    print("  TWO-STAGE RESNET-18 TRAINING PIPELINE")
    print("=" * 65)
    print(f"Device: {DEVICE}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    selected_labels = load_selected_labels(SELECTED_LABELS_JSON)
    num_classes = len(selected_labels)
    df_all = pd.read_csv(PROCESSED_CSV)
    train_df = df_all[df_all["split"] == "train"]

    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=PROCESSED_CSV,
        data_root=DATA_DIR,
        label_columns=selected_labels,
        batch_size=batch_size,
    )
    print(f"Loaded {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test samples.")

    # 2. Loss with Positive Weights
    pos_weights = calculate_positive_weights(train_df, selected_labels)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # 3. Initialize Model
    model = build_model(num_classes=num_classes, pretrained=True).to(DEVICE)

    history = {
        "epoch": [],
        "stage": [],
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_micro_f1": [],
        "val_macro_auc": [],
    }

    # =========================================================================
    # STAGE 1: FROZEN BACKBONE (Head Only)
    # =========================================================================
    print("\n" + "#" * 60)
    print("  STAGE 1: Training Classification Head with Frozen Backbone")
    print("#" * 60)
    model.freeze_backbone()
    print(f"Trainable parameters: {model.count_trainable_parameters():,}")

    optimizer_stage1 = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_stage1, weight_decay=1e-4)

    t0_stage1 = time.time()
    for epoch in range(1, epochs_stage1 + 1):
        tr_loss, _, _, _ = run_epoch(model, train_loader, criterion, optimizer_stage1, is_training=True)
        va_loss, va_metrics, _, _ = run_epoch(model, val_loader, criterion, is_training=False)

        history["epoch"].append(epoch)
        history["stage"].append("Stage 1 (Frozen)")
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_macro_f1"].append(va_metrics["macro_f1"])
        history["val_micro_f1"].append(va_metrics["micro_f1"])
        history["val_macro_auc"].append(va_metrics["macro_auc"])

        print(
            f"Stage 1 | Epoch {epoch}/{epochs_stage1} | "
            f"Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | "
            f"Val Macro F1: {va_metrics['macro_f1']:.4f} | Val ROC-AUC: {va_metrics['macro_auc']:.4f}"
        )

    t_stage1 = time.time() - t0_stage1

    # Save frozen stage checkpoint
    frozen_model_path = MODELS_DIR / "frozen_model.pth"
    torch.save(model.state_dict(), frozen_model_path)
    print(f"Stage 1 model saved to {frozen_model_path} (Training time: {t_stage1:.1f}s)")

    # Record Stage 1 final validation metrics for Experiment A
    stage1_val_loss, stage1_metrics, _, _ = run_epoch(model, val_loader, criterion, is_training=False)

    # =========================================================================
    # STAGE 2: FINE-TUNING FINAL LAYERS
    # =========================================================================
    print("\n" + "#" * 60)
    print("  STAGE 2: Fine-Tuning Final Residual Blocks (layer4 + fc)")
    print("#" * 60)
    model.unfreeze_final_blocks(["layer4", "fc"])
    print(f"Trainable parameters: {model.count_trainable_parameters():,}")

    optimizer_stage2 = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_stage2, weight_decay=1e-4)

    best_val_auc = stage1_metrics["macro_auc"]
    best_epoch = epochs_stage1
    best_model_path = MODELS_DIR / "best_model.pth"
    torch.save(model.state_dict(), best_model_path)

    t0_stage2 = time.time()
    for epoch in range(1, epochs_stage2 + 1):
        total_epoch = epochs_stage1 + epoch
        tr_loss, _, _, _ = run_epoch(model, train_loader, criterion, optimizer_stage2, is_training=True)
        va_loss, va_metrics, val_probs, val_targets = run_epoch(model, val_loader, criterion, is_training=False)

        history["epoch"].append(total_epoch)
        history["stage"].append("Stage 2 (Fine-Tuned)")
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_macro_f1"].append(va_metrics["macro_f1"])
        history["val_micro_f1"].append(va_metrics["micro_f1"])
        history["val_macro_auc"].append(va_metrics["macro_auc"])

        print(
            f"Stage 2 | Epoch {epoch}/{epochs_stage2} (Tot: {total_epoch}) | "
            f"Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | "
            f"Val Macro F1: {va_metrics['macro_f1']:.4f} | Val ROC-AUC: {va_metrics['macro_auc']:.4f}"
        )

        if va_metrics["macro_auc"] >= best_val_auc:
            best_val_auc = va_metrics["macro_auc"]
            best_epoch = total_epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Best model checkpointed at epoch {total_epoch} (Val ROC-AUC: {best_val_auc:.4f})")

    t_stage2 = time.time() - t0_stage2

    # Load best model for evaluation and threshold tuning
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    _, stage2_metrics, best_val_probs, best_val_targets = run_epoch(model, val_loader, criterion, is_training=False)
    print(f"\nBest model loaded from {best_model_path} (from epoch {best_epoch})")

    # =========================================================================
    # THRESHOLD TUNING (ON VALIDATION SET ONLY)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  OPTIMIZING DECISION THRESHOLDS (ON VALIDATION SET ONLY)")
    print("=" * 60)
    optimal_thresholds, threshold_metrics = find_optimal_thresholds(
        y_true=best_val_targets,
        y_probs=best_val_probs,
        class_names=selected_labels,
    )
    thresholds_file = MODELS_DIR / "thresholds.json"
    save_thresholds(optimal_thresholds, thresholds_file)
    print(f"Tuned thresholds saved to {thresholds_file}:")
    for name in selected_labels:
        tm = threshold_metrics[name]
        print(f"  {name:<18} -> Threshold: {tm['threshold']:.2f} | Val F1: {tm['f1']:.4f} (Prec: {tm['precision']:.3f}, Rec: {tm['recall']:.3f})")

    # Compute validation metrics with tuned thresholds
    tuned_val_preds = apply_thresholds(best_val_probs, optimal_thresholds, selected_labels)
    tuned_val_macro_f1 = float(f1_score(best_val_targets, tuned_val_preds, average="macro", zero_division=0))
    tuned_val_micro_f1 = float(f1_score(best_val_targets, tuned_val_preds, average="micro", zero_division=0))

    # =========================================================================
    # EXPERIMENT: FROZEN VS FINE-TUNED COMPARISON
    # =========================================================================
    print("\n" + "=" * 60)
    print("  EXPERIMENT A VS EXPERIMENT B COMPARISON")
    print("=" * 60)
    exp_data = [
        {
            "experiment": "Experiment A (Frozen Backbone)",
            "macro_f1": round(stage1_metrics["macro_f1"], 4),
            "micro_f1": round(stage1_metrics["micro_f1"], 4),
            "macro_roc_auc": round(stage1_metrics["macro_auc"], 4),
            "training_time_seconds": round(t_stage1, 2),
        },
        {
            "experiment": "Experiment B (Fine-Tuned Top Layers)",
            "macro_f1": round(stage2_metrics["macro_f1"], 4),
            "micro_f1": round(stage2_metrics["micro_f1"], 4),
            "macro_roc_auc": round(stage2_metrics["macro_auc"], 4),
            "training_time_seconds": round(t_stage2, 2),
        },
    ]
    exp_df = pd.DataFrame(exp_data)
    exp_csv = EXPERIMENTS_DIR / "frozen_vs_finetuned.csv"
    exp_df.to_csv(exp_csv, index=False)
    print(exp_df.to_string(index=False))
    print(f"\nExperiment comparison saved to {exp_csv}")

    # Plot experiment comparison
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    metrics_to_plot = ["macro_f1", "micro_f1", "macro_roc_auc"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    ax[0].bar(x - width/2, [exp_data[0][m] for m in metrics_to_plot], width, label="Frozen Backbone (A)", color="#4575b4")
    ax[0].bar(x + width/2, [exp_data[1][m] for m in metrics_to_plot], width, label="Fine-Tuned Layers (B)", color="#d73027")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(["Macro F1", "Micro F1", "Macro ROC-AUC"], fontweight="bold")
    ax[0].set_ylim(0, 1.0)
    ax[0].set_title("Model Performance Comparison", fontweight="bold")
    ax[0].legend()
    ax[0].grid(axis="y", linestyle="--", alpha=0.6)

    # Training time
    ax[1].bar(["Frozen (Stage 1)", "Fine-Tuned (Stage 2)"], [t_stage1, t_stage2], color=["#4575b4", "#d73027"], width=0.5)
    ax[1].set_ylabel("Training Time (seconds)", fontweight="bold")
    ax[1].set_title("Computational Efficiency", fontweight="bold")
    ax[1].grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    exp_fig_path = FIGURES_DIR / "frozen_vs_finetuned.png"
    plt.savefig(exp_fig_path, dpi=300)
    plt.close()
    print(f"Experiment plot saved to {exp_fig_path}")

    # Plot training curves
    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    plt.plot(history["epoch"], history["val_loss"], "s-", label="Val Loss", color="#ff7f0e")
    plt.axvline(x=epochs_stage1 + 0.5, color="gray", linestyle="--", label="Stage 1 -> Stage 2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Weighted BCE)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["val_macro_f1"], "o-", label="Val Macro F1", color="#2ca02c")
    plt.plot(history["epoch"], history["val_macro_auc"], "d-", label="Val Macro ROC-AUC", color="#9467bd")
    plt.axvline(x=epochs_stage1 + 0.5, color="gray", linestyle="--", label="Stage 1 -> Stage 2")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics Across Stages")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    curves_fig_path = FIGURES_DIR / "training_curves.png"
    plt.savefig(curves_fig_path, dpi=300)
    plt.close()
    print(f"Training curves saved to {curves_fig_path}")

    print("\n" + "=" * 60)
    print("  TRAINING PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    train_pipeline()
