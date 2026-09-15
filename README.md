# Multi-Label Chest X-Ray Abnormality Detection with Explainable AI

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multi-label thoracic pathology classifier built on the **NIH ChestX-ray14** dataset. Instead of forcing one diagnosis per image, it predicts the independent probability of **8 co-occurring abnormalities** and explains each prediction with **target-specific Grad-CAM** heatmaps.

```
Chest X-ray (224×224×3) → ResNet-18 → GAP(512) → Linear(512→8) → Sigmoid
                                                        │
                                          per-class probability + Grad-CAM
```

## Dataset

- **Source:** [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) (Wang et al., CVPR 2017), 112,120 frontal X-rays, 30,805 patients.
- **This run:** first 4,999 images; top 8 abnormalities with ≥100 positives selected — **Infiltration, Effusion, Atelectasis, Nodule, Consolidation, Pneumothorax, Cardiomegaly, Fibrosis**.
- **Splits:** strictly by Patient ID (no leakage) — 1,003 train / 233 val / 264 test images.

## Model & Training

- **Backbone:** ImageNet-pretrained ResNet-18, head replaced with `Linear(512, 8)`.
- **Loss:** `BCEWithLogitsLoss` with per-class positive weights computed from the training split (2×–14× depending on rarity).
- **Two-stage transfer learning:**
  1. Frozen backbone, train head only (4 epochs)
  2. Unfreeze `layer4`, fine-tune at lower LR (4 epochs)
- **Thresholds:** tuned per class on the validation set to maximize F1 (default 0.5 fails on rare classes).

| Experiment | Trainable Params | Val Macro ROC-AUC | Val Macro F1 |
|---|---:|---:|---:|
| Frozen backbone | 4,104 | 0.623 | 0.225 |
| Fine-tuned `layer4` | 8,397,832 | **0.697** | **0.324** |

## Test Set Results (264 images, evaluated once)

| Metric | Baseline (0.50) | Tuned Thresholds |
|---|---:|---:|
| Macro ROC-AUC | 0.682 | 0.682 |
| Micro F1 | 0.332 | **0.347** |
| Macro Recall | 0.518 | 0.368 |

Best per-class ROC-AUC: **Fibrosis (0.80)**, **Atelectasis (0.79)**, **Effusion (0.79)**. Full per-class table in `results/metrics/final_metrics.csv`.

## Explainability: Grad-CAM

Each prediction can be explained by backpropagating the target class's logit through `layer4` to produce a heatmap over the anatomically relevant region (e.g. cardiac silhouette for Cardiomegaly, costophrenic angles for Effusion).

## Interactive App

```bash
python -m streamlit run app/app.py
```
Upload or select an X-ray, view per-class probabilities against tuned thresholds, and inspect Grad-CAM overlays for any predicted abnormality.

## Quickstart

```bash
git clone https://github.com/yourusername/chest-xray-abnormality-detection.git
cd chest-xray-abnormality-detection
pip install -r requirements.txt

python src/setup_dataset.py   # download, verify, split data
python src/train.py           # two-stage training + threshold tuning
python src/evaluate.py        # test-set metrics + ROC/PR curves
python src/gradcam.py         # generate Grad-CAM figures
python -m pytest tests/ -v    # run test suite (12 tests)
python -m streamlit run app/app.py   # launch dashboard
```

## Tests

12 unit/integration tests cover patient-leakage checks, dataset tensor shapes, model freeze/unfreeze, threshold search bounds, and Grad-CAM class-specificity (`tests/`, run via `pytest`).

## Limitations

- Labels are NLP-mined from radiology reports (~90% accuracy) — noisy weak supervision.
- Grad-CAM shows where the model attended, not ground-truth pathology boundaries; coarse 7×7 resolution.
- Trained on a 4,999-image subset, not the full 112k-image dataset.

## License

MIT
