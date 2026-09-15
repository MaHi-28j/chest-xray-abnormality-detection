# Models Directory

This directory stores model configuration files and trained weights:
- `pos_weights.json`: Positive loss weights calculated from the training split.
- `thresholds.json`: Class-specific optimal decision thresholds tuned on the validation set.
- `best_model.pth`: Checkpoint of the fine-tuned ResNet-18 model (generated during training).
- `frozen_model.pth`: Checkpoint of the Stage 1 frozen backbone model (generated during training).

To train the model and generate the `.pth` checkpoints:
```bash
python src/train.py
```
