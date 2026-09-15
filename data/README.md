# Data Directory Structure

This directory contains the dataset structure for the Multi-Label Chest X-Ray Abnormality Detection project:

- `data/chestxray14/`: Expected location for the NIH ChestX-ray14 raw files (`Data_Entry_2017.csv` and `images/`).
- `data/processed/`: Contains processed metadata (`labels.csv`) with patient-stratified train/val/test splits.
- `data/selected_labels.json`: The 8 data-driven thoracic findings selected for training.

To automatically set up and download the dataset, simply run:
```bash
python src/setup_dataset.py
```
This will automatically verify any existing files or download and prepare the dataset without manual intervention.
