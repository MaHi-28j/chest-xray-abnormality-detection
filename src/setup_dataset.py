"""Dataset setup, download, verification, label analysis, and preprocessing pipeline.

Handles:
1. Existence checks and integrity verification.
2. Official source tracking, resumable download from public distribution.
3. Structure normalization into data/chestxray14/.
4. Metadata verification and dataset manifest generation.
5. Label distribution analysis and visualization.
6. Data-driven abnormality selection (6-8 classes).
7. Patient-level stratified subset creation (avoiding data leakage).
"""

import os
import sys
import json
import time
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Reproducibility seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHESTXRAY_DIR = DATA_DIR / "chestxray14"
IMAGES_DIR = CHESTXRAY_DIR / "images"
METADATA_CSV = CHESTXRAY_DIR / "Data_Entry_2017.csv"
MANIFEST_PATH = CHESTXRAY_DIR / "dataset_manifest.json"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_LABELS_CSV = PROCESSED_DIR / "labels.csv"
SELECTED_LABELS_JSON = DATA_DIR / "selected_labels.json"

RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

# Official NIH ChestX-ray14 References
OFFICIAL_SOURCE_INFO = {
    "dataset_name": "NIH ChestX-ray14",
    "official_portal": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
    "paper_citation": (
        "Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, "
        "Ronald M. Summers. ChestX-ray8: Hospital-scale Chest X-ray Database and "
        "Benchmarks on Weakly-Supervised Classification and Localization of Common "
        "Thorax Diseases, IEEE CVPR 2017."
    ),
    "metadata_url": (
        "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/Data_Entry_2017_v2020.csv"
    ),
    "sample_images_archive_url": (
        "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/images/images_001.zip"
    ),
}

ALL_14_FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]


def print_header(title: str):
    print("\n" + "=" * 65)
    print(f"  {title.upper()}")
    print("=" * 65)


def download_file_resumable(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
    """Download a file with HTTP Range resume support and tqdm progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    headers = {"User-Agent": "Mozilla/5.0"}
    existing_bytes = temp_path.stat().st_size if temp_path.exists() else 0

    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        print(f"Resuming download from byte {existing_bytes} for {dest_path.name}...")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, headers=headers, timeout=30) as r:
                if r.status_code in [200, 206]:
                    total_size = int(r.headers.get("content-length", 0)) + existing_bytes
                    mode = "ab" if (existing_bytes > 0 and r.status_code == 206) else "wb"
                    if mode == "wb":
                        existing_bytes = 0

                    with open(temp_path, mode) as f, tqdm(
                        total=total_size,
                        initial=existing_bytes,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=desc,
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=1024 * 64):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                    # Rename .part to final destination
                    if temp_path.exists():
                        if dest_path.exists():
                            dest_path.unlink()
                        temp_path.rename(dest_path)
                    return True
                else:
                    print(f"Download attempt {attempt} failed with HTTP {r.status_code}")
        except Exception as e:
            print(f"Download attempt {attempt} encountered error: {e}")
            time.sleep(2)

    print(f"Failed to download {url} after {max_retries} attempts.")
    return False


def setup_raw_dataset():
    """Ensure NIH ChestX-ray14 metadata and images are downloaded and extracted."""
    print_header("Step 1: Dataset Acquisition & Check")
    CHESTXRAY_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Check Metadata
    if not METADATA_CSV.exists():
        print("Metadata CSV not found locally. Initiating download...")
        print(f"Official Repository: {OFFICIAL_SOURCE_INFO['official_portal']}")
        success = download_file_resumable(
            OFFICIAL_SOURCE_INFO["metadata_url"],
            METADATA_CSV,
            desc="Data_Entry_2017.csv"
        )
        if not success:
            raise RuntimeError(
                f"Could not download metadata CSV. Please manually place Data_Entry_2017.csv "
                f"into {CHESTXRAY_DIR} from {OFFICIAL_SOURCE_INFO['official_portal']}."
            )
        print("Metadata CSV successfully downloaded.")
    else:
        print(f"Metadata CSV already exists: {METADATA_CSV}")

    # 2. Check Images
    existing_images = list(IMAGES_DIR.glob("*.png"))
    if len(existing_images) >= 500:
        print(f"Images already present: {len(existing_images)} PNG images found in {IMAGES_DIR}.")
    else:
        # Check if archive already downloaded in data/ or chestxray14/
        archive_candidates = list(CHESTXRAY_DIR.glob("*.zip")) + list(DATA_DIR.glob("*.zip"))
        archive_path = None
        for cand in archive_candidates:
            if "images" in cand.name.lower():
                archive_path = cand
                break

        if archive_path is None or not archive_path.exists():
            archive_path = CHESTXRAY_DIR / "images_001.zip"
            print("Image archive not found locally. Initiating automatic download of NIH images_001 batch (~1.9 GB)...")
            print(f"Source: NIH ChestX-ray14 public distribution archive.")
            success = download_file_resumable(
                OFFICIAL_SOURCE_INFO["sample_images_archive_url"],
                archive_path,
                desc="images_001.zip (~1.9 GB)"
            )
            if not success:
                raise RuntimeError(
                    f"Automatic image download interrupted. You may download images_001.zip "
                    f"manually from {OFFICIAL_SOURCE_INFO['official_portal']} and place it in {CHESTXRAY_DIR}."
                )

        print(f"Extracting images from {archive_path.name} to {IMAGES_DIR}...")
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = [m for m in zf.namelist() if m.endswith(".png") and not m.startswith("__MACOSX")]
            for member in tqdm(members, desc="Extracting PNGs"):
                filename = os.path.basename(member)
                target = IMAGES_DIR / filename
                if not target.exists():
                    source = zf.open(member)
                    with open(target, "wb") as target_file:
                        target_file.write(source.read())

        extracted_count = len(list(IMAGES_DIR.glob("*.png")))
        print(f"Extraction complete. Total images available: {extracted_count}")


def verify_dataset_integrity():
    """Perform comprehensive verification of metadata, image files, and labels."""
    print_header("Step 2: Dataset Verification & Integrity Checks")

    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata file missing at {METADATA_CSV}")
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images directory missing at {IMAGES_DIR}")

    df_meta = pd.read_csv(METADATA_CSV)
    required_cols = ["Image Index", "Finding Labels", "Patient ID"]
    for col in required_cols:
        if col not in df_meta.columns:
            raise ValueError(f"Required column '{col}' missing from metadata CSV!")

    # Check available local images
    local_images = set(p.name for p in IMAGES_DIR.glob("*.png"))
    total_local_images = len(local_images)

    # Filter metadata for rows corresponding to available images
    df_available = df_meta[df_meta["Image Index"].isin(local_images)].copy()
    images_referenced = len(df_available)

    # Verify PIL can open a sample of available images (up to 500 images checked for corruption)
    sample_to_check = list(local_images)[:min(500, len(local_images))]
    corrupted_count = 0
    valid_dims = []

    for img_name in sample_to_check:
        img_path = IMAGES_DIR / img_name
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                valid_dims.append(img.size)
        except Exception:
            corrupted_count += 1

    # Check labels present in dataset
    all_findings_in_data = set()
    for label_str in df_meta["Finding Labels"].dropna():
        for l in label_str.split("|"):
            l = l.strip()
            if l and l != "No Finding":
                all_findings_in_data.add(l)

    status = "READY" if (corrupted_count == 0 and total_local_images > 0) else "ERROR"

    print("\n## Dataset verification report")
    print(f"Metadata file:     OK ({METADATA_CSV.name})")
    print(f"Image directory:   OK ({IMAGES_DIR})")
    print(f"Metadata rows:     {len(df_meta):,}")
    print(f"Images found:      {total_local_images:,}")
    print(f"Images checked:    {len(sample_to_check):,}")
    print(f"Corrupted images:  {corrupted_count}")
    print(f"Labels found:      {len(all_findings_in_data)} (ChestX-ray14 standards)")
    print(f"Sample dimensions: {set(valid_dims)}")
    print(f"Dataset status:    {status}\n")

    if status != "READY":
        raise RuntimeError("Dataset verification failed: Corrupted images or missing data detected.")

    # Save manifest
    manifest = {
        "dataset_name": "NIH ChestX-ray14",
        "dataset_location": str(CHESTXRAY_DIR),
        "metadata_rows": int(len(df_meta)),
        "available_local_images": int(total_local_images),
        "corrupted_images_detected": int(corrupted_count),
        "all_14_labels_present": sorted(list(all_findings_in_data)),
        "creation_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {MANIFEST_PATH}")
    return df_available


def analyze_labels(df_available: pd.DataFrame):
    """Analyze label distribution across available images and generate plots and metrics."""
    print_header("Step 3: Label Analysis")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    total_images = len(df_available)
    finding_counts = {finding: 0 for finding in ALL_14_FINDINGS}
    finding_counts["No Finding"] = 0
    multi_label_count = 0

    for labels_str in df_available["Finding Labels"]:
        tags = [t.strip() for t in labels_str.split("|")]
        if len(tags) > 1:
            multi_label_count += 1
        for t in tags:
            if t in finding_counts:
                finding_counts[t] += 1

    dist_rows = []
    for finding in ALL_14_FINDINGS + ["No Finding"]:
        count = finding_counts.get(finding, 0)
        pct = (count / total_images) * 100 if total_images > 0 else 0.0
        dist_rows.append({
            "finding": finding,
            "positive_count": count,
            "percentage": round(pct, 2)
        })

    df_dist = pd.DataFrame(dist_rows).sort_values(by="positive_count", ascending=False)
    csv_path = METRICS_DIR / "label_distribution.csv"
    df_dist.to_csv(csv_path, index=False)
    print(f"Label distribution saved to {csv_path}")

    print("\nActual Label Distribution in Downloaded Pool:")
    print("-" * 45)
    for _, row in df_dist.iterrows():
        print(f"  {row['finding']:<20}: {int(row['positive_count']):>5} ({row['percentage']:>5.2f}%)")
    print("-" * 45)
    print(f"Total available images:               {total_images:,}")
    print(f"Images with multiple abnormalities:   {multi_label_count:,} ({(multi_label_count/total_images)*100:.1f}%)")

    # Generate visual figure
    plt.figure(figsize=(10, 6))
    plot_df = df_dist[df_dist["finding"] != "No Finding"].sort_values(by="positive_count")
    bars = plt.barh(plot_df["finding"], plot_df["positive_count"], color="#2c7bb6", edgecolor="#1a476f")
    plt.xlabel("Number of Positive Radiographs", fontsize=11, fontweight="bold")
    plt.title(f"NIH ChestX-ray14 Thoracic Abnormality Distribution (N={total_images:,})", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    for bar in bars:
        w = bar.get_width()
        plt.text(w + max(plot_df["positive_count"])*0.01, bar.get_y() + bar.get_height()/2, f"{int(w):,}",
                 va="center", ha="left", fontsize=9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "label_distribution.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Label distribution plot saved to {fig_path}")

    return df_dist


def select_abnormalities(df_dist: pd.DataFrame, min_positives: int = 100):
    """Select 6-8 abnormalities based on minimum positive count criterion."""
    print_header("Step 4: Data-Driven Abnormality Selection")

    # Filter candidates (excluding 'No Finding')
    candidates = df_dist[(df_dist["finding"] != "No Finding") & (df_dist["positive_count"] >= min_positives)]
    selected_findings = candidates["finding"].tolist()

    # If more than 8, pick top 8 most prevalent for high statistical quality
    if len(selected_findings) > 8:
        selected_findings = selected_findings[:8]
    elif len(selected_findings) < 6:
        # Fallback to top 6
        selected_findings = df_dist[df_dist["finding"] != "No Finding"]["finding"].tolist()[:6]

    selected_findings.sort()

    print(f"Documented Selection Criterion: Thoracic findings with >= {min_positives} positive samples.")
    print("Selected Abnormalities:")
    for i, finding in enumerate(selected_findings, 1):
        count = int(df_dist[df_dist["finding"] == finding]["positive_count"].values[0])
        print(f"  {i}. {finding:<18} (Positive count: {count})")

    with open(SELECTED_LABELS_JSON, "w") as f:
        json.dump(selected_findings, f, indent=2)
    print(f"\nSelection saved to {SELECTED_LABELS_JSON}")
    return selected_findings


def create_manageable_splits(df_available: pd.DataFrame, selected_labels: list, max_samples: int = 1500):
    """Create manageable patient-stratified subset avoiding data leakage.

    Splits:
    - Train: 70%
    - Val:   15%
    - Test:  15%

    Crucially: All images of any patient strictly belong to ONE split.
    """
    print_header("Step 5: Patient-Stratified Split & Subset Creation")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Parse multi-label binary indicators
    for label in selected_labels:
        df_available[label] = df_available["Finding Labels"].apply(
            lambda s: 1 if label in [t.strip() for t in str(s).split("|")] else 0
        )

    # Indicator for whether an image has ANY of the selected abnormalities
    df_available["has_selected_finding"] = df_available[selected_labels].sum(axis=1) > 0
    df_available["is_no_finding"] = df_available["Finding Labels"].apply(
        lambda s: 1 if "No Finding" in [t.strip() for t in str(s).split("|")] else 0
    )

    # Keep positive samples and healthy negatives (No Finding)
    relevant_df = df_available[df_available["has_selected_finding"] | (df_available["is_no_finding"] == 1)].copy()

    # Sample manageable subset if pool is larger than max_samples
    if len(relevant_df) > max_samples:
        pos_df = relevant_df[relevant_df["has_selected_finding"]]
        neg_df = relevant_df[~relevant_df["has_selected_finding"]]

        # Balance positive and negative cases
        n_pos = min(len(pos_df), int(max_samples * 0.75))
        n_neg = min(len(neg_df), max_samples - n_pos)

        sample_pos = pos_df.sample(n=n_pos, random_state=RANDOM_SEED)
        sample_neg = neg_df.sample(n=n_neg, random_state=RANDOM_SEED)
        subset_df = pd.concat([sample_pos, sample_neg]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        subset_df = relevant_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    # 2. Patient-Level Splitting
    patient_ids = subset_df["Patient ID"].unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(patient_ids)

    n_patients = len(patient_ids)
    n_train = int(n_patients * 0.70)
    n_val = int(n_patients * 0.15)

    train_patients = set(patient_ids[:n_train])
    val_patients = set(patient_ids[n_train:n_train + n_val])
    test_patients = set(patient_ids[n_train + n_val:])

    # Verify zero patient leakage
    assert len(train_patients.intersection(val_patients)) == 0, "Leakage detected between Train and Val!"
    assert len(train_patients.intersection(test_patients)) == 0, "Leakage detected between Train and Test!"
    assert len(val_patients.intersection(test_patients)) == 0, "Leakage detected between Val and Test!"

    def assign_split(pid):
        if pid in train_patients:
            return "train"
        elif pid in val_patients:
            return "val"
        else:
            return "test"

    subset_df["split"] = subset_df["Patient ID"].apply(assign_split)

    # Build relative image path
    subset_df["image_path"] = subset_df["Image Index"].apply(
        lambda fname: str(Path("chestxray14") / "images" / fname)
    )

    cols_to_save = ["image_path", "split", "Patient ID", "Finding Labels"] + selected_labels
    final_df = subset_df[cols_to_save]
    final_df.to_csv(PROCESSED_LABELS_CSV, index=False)

    print(f"Processed labels saved to {PROCESSED_LABELS_CSV}")
    print(f"Total images in subset:     {len(final_df):,}")
    print(f"Unique patients:            {n_patients:,}")
    print("\nSplit Breakdown:")
    for split_name in ["train", "val", "test"]:
        sub = final_df[final_df["split"] == split_name]
        n_pts = sub["Patient ID"].nunique()
        print(f"  {split_name.upper():<6}: {len(sub):>4} images from {n_pts:>4} unique patients")

    print("\nPositive Class Distribution Across Splits:")
    for label in selected_labels:
        tr_c = int(final_df[final_df["split"] == "train"][label].sum())
        va_c = int(final_df[final_df["split"] == "val"][label].sum())
        te_c = int(final_df[final_df["split"] == "test"][label].sum())
        print(f"  {label:<18} -> Train: {tr_c:>3} | Val: {va_c:>3} | Test: {te_c:>3}")

    print("\nZero Data Leakage Confirmed: No patient appears in more than one split.")
    return final_df


def main():
    print_header("ChestX-ray14 Automated Dataset Setup")
    print(f"Base Directory: {BASE_DIR}")

    # Step 1: Download / verify presence
    setup_raw_dataset()

    # Step 2: Integrity check & manifest
    df_available = verify_dataset_integrity()

    # Step 3: Label distribution analysis
    df_dist = analyze_labels(df_available)

    # Step 4: Class selection
    selected_labels = select_abnormalities(df_dist, min_positives=100)

    # Step 5: Patient-stratified split
    create_manageable_splits(df_available, selected_labels, max_samples=1500)

    print_header("Dataset Setup Complete & Verified")
    print("Ready for model training: python src/train.py")


if __name__ == "__main__":
    main()
