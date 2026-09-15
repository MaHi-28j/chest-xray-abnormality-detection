"""PyTorch Dataset, transforms, and DataLoader utilities for NIH ChestX-ray14."""

import json
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str = "train", image_size: int = 224) -> transforms.Compose:
    """Return data transforms based on the split.

    Training split includes subtle, medically plausible augmentations (horizontal flip,
    mild rotation, slight brightness/contrast variation). Validation and test sets use
    strictly deterministic resizing and normalization.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class ChestXrayDataset(Dataset):
    """Multi-label PyTorch Dataset for chest radiographs."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: Path,
        label_columns: List[str],
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            df: DataFrame containing at least 'image_path' and class label columns.
            data_root: Base data directory where 'chestxray14/images/...' resides.
            label_columns: List of abnormality column names.
            transform: Optional torchvision transforms.
        """
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.label_columns = label_columns
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        rel_path = row["image_path"]
        img_full_path = self.data_root / rel_path

        # Handle fallback if path stored is direct filename or relative path
        if not img_full_path.exists():
            fname = Path(rel_path).name
            img_full_path = self.data_root / "chestxray14" / "images" / fname

        if not img_full_path.exists():
            raise FileNotFoundError(f"Image not found at {img_full_path}")

        # Open image and convert to RGB (standard 3 channels for ResNet)
        image = Image.open(img_full_path).convert("RGB")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        # Multi-label binary target vector
        targets = row[self.label_columns].values.astype("float32")
        target_tensor = torch.tensor(targets, dtype=torch.float32)

        return image_tensor, target_tensor, str(img_full_path)


def load_selected_labels(labels_path: Optional[Path] = None) -> List[str]:
    """Load the list of selected abnormality names."""
    if labels_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        labels_path = base_dir / "data" / "selected_labels.json"

    with open(labels_path, "r") as f:
        labels = json.load(f)
    return labels


def create_dataloaders(
    csv_path: Path,
    data_root: Path,
    label_columns: List[str],
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test PyTorch DataLoaders."""
    df_all = pd.read_csv(csv_path)

    train_df = df_all[df_all["split"] == "train"]
    val_df = df_all[df_all["split"] == "val"]
    test_df = df_all[df_all["split"] == "test"]

    train_dataset = ChestXrayDataset(
        df=train_df,
        data_root=data_root,
        label_columns=label_columns,
        transform=get_transforms("train", image_size=image_size),
    )
    val_dataset = ChestXrayDataset(
        df=val_df,
        data_root=data_root,
        label_columns=label_columns,
        transform=get_transforms("val", image_size=image_size),
    )
    test_dataset = ChestXrayDataset(
        df=test_df,
        data_root=data_root,
        label_columns=label_columns,
        transform=get_transforms("test", image_size=image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader
