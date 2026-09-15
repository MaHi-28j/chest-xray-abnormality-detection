"""Inference pipeline for single chest radiograph abnormality prediction."""

import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from dataset import IMAGENET_MEAN, IMAGENET_STD, load_selected_labels
from model import build_model
from thresholds import load_thresholds

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"
SELECTED_LABELS_PATH = DATA_DIR / "selected_labels.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor:
    """Chest X-ray multi-label abnormality predictor."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        thresholds_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or DEVICE
        self.labels = load_selected_labels(labels_path or SELECTED_LABELS_PATH)
        self.thresholds = load_thresholds(thresholds_path or THRESHOLDS_PATH)

        self.model = build_model(num_classes=len(self.labels), pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(model_path or BEST_MODEL_PATH, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def predict(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Dict]:
        """Predict probabilities and binary detections for a chest X-ray.

        Args:
            image_input: Filepath to radiograph or PIL Image instance.

        Returns:
            Dictionary mapping abnormality -> {
                'probability': float,
                'threshold': float,
                'detected': bool,
                'percentage': str
            }
        """
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                p = DATA_DIR / image_input
            if not p.exists():
                p = DATA_DIR / "chestxray14" / "images" / Path(image_input).name
            if not p.exists():
                raise FileNotFoundError(f"Image not found at {image_input}")
            pil_image = Image.open(p).convert("RGB")
        else:
            pil_image = image_input.convert("RGB")

        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        results = {}
        for idx, name in enumerate(self.labels):
            p = float(probs[idx])
            t = float(self.thresholds.get(name, 0.5))
            detected = bool(p >= t)
            results[name] = {
                "probability": round(p, 4),
                "threshold": round(t, 3),
                "detected": detected,
                "percentage": f"{p * 100:.1f}%",
            }

        return results


def predict_single_image(image_path: Union[str, Path]) -> Dict[str, Dict]:
    """Helper function to run inference on a single image file."""
    predictor = Predictor()
    return predictor.predict(image_path)
