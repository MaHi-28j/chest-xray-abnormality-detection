"""ResNet-18 Multi-Label Thoracic Abnormality Classification Model."""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ChestXrayResNet18(nn.Module):
    """ResNet-18 adapted for multi-label thoracic abnormality classification.

    Architecture:
        Input Radiograph (3x224x224)
            ↓
        Pretrained ResNet-18 Backbone
            ↓
        Global Average Pooling (512-dim)
            ↓
        Linear Classification Head (512 -> num_classes)
            ↓
        Raw Logits (for BCEWithLogitsLoss)
            ↓ [Inference]
        Sigmoid -> Independent Probabilities per Class
    """

    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # Replace final classification head
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw unnormalized logits for numerical stability with BCEWithLogitsLoss."""
        return self.backbone(x)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Sigmoid activation to obtain independent multi-label probabilities in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def freeze_backbone(self):
        """Freeze all layers except the final classification head (Stage 1)."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def unfreeze_final_blocks(self, layers_to_unfreeze: list = None):
        """Unfreeze final residual block(s) and classification head for fine-tuning (Stage 2)."""
        if layers_to_unfreeze is None:
            layers_to_unfreeze = ["layer4", "fc"]

        for name, param in self.backbone.named_parameters():
            if any(layer_name in name for layer_name in layers_to_unfreeze):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(num_classes: int = 8, pretrained: bool = True) -> ChestXrayResNet18:
    """Factory helper to build and return the ResNet-18 model."""
    return ChestXrayResNet18(num_classes=num_classes, pretrained=pretrained)
