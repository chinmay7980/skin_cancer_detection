"""EfficientNet-B0 Transfer Learning Architecture for Skin Lesion Malignancy."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0Binary(nn.Module):
    """EfficientNet-B0 customized for binary lesion risk classification.

    Replaces the 1000-class ImageNet head with a single linear projection
    producing a single logit. Uses BCEWithLogitsLoss during training.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features

        # Replace 1000-class head with single-logit binary head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 1),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all parameters in the convolutional feature extractor."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_top_blocks(self, num_blocks: int = 2) -> None:
        """Unfreeze the top N blocks of the feature extractor for fine-tuning."""
        total_blocks = len(self.backbone.features)
        unfreeze_start = max(0, total_blocks - num_blocks)
        for i in range(unfreeze_start, total_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning single scalar logit per sample.

        Output shape: (batch_size,)
        """
        logits = self.backbone(x)
        return logits.squeeze(-1)


def get_torch_device() -> torch.device:
    """Detect and return preferred torch device (mps -> cuda -> cpu)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
