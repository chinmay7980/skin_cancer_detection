"""Multimodal Fusion Architecture combining EfficientNet-B0 and Tabular Metadata."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class MultimodalFusionNet(nn.Module):
    """Multimodal neural network for skin lesion risk classification.

    Combines:
    1. Image feature representation (1,280 dimensions) extracted from EfficientNet-B0.
    2. Leakage-free clinical & 3D geometric tabular metadata (e.g., 50 dimensions).
    3. Multimodal fusion head: Concatenation -> Linear -> BatchNorm -> ReLU -> Dropout -> Linear(1).
    """

    def __init__(
        self,
        metadata_dim: int = 50,
        fusion_hidden_dim: int = 128,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.metadata_dim = metadata_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.dropout_rate = dropout_rate

        # 1. Image Branch: EfficientNet-B0 feature extractor
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        cnn = efficientnet_b0(weights=weights)
        self.features = cnn.features
        self.avgpool = cnn.avgpool
        self.image_feature_dim = 1280

        # Freeze image backbone by default to isolate fusion head learning
        if freeze_backbone:
            self.freeze_backbone()

        # 2. Multimodal Fusion Head
        combined_dim = self.image_feature_dim + self.metadata_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def freeze_backbone(self) -> None:
        """Freeze all weights in the CNN feature extractor."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_top_blocks(self, num_blocks: int = 2) -> None:
        """Unfreeze the topmost N convolutional blocks for controlled fine-tuning."""
        total_blocks = len(self.features)
        unfreeze_start = max(0, total_blocks - num_blocks)
        for i in range(unfreeze_start, total_blocks):
            for param in self.features[i].parameters():
                param.requires_grad = True

    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract 1,280-dimensional image embeddings from raw image tensors."""
        x = self.features(images)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            metadata: Preprocessed tabular tensor of shape (batch_size, metadata_dim)

        Returns:
            Single scalar logit per sample with shape (batch_size,).
        """
        img_feats = self.extract_image_features(images)
        combined = torch.cat([img_feats, metadata], dim=1)
        logits = self.fusion_head(combined)
        return logits.squeeze(-1)


def get_torch_device() -> torch.device:
    """Detect and return preferred torch device (mps -> cuda -> cpu)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
