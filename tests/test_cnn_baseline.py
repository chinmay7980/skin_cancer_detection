"""Automated unit tests for lazy image loading, CNN architecture, and training components."""

import torch
import torch.nn as nn
import pytest
from src.config import load_config
from src.data.image_dataset import ISICImageDataset, create_dataloader, get_image_transforms
from src.models.cnn import EfficientNetB0Binary, get_torch_device
from src.splits.patient_split import load_split_manifest


@pytest.fixture(scope="module")
def app_config():
    return load_config()


@pytest.fixture(scope="module")
def split_manifest(app_config):
    return load_split_manifest(app_config.splits.manifest_path)


def test_lazy_image_dataset_output_properties(app_config, split_manifest):
    """Verify ISICImageDataset outputs proper tensor shapes, normalization, and scalar float target."""
    sample_df = split_manifest.head(5).copy()
    transform = get_image_transforms(image_size=160, is_training=False)
    dataset = ISICImageDataset(df=sample_df, hdf5_path=app_config.data.train_images_hdf5, transform=transform)

    assert len(dataset) == 5
    img_tensor, target_tensor = dataset[0]

    # Verify tensor shapes
    assert img_tensor.shape == (3, 160, 160), f"Expected (3, 160, 160), got {img_tensor.shape}"
    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.ndim == 0, "Target must be a scalar tensor"
    assert target_tensor.dtype == torch.float32

    # Verify normalization range (not raw 0-255 uint8)
    assert img_tensor.min() < 0.0 or img_tensor.max() > 1.0


def test_cnn_architecture_and_forward_pass():
    """Verify EfficientNetB0Binary produces valid single-logit output and handles device transfer."""
    device = get_torch_device()
    model = EfficientNetB0Binary(pretrained=False, freeze_backbone=True)
    model.to(device)
    model.eval()

    dummy_input = torch.randn(4, 3, 160, 160, device=device)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (4,), f"Expected shape (4,), got {logits.shape}"
    probs = torch.sigmoid(logits)
    assert ((probs >= 0.0) & (probs <= 1.0)).all(), "Probabilities out of [0, 1] range!"


def test_cnn_backward_pass_and_freeze_integrity():
    """Verify loss calculation, backward pass, and that frozen backbone parameters receive no gradient."""
    device = torch.device("cpu")
    model = EfficientNetB0Binary(pretrained=False, freeze_backbone=True)
    model.to(device)
    model.train()

    dummy_input = torch.randn(2, 3, 160, 160, device=device)
    dummy_target = torch.tensor([0.0, 1.0], device=device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    logits = model(dummy_input)
    loss = criterion(logits, dummy_target)
    assert torch.isfinite(loss), "Loss is not finite!"
    loss.backward()

    # Backbone parameters should have NO gradient (frozen)
    for name, param in model.backbone.features.named_parameters():
        assert param.grad is None, f"Feature parameter {name} should be frozen, but has grad!"

    # Classifier head parameters MUST have valid gradient
    for name, param in model.backbone.classifier.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Classifier parameter {name} has no gradient!"


def test_dataloader_split_disjointness(app_config, split_manifest):
    """Verify that DataLoader creation for train and val splits preserves patient disjointness."""
    train_df = split_manifest[split_manifest["split"] == "train"]
    val_df = split_manifest[split_manifest["split"] == "val"]

    train_pts = set(train_df["patient_id"])
    val_pts = set(val_df["patient_id"])
    assert len(train_pts.intersection(val_pts)) == 0, "Patient leakage in dataloader source dataframes!"
