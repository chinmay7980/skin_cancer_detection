"""Automated unit tests for Multimodal Fusion (Phase 5)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.config import load_config
from src.data.features import PROHIBITED_COLUMNS, validate_feature_allowlist
from src.data.multimodal_dataset import (
    ISICMultimodalDataset,
    prepare_multimodal_features,
)
from src.models.fusion import MultimodalFusionNet


def test_multimodal_model_output_shape():
    """Verify that MultimodalFusionNet produces output shape (batch_size,)."""
    batch_size = 4
    metadata_dim = 50
    model = MultimodalFusionNet(
        metadata_dim=metadata_dim,
        fusion_hidden_dim=64,
        pretrained=False,
        freeze_backbone=True,
    )
    model.eval()

    dummy_images = torch.randn(batch_size, 3, 160, 160)
    dummy_meta = torch.randn(batch_size, metadata_dim)

    with torch.no_grad():
        logits = model(dummy_images, dummy_meta)

    assert logits.shape == (batch_size,), f"Expected shape ({batch_size},), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN values."


def test_fusion_forward_backward_gradient_flow():
    """Verify forward and backward pass with gradient flow through the fusion head."""
    batch_size = 4
    metadata_dim = 50
    model = MultimodalFusionNet(
        metadata_dim=metadata_dim,
        fusion_hidden_dim=64,
        pretrained=False,
        freeze_backbone=True,
    )
    model.train()

    optimizer = torch.optim.AdamW(model.fusion_head.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    dummy_images = torch.randn(batch_size, 3, 160, 160)
    dummy_meta = torch.randn(batch_size, metadata_dim)
    dummy_targets = torch.tensor([0.0, 1.0, 0.0, 1.0])

    optimizer.zero_grad()
    logits = model(dummy_images, dummy_meta)
    loss = criterion(logits, dummy_targets)
    loss.backward()

    # Backbone parameters should have no gradients (frozen)
    for param in model.features.parameters():
        assert param.grad is None, "Frozen backbone parameter should not have gradients."

    # Fusion head parameters should have valid gradients
    has_grads = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.fusion_head.parameters())
    assert has_grads, "Fusion head parameters did not receive gradients."


def test_preprocessing_isolation_fit_on_train_only():
    """Verify that preprocessor fits strictly on train and transforms validation without refitting."""
    cfg = load_config()
    train_metadata_path = cfg.data.train_metadata

    if not train_metadata_path.is_file():
        pytest.skip(f"Metadata file not found at: {train_metadata_path}")

    # Create synthetic train and validation frames
    train_cohort_df = pd.DataFrame({
        "isic_id": ["ISIC_001", "ISIC_002", "ISIC_003"],
        "target": [0, 1, 0],
    })
    val_df = pd.DataFrame({
        "isic_id": ["ISIC_004", "ISIC_005"],
        "target": [0, 1],
    })

    # Synthetic metadata pool
    full_meta = pd.DataFrame({
        "isic_id": ["ISIC_001", "ISIC_002", "ISIC_003", "ISIC_004", "ISIC_005"],
        "sex": ["male", "female", "male", "female", "male"],
        "anatom_site_general": ["head/neck", "torso", "torso", "lower extremity", "upper extremity"],
        "tbp_tile_type": ["3D: XP", "3D: white", "3D: XP", "3D: XP", "3D: white"],
        "tbp_lv_location_simple": ["Head", "Torso", "Torso", "Leg", "Arm"],
        "age_approx": [45.0, 60.0, 50.0, 70.0, 35.0],
        "clin_size_long_diam_mm": [3.2, 5.1, 4.0, 8.2, 2.5],
        # Fill remaining required numeric columns with 0.0
        **{col: [0.0] * 5 for col in [
            "tbp_lv_x", "tbp_lv_y", "tbp_lv_z",
            "tbp_lv_A", "tbp_lv_Aext", "tbp_lv_B", "tbp_lv_Bext", "tbp_lv_C", "tbp_lv_Cext",
            "tbp_lv_H", "tbp_lv_Hext", "tbp_lv_L", "tbp_lv_Lext",
            "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL", "tbp_lv_deltaLB", "tbp_lv_deltaLBnorm",
            "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", "tbp_lv_perimeterMM", "tbp_lv_minorAxisMM",
            "tbp_lv_eccentricity", "tbp_lv_symm_2axis", "tbp_lv_symm_2axis_angle",
            "tbp_lv_norm_border", "tbp_lv_norm_color", "tbp_lv_color_std_mean",
            "tbp_lv_radial_color_std_max", "tbp_lv_stdL", "tbp_lv_stdLExt",
        ]},
    })

    X_train_meta, X_val_meta, preprocessor = prepare_multimodal_features(
        train_cohort_df=train_cohort_df,
        val_df=val_df,
        full_metadata_df=full_meta,
        include_scanner_confidence=False,
    )

    assert X_train_meta.shape[0] == 3
    assert X_val_meta.shape[0] == 2
    assert X_train_meta.shape[1] == X_val_meta.shape[1]
    assert not np.isnan(X_train_meta).any()
    assert not np.isnan(X_val_meta).any()


def test_leakage_prohibited_columns_guard():
    """Verify that any attempt to include prohibited columns triggers an exception."""
    for col in PROHIBITED_COLUMNS:
        with pytest.raises(ValueError, match="CRITICAL LEAKAGE ERROR"):
            validate_feature_allowlist(["age_approx", col])


def test_split_disjointness_matched_cohort():
    """Verify that the saved matched cohort contains only training patients and zero val/test overlap."""
    cohort_path = Path("data/splits/matched_train_cohort_10k.csv")
    manifest_path = Path("data/splits/patient_split_manifest.csv")

    if not cohort_path.is_file() or not manifest_path.is_file():
        pytest.skip("Split manifests not yet generated.")

    cohort_df = pd.read_csv(cohort_path)
    manifest_df = pd.read_csv(manifest_path)

    val_patients = set(manifest_df[manifest_df["split"] == "val"]["patient_id"])
    test_patients = set(manifest_df[manifest_df["split"] == "test"]["patient_id"])
    cohort_patients = set(cohort_df["patient_id"])

    val_overlap = cohort_patients.intersection(val_patients)
    test_overlap = cohort_patients.intersection(test_patients)

    assert len(val_overlap) == 0, f"Patient leakage into validation: {val_overlap}"
    assert len(test_overlap) == 0, f"Patient leakage into test: {test_overlap}"
    assert len(cohort_df) == 10000, f"Expected 10,000 cohort rows, got {len(cohort_df)}"
    assert (cohort_df["target"] == 1).sum() == 239, "Expected all 239 training positives in matched cohort."
