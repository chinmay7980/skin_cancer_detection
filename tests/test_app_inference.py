"""Automated tests for Phase 7 application inference pipeline and artifact loading."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from src.data.features import PROHIBITED_COLUMNS
from src.inference.pipeline import (
    LOADER,
    predict_image,
    predict_metadata,
    predict_multimodal,
    preprocess_image_tensor,
    preprocess_metadata_dict,
)


def test_image_preprocessing_tensor_shape():
    """Verify that preprocess_image_tensor outputs tensor with shape (1, 3, 160, 160)."""
    img = Image.new("RGB", (250, 300), color=(100, 150, 200))
    tensor = preprocess_image_tensor(img, image_size=160)
    assert tensor.shape == (1, 3, 160, 160)
    assert tensor.dtype == torch.float32


def test_missing_metadata_imputed_without_error():
    """Verify that missing metadata fields are imputed using fitted training statistics."""
    _, preprocessor, _ = LOADER.get_metadata_pipeline()
    # Provide only a single feature; all others should be gracefully imputed
    partial_meta = {"age_approx": 45.0}
    X_meta = preprocess_metadata_dict(partial_meta, preprocessor)
    assert X_meta.shape == (1, 50)
    assert not np.isnan(X_meta).any(), "NaN values found in imputed feature tensor."


def test_input_feature_validation_prohibited_leakage():
    """Verify that passing prohibited leakage features to metadata inference raises ValueError."""
    _, preprocessor, _ = LOADER.get_metadata_pipeline()
    for col in ["target", "iddx_full", "patient_id", "mel_thick_mm"]:
        with pytest.raises(ValueError, match="CRITICAL LEAKAGE ERROR"):
            preprocess_metadata_dict({"age_approx": 50.0, col: "leakage_val"}, preprocessor)


def test_no_refitting_during_inference():
    """Verify that preprocessing pipeline parameters remain strictly frozen during inference."""
    _, preprocessor, _ = LOADER.get_metadata_pipeline()
    # Check numeric scaler means before inference
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    means_before = scaler.mean_.copy()

    # Run inference multiple times with different extreme values
    _ = preprocess_metadata_dict({"age_approx": 99.0, "clin_size_long_diam_mm": 50.0}, preprocessor)
    _ = preprocess_metadata_dict({"age_approx": 5.0, "clin_size_long_diam_mm": 0.1}, preprocessor)

    means_after = scaler.mean_
    np.testing.assert_array_equal(means_before, means_after, err_msg="Scaler parameters were modified during inference!")


def test_predict_metadata_output_contract():
    """Verify that predict_metadata returns expected contract, probability bounds, and flags."""
    sample_meta = {
        "sex": "female",
        "age_approx": 40.0,
        "anatom_site_general": "lower extremity",
        "tbp_lv_location_simple": "Leg",
        "clin_size_long_diam_mm": 4.0,
        "tbp_lv_areaMM2": 12.0,
    }
    res = predict_metadata(sample_meta)

    assert "raw_score" in res
    assert 0.0 <= res["raw_score"] <= 1.0
    assert "raw_threshold" in res
    assert isinstance(res["is_elevated_risk"], bool)

    if "calibrated_score" in res:
        assert 0.0 <= res["calibrated_score"] <= 1.0
        assert isinstance(res["is_calibrated_elevated_risk"], bool)


def test_predict_image_output_contract():
    """Verify that predict_image returns valid probability bounds on a synthetic test image."""
    dummy_img = Image.new("RGB", (160, 160), color=(150, 90, 70))
    res = predict_image(dummy_img)

    assert res["modality"] == "Image CNN Only"
    assert 0.0 <= res["raw_score"] <= 1.0
    assert isinstance(res["is_elevated_risk"], bool)
    assert res["raw_threshold"] == 0.9284


def test_predict_multimodal_output_contract():
    """Verify that predict_multimodal runs both weighted and unweighted models correctly."""
    dummy_img = Image.new("RGB", (160, 160), color=(120, 80, 60))
    sample_meta = {
        "sex": "male",
        "age_approx": 60.0,
        "anatom_site_general": "torso",
        "clin_size_long_diam_mm": 5.5,
    }

    # Test weighted
    res_w = predict_multimodal(dummy_img, sample_meta, weighted=True)
    assert res_w["modality"] == "Multimodal Fusion"
    assert 0.0 <= res_w["raw_score"] <= 1.0
    assert res_w["raw_threshold"] == 0.9688

    # Test unweighted
    res_u = predict_multimodal(dummy_img, sample_meta, weighted=False)
    assert res_u["modality"] == "Multimodal Fusion"
    assert 0.0 <= res_u["raw_score"] <= 1.0
    assert res_u["raw_threshold"] == 0.5554
