"""Inference pipeline for SkinRisk Fusion models.

Provides unified, robust, and leakage-safe inference methods for:
1. Metadata-Only LightGBM (Raw and Platt-Calibrated)
2. Image-Only EfficientNet-B0 CNN
3. Multimodal Fusion Neural Network
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from src.data.features import (
    PRIMARY_CATEGORICAL_FEATURES,
    PRIMARY_NUMERIC_FEATURES,
    get_feature_lists,
    validate_feature_allowlist,
)
from src.data.image_dataset import get_image_transforms
from src.evaluation.calibration import apply_platt_calibrator
from src.models.cnn import EfficientNetB0Binary, get_torch_device
from src.models.fusion import MultimodalFusionNet


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Frozen validation decision thresholds
THRESHOLDS = {
    "metadata_raw": 0.3841,
    "metadata_calibrated": 0.0312,
    "image_cnn": 0.9284,
    "fusion_unweighted": 0.5554,
    "fusion_weighted": 0.9688,
}


class ModelArtifactLoader:
    """Manages loading and caching of frozen model weights and preprocessors."""

    def __init__(self, checkpoints_dir: Optional[Path | str] = None):
        self.checkpoints_dir = Path(checkpoints_dir or (PROJECT_ROOT / "checkpoints"))
        self.lgbm_model: Optional[Any] = None
        self.preprocessor: Optional[Any] = None
        self.calibrator: Optional[Any] = None
        self.cnn_model: Optional[EfficientNetB0Binary] = None
        self.fusion_unweighted: Optional[MultimodalFusionNet] = None
        self.fusion_weighted: Optional[MultimodalFusionNet] = None

    def get_metadata_pipeline(self) -> Tuple[Any, Any, Any]:
        """Load frozen LightGBM model, metadata preprocessor, and Platt calibrator."""
        if self.lgbm_model is None:
            lgbm_path = self.checkpoints_dir / "lgbm_matched_model.joblib"
            prep_path = self.checkpoints_dir / "metadata_preprocessor.joblib"
            cal_path = self.checkpoints_dir / "platt_calibrator.joblib"

            if not lgbm_path.is_file() or not prep_path.is_file():
                raise FileNotFoundError(
                    f"Metadata artifacts missing in {self.checkpoints_dir}. "
                    "Ensure 'lgbm_matched_model.joblib' and 'metadata_preprocessor.joblib' are present."
                )

            self.lgbm_model = joblib.load(lgbm_path)
            self.preprocessor = joblib.load(prep_path)
            if cal_path.is_file():
                self.calibrator = joblib.load(cal_path)

        return self.lgbm_model, self.preprocessor, self.calibrator

    def get_cnn_model(self, device: torch.device) -> EfficientNetB0Binary:
        """Load frozen EfficientNet-B0 image model."""
        if self.cnn_model is None:
            ckpt_path = self.checkpoints_dir / "cnn_efficientnet-b0_weighted_pos_20_best.pt"
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"CNN checkpoint not found at: {ckpt_path}")

            model = EfficientNetB0Binary(pretrained=False).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.cnn_model = model

        return self.cnn_model

    def get_fusion_model(self, device: torch.device, weighted: bool = True) -> MultimodalFusionNet:
        """Load frozen Multimodal Fusion network."""
        target_attr = "fusion_weighted" if weighted else "fusion_unweighted"
        cached = getattr(self, target_attr)

        if cached is None:
            ckpt_name = (
                "multimodal_fusion_weighted_pos_20_best.pt"
                if weighted
                else "multimodal_fusion_unweighted_best.pt"
            )
            ckpt_path = self.checkpoints_dir / ckpt_name
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Multimodal checkpoint not found at: {ckpt_path}")

            model = MultimodalFusionNet(
                metadata_dim=50,
                fusion_hidden_dim=128,
                pretrained=False,
                freeze_backbone=True,
            ).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            setattr(self, target_attr, model)
            cached = model

        return cached


# Global singleton loader
LOADER = ModelArtifactLoader()


def preprocess_metadata_dict(
    feature_dict: Dict[str, Any],
    preprocessor: Any,
) -> np.ndarray:
    """Validate, structure, and transform user-provided metadata features.

    Missing numeric features are automatically imputed using the training median;
    missing categoricals are imputed using the most-frequent training category.
    """
    num_cols, cat_cols = get_feature_lists(include_scanner_confidence=False)
    validate_feature_allowlist(list(feature_dict.keys()))

    row: Dict[str, Any] = {}
    for col in num_cols:
        val = feature_dict.get(col, np.nan)
        try:
            row[col] = float(val) if val is not None and not pd.isna(val) else np.nan
        except (ValueError, TypeError):
            row[col] = np.nan

    for col in cat_cols:
        val = feature_dict.get(col, None)
        row[col] = str(val) if val is not None and not pd.isna(val) else np.nan

    df = pd.DataFrame([row])
    # Transform using frozen preprocessor without refitting
    X_meta = preprocessor.transform(df)
    return X_meta.astype(np.float32)


def preprocess_image_tensor(
    image: Image.Image,
    image_size: int = 160,
) -> torch.Tensor:
    """Preprocess PIL Image into standardized (1, 3, H, W) tensor."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = get_image_transforms(image_size=image_size, is_training=False)
    tensor = transform(image).unsqueeze(0)
    return tensor


def predict_metadata(
    feature_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Run inference for metadata-only LightGBM model."""
    lgbm, preprocessor, calibrator = LOADER.get_metadata_pipeline()
    X_meta = preprocess_metadata_dict(feature_dict, preprocessor)

    raw_prob = float(lgbm.predict_proba(X_meta)[0, 1])
    raw_thresh = THRESHOLDS["metadata_raw"]

    result: Dict[str, Any] = {
        "modality": "Metadata Only",
        "model_name": "LightGBM (Matched Cohort)",
        "raw_score": round(raw_prob, 5),
        "raw_threshold": raw_thresh,
        "is_elevated_risk": raw_prob >= raw_thresh,
    }

    if calibrator is not None:
        cal_prob = float(apply_platt_calibrator(calibrator, np.array([raw_prob]))[0])
        cal_thresh = THRESHOLDS["metadata_calibrated"]
        result["calibrated_score"] = round(cal_prob, 5)
        result["calibrated_threshold"] = cal_thresh
        result["is_calibrated_elevated_risk"] = cal_prob >= cal_thresh

    return result


def predict_image(
    image: Image.Image,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run inference for image-only EfficientNet-B0 CNN."""
    dev = device or get_torch_device()
    model = LOADER.get_cnn_model(dev)

    img_tensor = preprocess_image_tensor(image).to(dev)
    with torch.no_grad():
        logit = model(img_tensor).item()
        prob = float(torch.sigmoid(torch.tensor(logit)).item())

    thresh = THRESHOLDS["image_cnn"]
    return {
        "modality": "Image CNN Only",
        "model_name": "EfficientNet-B0 (Weighted pos=20)",
        "raw_score": round(prob, 5),
        "raw_threshold": thresh,
        "is_elevated_risk": prob >= thresh,
    }


def predict_multimodal(
    image: Image.Image,
    feature_dict: Dict[str, Any],
    weighted: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run inference for multimodal neural fusion network."""
    dev = device or get_torch_device()
    model = LOADER.get_fusion_model(dev, weighted=weighted)
    _, preprocessor, _ = LOADER.get_metadata_pipeline()

    img_tensor = preprocess_image_tensor(image).to(dev)
    X_meta = preprocess_metadata_dict(feature_dict, preprocessor)
    meta_tensor = torch.tensor(X_meta, dtype=torch.float32, device=dev)

    with torch.no_grad():
        logit = model(img_tensor, meta_tensor).item()
        prob = float(torch.sigmoid(torch.tensor(logit)).item())

    thresh_key = "fusion_weighted" if weighted else "fusion_unweighted"
    thresh = THRESHOLDS[thresh_key]
    model_title = (
        "Multimodal Fusion Net (Weighted pos=20)"
        if weighted
        else "Multimodal Fusion Net (Unweighted)"
    )

    return {
        "modality": "Multimodal Fusion",
        "model_name": model_title,
        "raw_score": round(prob, 5),
        "raw_threshold": thresh,
        "is_elevated_risk": prob >= thresh,
    }
