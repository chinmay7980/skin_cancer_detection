"""Feature definitions and allowlists for tabular models."""

from __future__ import annotations

from typing import List, Tuple
import pandas as pd

# 1. Prohibited columns (Strict Leakage & Non-Predictive Guardrails)
PROHIBITED_COLUMNS: List[str] = [
    # Identifiers
    "isic_id",
    "patient_id",
    "lesion_id",
    # Target
    "target",
    # Ground-truth pathology / post-biopsy diagnosis
    "iddx_full",
    "iddx_1",
    "iddx_2",
    "iddx_3",
    "iddx_4",
    "iddx_5",
    "mel_mitotic_index",
    "mel_thick_mm",
    # Administrative provenance
    "attribution",
    "copyright_license",
    # Zero-variance constant
    "image_type",
]

# 2. Primary Baseline Categorical Features (Low cardinality, legitimately available at prediction time)
PRIMARY_CATEGORICAL_FEATURES: List[str] = [
    "sex",                     # Categorical: 'male', 'female' (2 categories)
    "anatom_site_general",     # Anatomical site: 5 categories
    "tbp_tile_type",           # Camera modality: '3D: XP', '3D: white' (2 categories)
    "tbp_lv_location_simple",  # Simplified body location (8 categories)
]

# 3. Primary Baseline Numeric Features
PRIMARY_NUMERIC_FEATURES: List[str] = [
    # Clinical / Demographics
    "age_approx",
    "clin_size_long_diam_mm",
    # 3D Coordinates
    "tbp_lv_x",
    "tbp_lv_y",
    "tbp_lv_z",
    # Colorimetry (Lab color space)
    "tbp_lv_A", "tbp_lv_Aext",
    "tbp_lv_B", "tbp_lv_Bext",
    "tbp_lv_C", "tbp_lv_Cext",
    "tbp_lv_H", "tbp_lv_Hext",
    "tbp_lv_L", "tbp_lv_Lext",
    # Color Contrast
    "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL",
    "tbp_lv_deltaLB", "tbp_lv_deltaLBnorm",
    # Geometry & Morphology
    "tbp_lv_areaMM2",
    "tbp_lv_area_perim_ratio",
    "tbp_lv_perimeterMM",
    "tbp_lv_minorAxisMM",
    "tbp_lv_eccentricity",
    "tbp_lv_symm_2axis",
    "tbp_lv_symm_2axis_angle",
    # Texture & Uniformity
    "tbp_lv_norm_border",
    "tbp_lv_norm_color",
    "tbp_lv_color_std_mean",
    "tbp_lv_radial_color_std_max",
    "tbp_lv_stdL",
    "tbp_lv_stdLExt",
]

# 4. Scanner Detection Confidence (For Ablation Only)
SCANNER_CONFIDENCE_FEATURES: List[str] = [
    "tbp_lv_nevi_confidence",
    "tbp_lv_dnn_lesion_confidence",
]


def get_feature_lists(include_scanner_confidence: bool = False) -> Tuple[List[str], List[str]]:
    """Return (numeric_features, categorical_features) allowlists."""
    numeric = list(PRIMARY_NUMERIC_FEATURES)
    if include_scanner_confidence:
        numeric.extend(SCANNER_CONFIDENCE_FEATURES)
    categorical = list(PRIMARY_CATEGORICAL_FEATURES)
    return numeric, categorical


def validate_feature_allowlist(feature_names: List[str]) -> None:
    """Ensure no prohibited or leakage features exist in the selected feature set."""
    prohibited_set = set(PROHIBITED_COLUMNS)
    violations = [f for f in feature_names if f in prohibited_set]
    if violations:
        raise ValueError(f"CRITICAL LEAKAGE ERROR: Prohibited features found in feature set: {violations}")
