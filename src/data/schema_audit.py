"""Schema auditing, column categorization, and data leakage detection."""

from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd

# 1. Identifiers: strictly join / grouping keys, NEVER predictive features
IDENTIFIER_COLUMNS: List[str] = [
    "isic_id",     # Primary key, joins metadata to HDF5 image
    "patient_id",  # Patient grouping key for cross-validation splitting
    "lesion_id",   # Lesion tracker ID (94.5% missing)
]

# 2. Supervised target
TARGET_COLUMN: str = "target"

# 3. Data Leakage Columns: Post-diagnosis, biopsy, or pathology outcomes
# These must NEVER be fed to predictive models.
LEAKAGE_COLUMNS: List[str] = [
    "iddx_full",          # Full diagnostic outcome (e.g. 'Benign', 'Malignant')
    "iddx_1",             # Primary diagnostic outcome category
    "iddx_2",             # Secondary diagnostic outcome
    "iddx_3",             # Tertiary diagnostic outcome
    "iddx_4",             # Additional diagnostic outcome
    "iddx_5",             # Additional diagnostic outcome
    "mel_mitotic_index",  # Histological mitotic rate (biopsy pathology only)
    "mel_thick_mm",       # Breslow thickness (excisional biopsy histology only)
]

# 4. Administrative / Metadata Columns: Non-predictive provenance
NON_PREDICTIVE_COLUMNS: List[str] = [
    "attribution",        # Hospital / clinical source institution
    "copyright_license",  # License string (e.g. 'CC-BY')
]

# 5. Invariant / Zero-Variance Columns
CONSTANT_COLUMNS: List[str] = [
    "image_type",         # 100% constant: 'TBP tile: close-up'
]

# 6. Candidate Predictor Features (Available legitimately at inference time)
# Note: Further feature selection and encoding will occur in Phase 3.
CANDIDATE_CLINICAL_FEATURES: List[str] = [
    "age_approx",
    "sex",
    "anatom_site_general",
    "clin_size_long_diam_mm",
]

CANDIDATE_TBP_METADATA_FEATURES: List[str] = [
    "tbp_tile_type",
    "tbp_lv_location",
    "tbp_lv_location_simple",
    "tbp_lv_x",
    "tbp_lv_y",
    "tbp_lv_z",
]

CANDIDATE_TBP_COLOR_FEATURES: List[str] = [
    "tbp_lv_A", "tbp_lv_Aext",
    "tbp_lv_B", "tbp_lv_Bext",
    "tbp_lv_C", "tbp_lv_Cext",
    "tbp_lv_H", "tbp_lv_Hext",
    "tbp_lv_L", "tbp_lv_Lext",
    "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL",
    "tbp_lv_deltaLB", "tbp_lv_deltaLBnorm",
]

CANDIDATE_TBP_MORPHOLOGY_FEATURES: List[str] = [
    "tbp_lv_areaMM2",
    "tbp_lv_area_perim_ratio",
    "tbp_lv_perimeterMM",
    "tbp_lv_minorAxisMM",
    "tbp_lv_eccentricity",
    "tbp_lv_symm_2axis",
    "tbp_lv_symm_2axis_angle",
    "tbp_lv_norm_border",
    "tbp_lv_norm_color",
    "tbp_lv_color_std_mean",
    "tbp_lv_radial_color_std_max",
    "tbp_lv_stdL",
    "tbp_lv_stdLExt",
]

# Scanner baseline detection confidence scores (require careful validation)
CANDIDATE_SCANNER_CONFIDENCE_FEATURES: List[str] = [
    "tbp_lv_nevi_confidence",
    "tbp_lv_dnn_lesion_confidence",
]

ALL_CANDIDATE_PREDICTORS: List[str] = (
    CANDIDATE_CLINICAL_FEATURES
    + CANDIDATE_TBP_METADATA_FEATURES
    + CANDIDATE_TBP_COLOR_FEATURES
    + CANDIDATE_TBP_MORPHOLOGY_FEATURES
    + CANDIDATE_SCANNER_CONFIDENCE_FEATURES
)


def audit_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform a comprehensive schema and integrity audit on the metadata dataframe."""
    total_rows = len(df)
    columns_present = list(df.columns)

    # 1. Identifier checks
    duplicate_isic = int(df["isic_id"].duplicated().sum()) if "isic_id" in df else None
    missing_isic = int(df["isic_id"].isnull().sum()) if "isic_id" in df else None
    missing_patient = int(df["patient_id"].isnull().sum()) if "patient_id" in df else None
    unique_patients = int(df["patient_id"].nunique()) if "patient_id" in df else None

    # Patient lesion counts distribution
    if "patient_id" in df:
        patient_counts = df["patient_id"].value_counts()
        patient_stats = {
            "min_lesions": int(patient_counts.min()),
            "median_lesions": float(patient_counts.median()),
            "mean_lesions": float(patient_counts.mean()),
            "max_lesions": int(patient_counts.max()),
        }
    else:
        patient_stats = {}

    # 2. Target checks
    target_counts = df["target"].value_counts().to_dict() if "target" in df else {}
    target_values = sorted(list(df["target"].unique())) if "target" in df else []
    is_binary = target_values == [0, 1]

    # Patients with >=1 positive target
    if "patient_id" in df and "target" in df:
        pos_patients = int((df.groupby("patient_id")["target"].max() == 1).sum())
    else:
        pos_patients = None

    # 3. Missing values summary
    null_counts = df.isnull().sum()
    missing_summary = {
        col: {"null_count": int(val), "percent": round(float(val / total_rows * 100), 2)}
        for col, val in null_counts.items()
        if val > 0
    }

    # 4. Leakage columns verification
    leakage_present = [c for c in LEAKAGE_COLUMNS if c in columns_present]
    leakage_absent = [c for c in LEAKAGE_COLUMNS if c not in columns_present]

    return {
        "total_rows": total_rows,
        "total_columns": len(columns_present),
        "columns_present": columns_present,
        "duplicate_isic": duplicate_isic,
        "missing_isic": missing_isic,
        "missing_patient": missing_patient,
        "unique_patients": unique_patients,
        "patients_with_positive_lesion": pos_patients,
        "patient_lesions_distribution": patient_stats,
        "target_values": target_values,
        "target_is_binary": is_binary,
        "target_counts": target_counts,
        "missing_summary": missing_summary,
        "leakage_columns_present": leakage_present,
        "leakage_columns_absent": leakage_absent,
        "candidate_predictors_count": len(ALL_CANDIDATE_PREDICTORS),
        "candidate_predictors": ALL_CANDIDATE_PREDICTORS,
    }
