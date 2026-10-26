"""Automated unit tests for patient-grouped splitting and disjointness."""

import pytest
import pandas as pd
from src.config import load_config
from src.splits.patient_split import (
    load_split_manifest,
    validate_split_disjointness,
    generate_patient_splits,
    calculate_split_statistics,
)


@pytest.fixture(scope="module")
def app_config():
    return load_config()


@pytest.fixture(scope="module")
def split_manifest(app_config):
    """Load the saved split manifest generated during Phase 2."""
    assert app_config.splits.manifest_path.is_file(), (
        f"Split manifest does not exist at {app_config.splits.manifest_path}. "
        "Run scripts/run_phase2_audit.py first."
    )
    return load_split_manifest(app_config.splits.manifest_path)


def test_split_manifest_columns(split_manifest):
    """Verify split manifest has expected minimal columns."""
    expected_cols = {"isic_id", "patient_id", "target", "fold", "split"}
    assert set(split_manifest.columns) == expected_cols
    assert len(split_manifest) == 401059


def test_patient_disjointness(split_manifest):
    """Verify that NO patient appears in more than one split."""
    train_pts = set(split_manifest[split_manifest["split"] == "train"]["patient_id"])
    val_pts = set(split_manifest[split_manifest["split"] == "val"]["patient_id"])
    test_pts = set(split_manifest[split_manifest["split"] == "test"]["patient_id"])

    assert len(train_pts.intersection(val_pts)) == 0, "Patient leakage between train and val!"
    assert len(train_pts.intersection(test_pts)) == 0, "Patient leakage between train and test!"
    assert len(val_pts.intersection(test_pts)) == 0, "Patient leakage between val and test!"


def test_record_disjointness(split_manifest):
    """Verify that NO isic_id appears in more than one split."""
    train_ids = set(split_manifest[split_manifest["split"] == "train"]["isic_id"])
    val_ids = set(split_manifest[split_manifest["split"] == "val"]["isic_id"])
    test_ids = set(split_manifest[split_manifest["split"] == "test"]["isic_id"])

    assert len(train_ids.intersection(val_ids)) == 0, "Record overlap between train and val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Record overlap between train and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Record overlap between val and test!"


def test_positive_cases_in_all_splits(split_manifest):
    """Verify that train, val, and test all contain positive cases."""
    stats = calculate_split_statistics(split_manifest)
    for s in ["train", "val", "test"]:
        assert stats[s]["positive_count"] > 0, f"Split '{s}' contains 0 positive cases!"
        # Verify prevalence is close to overall dataset prevalence (~0.098%)
        prev = stats[s]["positive_prevalence_pct"]
        assert 0.08 <= prev <= 0.12, f"Prevalence in '{s}' skewed unexpectedly: {prev:.4f}%"


def test_disjointness_validator_catches_leakage():
    """Verify that validate_split_disjointness raises AssertionError on corrupted splits."""
    bad_df = pd.DataFrame({
        "isic_id": ["id1", "id2", "id3"],
        "patient_id": ["P1", "P1", "P2"],  # P1 appears in both train and val!
        "target": [0, 1, 0],
        "split": ["train", "val", "test"],
    })

    with pytest.raises(AssertionError, match="Patient leakage detected"):
        validate_split_disjointness(bad_df)


def test_split_reproducibility(app_config):
    """Verify that generate_patient_splits is strictly deterministic across repeated runs."""
    # Synthetic test set with 20 patients, each with benign and malignant cases
    data = []
    for p_id in range(20):
        for lesion_num in range(10):
            data.append({
                "isic_id": f"ID_{p_id}_{lesion_num}",
                "patient_id": f"PATIENT_{p_id}",
                "target": 1 if lesion_num < 2 else 0,
            })
    toy_df = pd.DataFrame(data)

    run1 = generate_patient_splits(toy_df, n_splits=5, seed=42)
    run2 = generate_patient_splits(toy_df, n_splits=5, seed=42)

    assert (run1["fold"].values == run2["fold"].values).all()
    assert (run1["split"].values == run2["split"].values).all()

