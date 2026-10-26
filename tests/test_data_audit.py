"""Automated unit tests for data ingestion, schema audit, and HDF5 image access."""

import pytest
from src.config import load_config
from src.data.loader import load_metadata, get_image_from_hdf5, verify_image_presence
from src.data.schema_audit import (
    audit_schema,
    ALL_CANDIDATE_PREDICTORS,
    IDENTIFIER_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
)


@pytest.fixture(scope="module")
def app_config():
    return load_config()


@pytest.fixture(scope="module")
def metadata_sample(app_config):
    """Load small sample of metadata for fast test verification."""
    return load_metadata(app_config.data.train_metadata, nrows=1000)



def test_target_binary_integrity(app_config):
    """Verify target column contains only {0, 1} and no nulls."""
    df = load_metadata(app_config.data.train_metadata, usecols=["target"])
    unique_targets = set(df["target"].unique())
    assert unique_targets == {0, 1}, f"Unexpected target values: {unique_targets}"
    assert df["target"].isnull().sum() == 0, "Target column contains null values!"


def test_identifier_uniqueness(app_config):
    """Verify isic_id has no duplicates and no missing values."""
    df = load_metadata(app_config.data.train_metadata, usecols=["isic_id"])
    assert df["isic_id"].duplicated().sum() == 0, "Duplicate isic_id found!"
    assert df["isic_id"].isnull().sum() == 0, "Missing isic_id found!"


def test_hdf5_sample_image_decoding(app_config):
    """Verify that images can be read and decoded from HDF5 as valid RGB JPEGs."""
    df = load_metadata(app_config.data.train_metadata, usecols=["isic_id"])
    sample_ids = df["isic_id"].head(5).tolist()

    presence = verify_image_presence(app_config.data.train_images_hdf5, sample_ids)
    assert all(presence), "One or more sample image IDs are missing in HDF5!"

    for sid in sample_ids:
        img = get_image_from_hdf5(app_config.data.train_images_hdf5, sid)
        assert img.mode == "RGB", f"Expected RGB mode, got {img.mode}"
        assert img.format in ("JPEG", "None", None, "MPO"), f"Unexpected format: {img.format}"
        w, h = img.size
        assert w > 50 and h > 50, f"Image dimensions too small: ({w}, {h})"


def test_feature_list_safeguards():
    """Verify that target, identifiers, and leakage columns are strictly excluded from predictors."""
    # Target guard
    assert TARGET_COLUMN not in ALL_CANDIDATE_PREDICTORS, (
        f"CRITICAL: Target '{TARGET_COLUMN}' leaked into candidate predictors!"
    )

    # Identifiers guard
    for id_col in IDENTIFIER_COLUMNS:
        assert id_col not in ALL_CANDIDATE_PREDICTORS, (
            f"CRITICAL: Identifier '{id_col}' leaked into candidate predictors!"
        )

    # Leakage columns guard
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in ALL_CANDIDATE_PREDICTORS, (
            f"CRITICAL: Leakage column '{leak_col}' leaked into candidate predictors!"
        )
