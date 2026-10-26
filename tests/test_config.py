"""Tests for configuration and environment setup."""

from pathlib import Path
from src.config import load_config, AppConfig


def test_default_config_loads():
    """Verify that default_config.yaml loads into a typed AppConfig dataclass."""
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.project.name == "SkinRisk Fusion"
    assert config.project.target_col == "target"
    assert config.project.patient_id_col == "patient_id"
    assert config.project.image_id_col == "isic_id"
    assert config.project.random_seed == 42


def test_local_dataset_paths_exist():
    """Verify that configured dataset files exist on disk without reading full data."""
    config = load_config()
    status = config.data.validate()

    assert status["raw_dir_exists"], f"Dataset directory missing: {config.data.raw_dir}"
    assert status["train_metadata_exists"], f"train-metadata.csv missing at {config.data.train_metadata}"
    assert status["train_images_hdf5_exists"], f"train-image.hdf5 missing at {config.data.train_images_hdf5}"
    assert status["test_metadata_exists"], f"test-metadata.csv missing at {config.data.test_metadata}"
    assert status["test_images_hdf5_exists"], f"test-image.hdf5 missing at {config.data.test_images_hdf5}"


def test_env_var_override(monkeypatch, tmp_path):
    """Verify that SKINRISK_DATA_DIR environment variable overrides default path."""
    dummy_dir = tmp_path / "dummy_isic"
    dummy_dir.mkdir()
    monkeypatch.setenv("SKINRISK_DATA_DIR", str(dummy_dir))

    config = load_config()
    assert config.data.raw_dir == dummy_dir
    # In a dummy dir without files, validation flags should be False
    status = config.data.validate()
    assert status["raw_dir_exists"] is True
    assert status["train_metadata_exists"] is False


def test_torch_device_resolution():
    """Verify that device resolution returns a valid torch device string."""
    config = load_config()
    device = config.hardware.get_torch_device()
    assert device in ("mps", "cuda", "cpu")
