"""Configuration manager for SkinRisk Fusion.

Handles loading settings from YAML configs and environment variables,
with path validation without loading large datasets into memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default_config.yaml"


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    train_metadata: Path
    train_images_hdf5: Path
    test_metadata: Path
    test_images_hdf5: Path

    def validate(self) -> Dict[str, bool]:
        """Check whether dataset files exist on disk without reading them."""
        return {
            "raw_dir_exists": self.raw_dir.is_dir(),
            "train_metadata_exists": self.train_metadata.is_file(),
            "train_images_hdf5_exists": self.train_images_hdf5.is_file(),
            "test_metadata_exists": self.test_metadata.is_file(),
            "test_images_hdf5_exists": self.test_images_hdf5.is_file(),
        }


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    target_col: str
    patient_id_col: str
    image_id_col: str
    random_seed: int


@dataclass(frozen=True)
class HardwareConfig:
    preferred_device: str
    num_workers: int
    pin_memory: bool

    def get_torch_device(self) -> str:
        """Resolve the active PyTorch device (mps, cuda, or cpu)."""
        import torch

        if self.preferred_device in ("mps", "auto") and torch.backends.mps.is_available():
            return "mps"
        if self.preferred_device in ("cuda", "auto") and torch.cuda.is_available():
            return "cuda"
        return "cpu"


@dataclass(frozen=True)
class SplitsConfig:
    manifest_path: Path
    n_splits: int
    seed: int
    train_folds: Tuple[int, ...]
    val_fold: int
    test_fold: int


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    project: ProjectConfig
    hardware: HardwareConfig
    splits: SplitsConfig


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """Load configuration from YAML file, with optional env var overrides."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    # Allow overriding data directory from environment variable
    data_dict = raw_cfg.get("data", {})
    raw_dir_str = os.getenv("SKINRISK_DATA_DIR", data_dict.get("raw_dir", ""))
    raw_dir = Path(raw_dir_str).expanduser().resolve()

    train_meta_name = data_dict.get("train_metadata", "train-metadata.csv")
    train_h5_name = data_dict.get("train_images_hdf5", "train-image.hdf5")
    test_meta_name = data_dict.get("test_metadata", "test-metadata.csv")
    test_h5_name = data_dict.get("test_images_hdf5", "test-image.hdf5")

    data_cfg = DataConfig(
        raw_dir=raw_dir,
        train_metadata=raw_dir / train_meta_name,
        train_images_hdf5=raw_dir / train_h5_name,
        test_metadata=raw_dir / test_meta_name,
        test_images_hdf5=raw_dir / test_h5_name,
    )

    proj_dict = raw_cfg.get("project", {})
    proj_cfg = ProjectConfig(
        name=proj_dict.get("name", "SkinRisk Fusion"),
        target_col=proj_dict.get("target_col", "target"),
        patient_id_col=proj_dict.get("patient_id_col", "patient_id"),
        image_id_col=proj_dict.get("image_id_col", "isic_id"),
        random_seed=int(proj_dict.get("random_seed", 42)),
    )

    hw_dict = raw_cfg.get("hardware", {})
    hw_cfg = HardwareConfig(
        preferred_device=hw_dict.get("preferred_device", "auto"),
        num_workers=int(hw_dict.get("num_workers", 0)),
        pin_memory=bool(hw_dict.get("pin_memory", False)),
    )

    splits_dict = raw_cfg.get("splits", {})
    manifest_rel = splits_dict.get("manifest_path", "data/splits/patient_split_manifest.csv")
    manifest_path = (Path(__file__).resolve().parent.parent / manifest_rel).resolve()
    splits_cfg = SplitsConfig(
        manifest_path=manifest_path,
        n_splits=int(splits_dict.get("n_splits", 5)),
        seed=int(splits_dict.get("seed", 42)),
        train_folds=tuple(splits_dict.get("train_folds", [0, 1, 2])),
        val_fold=int(splits_dict.get("val_fold", 3)),
        test_fold=int(splits_dict.get("test_fold", 4)),
    )

    return AppConfig(data=data_cfg, project=proj_cfg, hardware=hw_cfg, splits=splits_cfg)

