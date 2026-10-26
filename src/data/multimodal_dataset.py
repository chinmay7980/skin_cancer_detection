"""Multimodal Dataset and DataLoader for combined Image + Clinical/3D Metadata."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from src.data.features import get_feature_lists, validate_feature_allowlist
from src.data.image_dataset import get_image_transforms
from src.models.tabular_pipeline import create_preprocessor


class ISICMultimodalDataset(Dataset):
    """PyTorch Dataset returning (image_tensor, metadata_tensor, target_tensor).

    Design Principles:
    - Lazy image loading: Never loads full image archive into RAM.
    - Preprocessed metadata: Compact float32 tensor (~16 MB for 80k rows).
    - Leakage isolation: Preprocessor fitted on training cohort only.
    - No identifier features: `isic_id` and `patient_id` are never fed to the model.
    """

    def __init__(
        self,
        isic_ids: List[str],
        metadata_features: np.ndarray,
        hdf5_path: Union[str, Path],
        targets: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
    ):
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.is_file():
            raise FileNotFoundError(f"HDF5 archive not found at: {self.hdf5_path}")

        if len(isic_ids) != len(metadata_features):
            raise ValueError(f"Mismatch: {len(isic_ids)} image IDs vs {len(metadata_features)} metadata rows.")

        self.isic_ids = list(isic_ids)
        self.metadata_features = metadata_features.astype(np.float32)
        self.targets = targets.astype(np.float32) if targets is not None else None
        self.transform = transform
        self._h5_file: Optional[h5py.File] = None

    def _get_h5(self) -> h5py.File:
        """Lazily initialize HDF5 handle per worker process."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.hdf5_path, "r")
        return self._h5_file

    def __len__(self) -> int:
        return len(self.isic_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        isic_id = self.isic_ids[idx]
        h5 = self._get_h5()

        # 1. Image extraction & decoding
        try:
            if isic_id not in h5:
                raise KeyError(f"Image key '{isic_id}' not found in HDF5 archive.")
            raw_bytes = h5[isic_id][()]
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception:
            # Fallback for corrupt or missing entries
            img = Image.new("RGB", (160, 160), color=(128, 128, 128))

        if self.transform is not None:
            image_tensor = self.transform(img)
        else:
            image_tensor = transforms.ToTensor()(img)

        # 2. Metadata representation
        meta_tensor = torch.tensor(self.metadata_features[idx], dtype=torch.float32)

        # 3. Target label
        if self.targets is not None:
            target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            target_tensor = torch.tensor(-1.0, dtype=torch.float32)

        return image_tensor, meta_tensor, target_tensor

    def close(self) -> None:
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
            self._h5_file = None

    def __del__(self) -> None:
        self.close()


def prepare_multimodal_features(
    train_cohort_df: pd.DataFrame,
    val_df: pd.DataFrame,
    full_metadata_df: pd.DataFrame,
    include_scanner_confidence: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """Preprocess metadata features strictly isolating train from validation.

    Guarantees:
    - Preprocessor is fit ONLY on the training cohort records.
    - Validation records are transformed using the fitted preprocessor without refitting.
    - Prohibited leakage columns are strictly excluded.
    """
    num_cols, cat_cols = get_feature_lists(include_scanner_confidence=include_scanner_confidence)
    validate_feature_allowlist(num_cols + cat_cols)

    # Merge metadata features based on isic_id while preserving exact dataframe ordering
    train_meta = train_cohort_df[["isic_id"]].merge(full_metadata_df, on="isic_id", how="left")
    val_meta = val_df[["isic_id"]].merge(full_metadata_df, on="isic_id", how="left")

    preprocessor = create_preprocessor(numeric_features=num_cols, categorical_features=cat_cols, scale_numeric=True)

    # Fit strictly on train cohort only
    X_train_meta = preprocessor.fit_transform(train_meta)
    # Transform validation without fitting
    X_val_meta = preprocessor.transform(val_meta)

    return X_train_meta.astype(np.float32), X_val_meta.astype(np.float32), preprocessor


def create_multimodal_dataloader(
    isic_ids: List[str],
    metadata_features: np.ndarray,
    hdf5_path: Union[str, Path],
    targets: Optional[np.ndarray] = None,
    batch_size: int = 16,
    is_training: bool = False,
    image_size: int = 160,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a configured DataLoader for multimodal training or validation."""
    transform = get_image_transforms(image_size=image_size, is_training=is_training)
    dataset = ISICMultimodalDataset(
        isic_ids=isic_ids,
        metadata_features=metadata_features,
        hdf5_path=hdf5_path,
        targets=targets,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_training,
    )
