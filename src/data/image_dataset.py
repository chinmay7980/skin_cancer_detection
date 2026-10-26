"""Lazy-loading PyTorch Dataset and DataLoader for HDF5 skin lesion images."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_image_transforms(
    image_size: int = 160,
    is_training: bool = False,
) -> transforms.Compose:
    """Build torchvision transforms for ISIC skin lesion images.

    Training transforms apply conservative spatial flips/rotations and subtle
    color adjustments (lesion color carries diagnostic significance).
    Validation transforms are strictly deterministic: resize, to-tensor, normalize.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            # Subtle jitter only: preserve critical lesion color information
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class ISICImageDataset(Dataset):
    """PyTorch Dataset reading lesion images on-the-fly from HDF5 archive.

    Guarantees:
    - Never loads full dataset into memory.
    - Safe multi-worker HDF5 handle handling (opens handle lazily per worker).
    - Strictly returns only (image_tensor, target_tensor). Never feeds IDs to model.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        hdf5_path: Union[str, Path],
        transform: Optional[Callable] = None,
        has_targets: bool = True,
    ):
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.is_file():
            raise FileNotFoundError(f"HDF5 archive not found at: {self.hdf5_path}")

        self.isic_ids = df["isic_id"].values.tolist()
        self.has_targets = has_targets and ("target" in df.columns)
        if self.has_targets:
            self.targets = df["target"].astype(np.float32).values
        else:
            self.targets = None

        self.transform = transform
        self._h5_file: Optional[h5py.File] = None

    def _get_h5(self) -> h5py.File:
        """Lazily initialize HDF5 handle per process / worker."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.hdf5_path, "r")
        return self._h5_file

    def __len__(self) -> int:
        return len(self.isic_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        isic_id = self.isic_ids[idx]
        h5 = self._get_h5()

        try:
            if isic_id not in h5:
                raise KeyError(f"Image key '{isic_id}' not found in HDF5 archive.")
            raw_bytes = h5[isic_id][()]
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception as e:
            # Fallback: create blank 160x160 RGB image if corrupt or missing
            img = Image.new("RGB", (160, 160), color=(128, 128, 128))

        if self.transform is not None:
            image_tensor = self.transform(img)
        else:
            image_tensor = transforms.ToTensor()(img)

        if self.has_targets and self.targets is not None:
            target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            target_tensor = torch.tensor(-1.0, dtype=torch.float32)

        return image_tensor, target_tensor

    def close(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
            self._h5_file = None

    def __del__(self):
        self.close()


def create_dataloader(
    df: pd.DataFrame,
    hdf5_path: Union[str, Path],
    batch_size: int = 8,
    is_training: bool = False,
    image_size: int = 160,
    num_workers: int = 0,
    pin_memory: bool = False,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> DataLoader:
    """Create a configured DataLoader for training or validation."""
    if max_samples is not None and max_samples < len(df):
        # Sample deterministically for smoke tests / subset experiments
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    transform = get_image_transforms(image_size=image_size, is_training=is_training)
    dataset = ISICImageDataset(df=df, hdf5_path=hdf5_path, transform=transform, has_targets=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_training,
    )
    return loader
