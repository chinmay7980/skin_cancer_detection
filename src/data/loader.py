"""Memory-conscious dataset loading and HDF5 image lookup."""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Union
import h5py
import pandas as pd
from PIL import Image


def load_metadata(
    filepath: Union[str, Path],
    usecols: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    low_memory: bool = False,
) -> pd.DataFrame:
    """Load metadata CSV with optional column filtering and row limit to save memory."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path, usecols=usecols, nrows=nrows, low_memory=low_memory)
    return df



def get_image_from_hdf5(
    hdf5_source: Union[str, Path, h5py.File],
    isic_id: str,
) -> Image.Image:
    """Lazily read a single image from an open HDF5 file or path and decode to PIL Image."""
    if isinstance(hdf5_source, (str, Path)):
        with h5py.File(hdf5_source, "r") as f:
            return _decode_h5_image(f, isic_id)
    else:
        return _decode_h5_image(hdf5_source, isic_id)


def _decode_h5_image(h5_file: h5py.File, isic_id: str) -> Image.Image:
    if isic_id not in h5_file:
        raise KeyError(f"Image key '{isic_id}' not found in HDF5 archive.")
    raw_bytes = h5_file[isic_id][()]
    img = Image.open(io.BytesIO(raw_bytes))
    fmt = img.format or "JPEG"
    rgb_img = img.convert("RGB")
    rgb_img.format = fmt
    return rgb_img



def verify_image_presence(
    hdf5_path: Union[str, Path],
    isic_ids: List[str],
) -> List[bool]:
    """Check whether a list of image IDs exist in the HDF5 archive without reading pixels."""
    with h5py.File(hdf5_path, "r") as f:
        return [iid in f for iid in isic_ids]
