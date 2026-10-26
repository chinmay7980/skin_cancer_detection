"""Patient-grouped cross-validation splitting and leakage verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def generate_patient_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    train_folds: Tuple[int, ...] = (0, 1, 2),
    val_fold: int = 3,
    test_fold: int = 4,
) -> pd.DataFrame:
    """Generate reproducible patient-grouped train (60%), val (20%), and test (20%) splits.

    Uses StratifiedGroupKFold on `patient_id` so that all lesions from a given
    patient stay exclusively in one split, preventing patient leakage.
    """
    required_cols = {"isic_id", "patient_id", "target"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for splitting: {missing}")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_array = pd.Series(index=df.index, dtype=int)
    for fold, (_, val_idx) in enumerate(sgkf.split(df, df["target"], groups=df["patient_id"])):
        fold_array.iloc[val_idx] = fold

    result_df = df.copy()
    result_df["fold"] = fold_array

    split_map = {f: "train" for f in train_folds}
    split_map[val_fold] = "val"
    split_map[test_fold] = "test"

    result_df["split"] = result_df["fold"].map(split_map)

    # Validate disjointness immediately
    validate_split_disjointness(result_df)

    return result_df


def validate_split_disjointness(df: pd.DataFrame) -> None:
    """Assert strict disjointness of patients and rows across all splits."""
    splits = ["train", "val", "test"]
    for s in splits:
        if s not in df["split"].unique():
            raise ValueError(f"Required split '{s}' is missing from the dataframe.")

    # 1. Patient disjointness
    patient_sets = {s: set(df[df["split"] == s]["patient_id"]) for s in splits}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = patient_sets[s1].intersection(patient_sets[s2])
            if overlap:
                raise AssertionError(
                    f"FATAL: Patient leakage detected between '{s1}' and '{s2}'. "
                    f"Overlapping patients count: {len(overlap)} (e.g. {list(overlap)[:3]})"
                )

    # 2. Row disjointness
    row_sets = {s: set(df[df["split"] == s]["isic_id"]) for s in splits}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = row_sets[s1].intersection(row_sets[s2])
            if overlap:
                raise AssertionError(
                    f"FATAL: Row/isic_id overlap detected between '{s1}' and '{s2}'."
                )

    # 3. Positive cases presence in all splits
    for s in splits:
        pos_count = (df[df["split"] == s]["target"] == 1).sum()
        if pos_count == 0:
            raise AssertionError(f"FATAL: Split '{s}' contains 0 positive cases!")


def calculate_split_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute detailed row counts, patient counts, and positive prevalence per split."""
    stats: Dict[str, Dict[str, Any]] = {}
    total_rows = len(df)
    total_patients = df["patient_id"].nunique()
    total_positives = (df["target"] == 1).sum()

    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        n_rows = len(sub)
        n_patients = sub["patient_id"].nunique()
        n_pos = int((sub["target"] == 1).sum())
        n_neg = n_rows - n_pos
        prev = (n_pos / n_rows * 100) if n_rows > 0 else 0.0

        stats[s] = {
            "row_count": n_rows,
            "row_percent": round(n_rows / total_rows * 100, 2),
            "patient_count": n_patients,
            "patient_percent": round(n_patients / total_patients * 100, 2),
            "positive_count": n_pos,
            "negative_count": n_neg,
            "positive_prevalence_pct": round(prev, 4),
        }

    # Overall dataset stats for comparison
    stats["overall"] = {
        "row_count": total_rows,
        "row_percent": 100.0,
        "patient_count": total_patients,
        "patient_percent": 100.0,
        "positive_count": int(total_positives),
        "negative_count": total_rows - int(total_positives),
        "positive_prevalence_pct": round(total_positives / total_rows * 100, 4),
    }

    return stats


def save_split_manifest(df: pd.DataFrame, output_path: Path | str) -> Path:
    """Save minimal identifiers required to reconstruct the splits reproducibly."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    manifest_cols = ["isic_id", "patient_id", "target", "fold", "split"]
    manifest_df = df[manifest_cols].copy()

    manifest_df.to_csv(out_file, index=False)
    return out_file


def load_split_manifest(manifest_path: Path | str) -> pd.DataFrame:
    """Load a previously saved split manifest CSV."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    return pd.read_csv(path)
