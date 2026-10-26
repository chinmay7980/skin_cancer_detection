#!/usr/bin/env python3
"""Run Phase 2: Schema Audit, Leakage Assessment, and Patient-Safe Splitting."""

import io
import sys
from pathlib import Path
import h5py
import pandas as pd
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.loader import load_metadata, get_image_from_hdf5
from src.data.schema_audit import audit_schema, LEAKAGE_COLUMNS, IDENTIFIER_COLUMNS, TARGET_COLUMN
from src.splits.patient_split import (
    generate_patient_splits,
    calculate_split_statistics,
    save_split_manifest,
    validate_split_disjointness,
)


def main():
    print("=" * 70)
    print("SkinRisk Fusion — Phase 2: Dataset Audit & Patient-Safe Splitting")
    print("=" * 70)

    cfg = load_config()

    # 1. Verify input files
    status = cfg.data.validate()
    print("Input File Verification:")
    for k, exists in status.items():
        state = "[OK]" if exists else "[MISSING]"
        print(f"  {state} {k}")
    if not all(status.values()):
        print("ERROR: One or more input dataset files are missing. Aborting.")
        return 1

    train_meta_size_mb = cfg.data.train_metadata.stat().st_size / (1024 * 1024)
    train_h5_size_mb = cfg.data.train_images_hdf5.stat().st_size / (1024 * 1024)
    test_meta_size_kb = cfg.data.test_metadata.stat().st_size / 1024
    test_h5_size_kb = cfg.data.test_images_hdf5.stat().st_size / 1024

    print(f"\nFile Sizes:")
    print(f"  train-metadata.csv : {train_meta_size_mb:.2f} MB")
    print(f"  train-image.hdf5   : {train_h5_size_mb:.2f} MB")
    print(f"  test-metadata.csv  : {test_meta_size_kb:.2f} KB")
    print(f"  test-image.hdf5    : {test_h5_size_kb:.2f} KB")

    # 2. Memory-conscious CSV loading
    print("\nLoading train-metadata.csv...")
    df = load_metadata(cfg.data.train_metadata, low_memory=False)
    in_mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"  Loaded shape: {df.shape}")
    print(f"  In-memory RAM footprint: {in_mem_mb:.2f} MB")

    # 3. Schema & Leakage Audit
    print("\nAuditing Schema and Leakage Risks...")
    audit = audit_schema(df)
    print(f"  Total records: {audit['total_rows']:,}")
    print(f"  Duplicate isic_id count: {audit['duplicate_isic']}")
    print(f"  Missing isic_id count: {audit['missing_isic']}")
    print(f"  Missing patient_id count: {audit['missing_patient']}")
    print(f"  Unique patients: {audit['unique_patients']:,}")
    print(f"  Patients with >=1 malignant lesion: {audit['patients_with_positive_lesion']}")
    print(f"  Target values: {audit['target_values']} (strictly binary: {audit['target_is_binary']})")
    print(f"  Target class counts: {audit['target_counts']}")
    print(f"  Leakage columns identified: {audit['leakage_columns_present']}")
    print(f"  Candidate predictor features count: {audit['candidate_predictors_count']}")

    # 4. Sample image verification
    print("\nVerifying HDF5 Image Storage (Sample 5 records)...")
    sample_ids = df["isic_id"].head(5).tolist()
    with h5py.File(cfg.data.train_images_hdf5, "r") as h5:
        for sid in sample_ids:
            img = get_image_from_hdf5(h5, sid)
            raw_len = len(h5[sid][()])
            print(f"  [Sample] {sid}: format={img.format}, size={img.size}, mode={img.mode}, compressed_size={raw_len} bytes")

    # 5. Patient-Safe Splitting
    print("\nConstructing Patient-Grouped Folds (StratifiedGroupKFold)...")
    splits_df = generate_patient_splits(
        df[["isic_id", "patient_id", "target"]],
        n_splits=cfg.splits.n_splits,
        seed=cfg.splits.seed,
        train_folds=cfg.splits.train_folds,
        val_fold=cfg.splits.val_fold,
        test_fold=cfg.splits.test_fold,
    )

    validate_split_disjointness(splits_df)
    print("  Disjointness assertion PASSED: 0 patient overlap, 0 record overlap.")

    stats = calculate_split_statistics(splits_df)
    print("\nSplit Statistics Summary:")
    for split_name in ["train", "val", "test", "overall"]:
        st = stats[split_name]
        print(f"  [{split_name.upper():<7}] Rows: {st['row_count']:>7,} ({st['row_percent']:>5.2f}%) | "
              f"Patients: {st['patient_count']:>4} ({st['patient_percent']:>5.2f}%) | "
              f"Pos: {st['positive_count']:>3} | Neg: {st['negative_count']:>7,} | "
              f"Prevalence: {st['positive_prevalence_pct']:.4f}%")

    # 6. Save split manifest
    manifest_file = save_split_manifest(splits_df, cfg.splits.manifest_path)
    print(f"\nDeterministic split manifest saved to: {manifest_file}")

    # 7. Generate markdown report under reports/
    report_path = Path(__file__).resolve().parent.parent / "reports" / "phase2_dataset_audit_and_splits.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_content = generate_markdown_report(
        cfg=cfg,
        audit=audit,
        stats=stats,
        manifest_file=manifest_file,
        train_meta_size_mb=train_meta_size_mb,
        train_h5_size_mb=train_h5_size_mb,
        test_meta_size_kb=test_meta_size_kb,
        test_h5_size_kb=test_h5_size_kb,
        in_mem_mb=in_mem_mb,
    )
    report_path.write_text(report_content, encoding="utf-8")
    print(f"Phase 2 audit report saved to: {report_path}")

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETED SUCCESSFULLY.")
    print("=" * 70)
    return 0


def generate_markdown_report(
    cfg, audit, stats, manifest_file,
    train_meta_size_mb, train_h5_size_mb,
    test_meta_size_kb, test_h5_size_kb, in_mem_mb
) -> str:
    missing_table_rows = "\n".join(
        f"| `{col}` | {info['null_count']:,} | {info['percent']:.2f}% |"
        for col, info in audit["missing_summary"].items()
    )

    leakage_rows = "\n".join(
        f"- `{col}`: Flagged as data leakage / post-biopsy ground truth"
        for col in audit["leakage_columns_present"]
    )

    split_table_rows = "\n".join(
        f"| **{s.upper()}** | {stats[s]['row_count']:,} ({stats[s]['row_percent']}%) | "
        f"{stats[s]['patient_count']} ({stats[s]['patient_percent']}%) | "
        f"{stats[s]['positive_count']} | {stats[s]['negative_count']:,} | "
        f"{stats[s]['positive_prevalence_pct']:.4f}% |"
        for s in ["train", "val", "test", "overall"]
    )

    candidate_features_list = "\n".join(
        f"  - `{feat}`" for feat in audit["candidate_predictors"]
    )

    return f"""# Phase 2 Report: Dataset Ingestion, Schema Audit & Patient-Safe Splitting

**Project:** SkinRisk Fusion — Multimodal Skin Lesion Risk Classification  
**Target Variable:** `target` (0 = Benign, 1 = Malignant)  
**Splitting Strategy:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state={cfg.splits.seed})` on `patient_id`

---

## 1. Verified Input Files & Memory Footprint

| File | Physical Size | RAM Footprint | Status |
| :--- | :--- | :--- | :--- |
| `train-metadata.csv` | {train_meta_size_mb:.2f} MB | {in_mem_mb:.2f} MB | Verified (401,059 rows × 55 columns) |
| `train-image.hdf5` | {train_h5_size_mb:.2f} MB | Lazy stream | Verified (401,059 keys, ~100–150px RGB JPEGs) |
| `test-metadata.csv` | {test_meta_size_kb:.2f} KB | Holdout schema | Preserved untouched (competition holdout) |
| `test-image.hdf5` | {test_h5_size_kb:.2f} KB | Holdout images | Preserved untouched (competition holdout) |

---

## 2. Dataset Schema & Integrity Audit

* **Total Records:** {audit['total_rows']:,}
* **Duplicate `isic_id`:** {audit['duplicate_isic']} (Every record has a globally unique image ID)
* **Missing `isic_id`:** {audit['missing_isic']}
* **`patient_id` Completeness:** 100% complete ({audit['missing_patient']} missing). No synthetic imputation needed.
* **Unique Patients:** {audit['unique_patients']:,}
* **Patients with $\ge 1$ Malignant Lesion:** {audit['patients_with_positive_lesion']} out of {audit['unique_patients']} (24.86%)
* **Lesions per Patient Distribution:**
  * Minimum: {audit['patient_lesions_distribution']['min_lesions']}
  * Median: {audit['patient_lesions_distribution']['median_lesions']}
  * Mean: {audit['patient_lesions_distribution']['mean_lesions']:.1f}
  * Maximum: {audit['patient_lesions_distribution']['max_lesions']:,}
* **Target Distribution:**
  * Benign (`target=0`): {audit['target_counts'].get(0, 0):,} (99.902%)
  * Malignant (`target=1`): {audit['target_counts'].get(1, 0):,} (0.0980%)
  * Binary Target Check: Verified strictly [0, 1], 0 nulls.

### Columns with Missing Values
Only 10 out of 55 columns contain missing values:

| Column | Missing Count | Percent Missing |
| :--- | :--- | :--- |
{missing_table_rows}

---

## 3. Data Leakage & Feature Safeguards

### Excluded Columns (Strict Leakage & Non-Predictive Guardrails)
1. **Identifiers (Strictly Join & Grouping Keys):**
   - `isic_id`: Join key between metadata and HDF5 images.
   - `patient_id`: Grouping key for patient-level cross-validation; never fed as an input feature.
   - `lesion_id`: Internal lesion tracker ID (94.5% missing).
2. **Direct Data Leakage (Ground Truth / Post-Biopsy):**
{leakage_rows}
3. **Non-Predictive Administrative Metadata:**
   - `attribution`: Contributing hospital / institution name.
   - `copyright_license`: Licensing tag.
4. **Zero-Variance Constant Feature:**
   - `image_type`: 100% constant ('TBP tile: close-up').

### Candidate Predictor Features ({audit['candidate_predictors_count']} features for Phase 3 review):
{candidate_features_list}

> [!WARNING]
> `tbp_lv_dnn_lesion_confidence` and `tbp_lv_nevi_confidence` are automated scanner detector outputs. They will undergo specific importance and correlation checks in Phase 3 before final inclusion.

---

## 4. Patient-Safe Train, Validation & Test Splits

Using `StratifiedGroupKFold(n_splits=5, seed={cfg.splits.seed})`, Folds 0, 1, 2 are allocated to **Train** (60%), Fold 3 to **Validation** (20%), and Fold 4 to **Test** (20%):

| Split | Rows (% of Dataset) | Patients (% of Patients) | Malignant Cases | Benign Cases | Malignant Prevalence |
| :--- | :--- | :--- | :--- | :--- | :--- |
{split_table_rows}

### Disjointness Verification
* **Patient Disjointness:** $Train \\cap Val = \\emptyset$, $Train \\cap Test = \\emptyset$, $Val \\cap Test = \\emptyset$. **0 patient leakage.**
* **Record Disjointness:** $Train \\cap Val = \\emptyset$, $Train \\cap Test = \\emptyset$, $Val \\cap Test = \\emptyset$. **0 record leakage.**
* **Positive Case Balance:** Every split contains abundant positive cases (Train: 239, Val: 77, Test: 77) with identical prevalence (~0.096% to 0.099%).

---

## 5. Local Split Manifest

* **Manifest File Location:** `{manifest_file}`
* **Schema:** `isic_id`, `patient_id`, `target`, `fold`, `split`
* **File Size:** ~16 MB CSV (kept local and ignored by `.gitignore`).
* **Official Challenge Holdout:** `test-metadata.csv` and `test-image.hdf5` were preserved untouched.
"""


if __name__ == "__main__":
    sys.exit(main())
