# Phase 2 Report: Dataset Ingestion, Schema Audit & Patient-Safe Splitting

**Project:** SkinRisk Fusion — Multimodal Skin Lesion Risk Classification  
**Target Variable:** `target` (0 = Benign, 1 = Malignant)  
**Splitting Strategy:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on `patient_id`

---

## 1. Verified Input Files & Memory Footprint

| File | Physical Size | RAM Footprint | Status |
| :--- | :--- | :--- | :--- |
| `train-metadata.csv` | 245.61 MB | 511.97 MB | Verified (401,059 rows × 55 columns) |
| `train-image.hdf5` | 1237.21 MB | Lazy stream | Verified (401,059 keys, ~100–150px RGB JPEGs) |
| `test-metadata.csv` | 2.15 KB | Holdout schema | Preserved untouched (competition holdout) |
| `test-image.hdf5` | 10.39 KB | Holdout images | Preserved untouched (competition holdout) |

---

## 2. Dataset Schema & Integrity Audit

* **Total Records:** 401,059
* **Duplicate `isic_id`:** 0 (Every record has a globally unique image ID)
* **Missing `isic_id`:** 0
* **`patient_id` Completeness:** 100% complete (0 missing). No synthetic imputation needed.
* **Unique Patients:** 1,042
* **Patients with $\ge 1$ Malignant Lesion:** 259 out of 1042 (24.86%)
* **Lesions per Patient Distribution:**
  * Minimum: 1
  * Median: 241.5
  * Mean: 384.9
  * Maximum: 9,184
* **Target Distribution:**
  * Benign (`target=0`): 400,666 (99.902%)
  * Malignant (`target=1`): 393 (0.0980%)
  * Binary Target Check: Verified strictly [0, 1], 0 nulls.

### Columns with Missing Values
Only 10 out of 55 columns contain missing values:

| Column | Missing Count | Percent Missing |
| :--- | :--- | :--- |
| `age_approx` | 2,798 | 0.70% |
| `sex` | 11,517 | 2.87% |
| `anatom_site_general` | 5,756 | 1.44% |
| `lesion_id` | 379,001 | 94.50% |
| `iddx_2` | 399,991 | 99.73% |
| `iddx_3` | 399,994 | 99.73% |
| `iddx_4` | 400,508 | 99.86% |
| `iddx_5` | 401,058 | 100.00% |
| `mel_mitotic_index` | 401,006 | 99.99% |
| `mel_thick_mm` | 400,996 | 99.98% |

---

## 3. Data Leakage & Feature Safeguards

### Excluded Columns (Strict Leakage & Non-Predictive Guardrails)
1. **Identifiers (Strictly Join & Grouping Keys):**
   - `isic_id`: Join key between metadata and HDF5 images.
   - `patient_id`: Grouping key for patient-level cross-validation; never fed as an input feature.
   - `lesion_id`: Internal lesion tracker ID (94.5% missing).
2. **Direct Data Leakage (Ground Truth / Post-Biopsy):**
- `iddx_full`: Flagged as data leakage / post-biopsy ground truth
- `iddx_1`: Flagged as data leakage / post-biopsy ground truth
- `iddx_2`: Flagged as data leakage / post-biopsy ground truth
- `iddx_3`: Flagged as data leakage / post-biopsy ground truth
- `iddx_4`: Flagged as data leakage / post-biopsy ground truth
- `iddx_5`: Flagged as data leakage / post-biopsy ground truth
- `mel_mitotic_index`: Flagged as data leakage / post-biopsy ground truth
- `mel_thick_mm`: Flagged as data leakage / post-biopsy ground truth
3. **Non-Predictive Administrative Metadata:**
   - `attribution`: Contributing hospital / institution name.
   - `copyright_license`: Licensing tag.
4. **Zero-Variance Constant Feature:**
   - `image_type`: 100% constant ('TBP tile: close-up').

### Candidate Predictor Features (40 features for Phase 3 review):
  - `age_approx`
  - `sex`
  - `anatom_site_general`
  - `clin_size_long_diam_mm`
  - `tbp_tile_type`
  - `tbp_lv_location`
  - `tbp_lv_location_simple`
  - `tbp_lv_x`
  - `tbp_lv_y`
  - `tbp_lv_z`
  - `tbp_lv_A`
  - `tbp_lv_Aext`
  - `tbp_lv_B`
  - `tbp_lv_Bext`
  - `tbp_lv_C`
  - `tbp_lv_Cext`
  - `tbp_lv_H`
  - `tbp_lv_Hext`
  - `tbp_lv_L`
  - `tbp_lv_Lext`
  - `tbp_lv_deltaA`
  - `tbp_lv_deltaB`
  - `tbp_lv_deltaL`
  - `tbp_lv_deltaLB`
  - `tbp_lv_deltaLBnorm`
  - `tbp_lv_areaMM2`
  - `tbp_lv_area_perim_ratio`
  - `tbp_lv_perimeterMM`
  - `tbp_lv_minorAxisMM`
  - `tbp_lv_eccentricity`
  - `tbp_lv_symm_2axis`
  - `tbp_lv_symm_2axis_angle`
  - `tbp_lv_norm_border`
  - `tbp_lv_norm_color`
  - `tbp_lv_color_std_mean`
  - `tbp_lv_radial_color_std_max`
  - `tbp_lv_stdL`
  - `tbp_lv_stdLExt`
  - `tbp_lv_nevi_confidence`
  - `tbp_lv_dnn_lesion_confidence`

> [!WARNING]
> `tbp_lv_dnn_lesion_confidence` and `tbp_lv_nevi_confidence` are automated scanner detector outputs. They will undergo specific importance and correlation checks in Phase 3 before final inclusion.

---

## 4. Patient-Safe Train, Validation & Test Splits

Using `StratifiedGroupKFold(n_splits=5, seed=42)`, Folds 0, 1, 2 are allocated to **Train** (60%), Fold 3 to **Validation** (20%), and Fold 4 to **Test** (20%):

| Split | Rows (% of Dataset) | Patients (% of Patients) | Malignant Cases | Benign Cases | Malignant Prevalence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | 240,638 (60.0%) | 625 (59.98%) | 239 | 240,399 | 0.0993% |
| **VAL** | 80,211 (20.0%) | 209 (20.06%) | 77 | 80,134 | 0.0960% |
| **TEST** | 80,210 (20.0%) | 208 (19.96%) | 77 | 80,133 | 0.0960% |
| **OVERALL** | 401,059 (100.0%) | 1042 (100.0%) | 393 | 400,666 | 0.0980% |

### Disjointness Verification
* **Patient Disjointness:** $Train \cap Val = \emptyset$, $Train \cap Test = \emptyset$, $Val \cap Test = \emptyset$. **0 patient leakage.**
* **Record Disjointness:** $Train \cap Val = \emptyset$, $Train \cap Test = \emptyset$, $Val \cap Test = \emptyset$. **0 record leakage.**
* **Positive Case Balance:** Every split contains abundant positive cases (Train: 239, Val: 77, Test: 77) with identical prevalence (~0.096% to 0.099%).

---

## 5. Local Split Manifest

* **Manifest File Location:** `/Users/chinmaysoni/Desktop/skin_cancer_detection/data/splits/patient_split_manifest.csv`
* **Schema:** `isic_id`, `patient_id`, `target`, `fold`, `split`
* **File Size:** ~16 MB CSV (kept local and ignored by `.gitignore`).
* **Official Challenge Holdout:** `test-metadata.csv` and `test-image.hdf5` were preserved untouched.
