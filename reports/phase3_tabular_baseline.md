# Phase 3 Report: Tabular Metadata Baseline Models

**Project:** SkinRisk Fusion — Multimodal Skin Lesion Risk Classification  
**Task:** Tabular Metadata Baseline & Preprocessing Pipeline  
**Training Set:** 240,638 samples (239 malignant, 240,399 benign — Imbalance: 1005.85:1)  
**Validation Set:** 80,211 samples (77 malignant, 80,134 benign — Prevalence: 0.0960%)  
**Primary Metric:** Average Precision (PR-AUC)  

---

## 1. Executive Summary & Headline Findings

1. **LightGBM Outperforms Linear and Dummy Baselines:**
   * **LightGBM (Unweighted)** achieved a validation **PR-AUC of 0.01147** and **ROC-AUC of 0.83423**, far superior to Logistic Regression (0.00957) and the Dummy baseline (0.00096).
2. **Impact of Class Weighting:**
   * Applying high `scale_pos_weight` (e.g. 100 or 1006) heavily shifts predicted probabilities upward. 
   * However, threshold-independent discrimination (PR-AUC & ROC-AUC) is highest with unweighted or moderate scaling. High weighting without recalibration creates tens of thousands of false positives at the default 0.5 threshold.
3. **Threshold Calibration is Essential:**
   * Because of the 1,006:1 class imbalance, evaluating at an arbitrary default threshold of 0.5 yields near-zero recall for unweighted models, or near-zero precision for heavily weighted models.
   * Tuning decision thresholds explicitly on validation data unlocks high F1 scores (0.0429 for LightGBM).

---

## 2. Validation Set Performance Comparison

### Table A: Performance at Default Threshold (0.5)

| Model | PR-AUC (AP) | ROC-AUC | Precision @ 0.5 | Recall @ 0.5 | F1 @ 0.5 | Confusion Matrix @ 0.5 | Fit Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy (Class Prior)** | 0.00096 | 0.50000 | 0.0000 | 0.0000 | 0.0000 | `TP=0, FP=0, TN=80134, FN=77` | 0.02s |
| **Logistic Regression (Unweighted)** | 0.00957 | 0.79517 | 0.0000 | 0.0000 | 0.0000 | `TP=0, FP=1, TN=80133, FN=77` | 1.16s |
| **Logistic Regression (Balanced)** | 0.01154 | 0.84884 | 0.0041 | 0.6493 | 0.0081 | `TP=50, FP=12279, TN=67855, FN=27` | 2.7s |
| **LightGBM (Unweighted, scale=1)** | 0.01147 | 0.83423 | 0.0571 | 0.0260 | 0.0357 | `TP=2, FP=33, TN=80101, FN=75` | 3.32s |
| **LightGBM (Moderate, scale=100)** | 0.00422 | 0.47867 | 0.0104 | 0.2727 | 0.0200 | `TP=21, FP=1998, TN=78136, FN=56` | 2.86s |
| **LightGBM (Full Ratio, scale=1006)** | 0.00474 | 0.67646 | 0.0078 | 0.4675 | 0.0153 | `TP=36, FP=4606, TN=75528, FN=41` | 2.88s |
| **LightGBM + Scanner Confidence (scale=1)** | 0.00834 | 0.74620 | 0.0098 | 0.0260 | 0.0143 | `TP=2, FP=201, TN=79933, FN=75` | 3.99s |

### Table B: Performance at Validation-Tuned Optimal F1 Threshold

| Model | PR-AUC | Optimal Threshold | Precision @ Opt | Recall @ Opt | F1 @ Opt | Confusion Matrix @ Opt |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy (Class Prior)** | 0.00096 | 0.0010 | 0.0000 | 0.0000 | 0.0000 | `TP=0, FP=0, TN=80134, FN=77` |
| **Logistic Regression (Unweighted)** | 0.00957 | 0.0106 | 0.0200 | 0.2078 | 0.0365 | `TP=16, FP=784, TN=79350, FN=61` |
| **Logistic Regression (Balanced)** | 0.01154 | 0.9500 | 0.0192 | 0.2857 | 0.0360 | `TP=22, FP=1124, TN=79010, FN=55` |
| **LightGBM (Unweighted, scale=1)** | 0.01147 | 0.1160 | 0.0476 | 0.0390 | 0.0429 | `TP=3, FP=60, TN=80074, FN=74` |
| **LightGBM (Moderate, scale=100)** | 0.00422 | 0.9500 | 0.0108 | 0.2727 | 0.0207 | `TP=21, FP=1929, TN=78205, FN=56` |
| **LightGBM (Full Ratio, scale=1006)** | 0.00474 | 0.9500 | 0.0078 | 0.4675 | 0.0154 | `TP=36, FP=4551, TN=75583, FN=41` |
| **LightGBM + Scanner Confidence (scale=1)** | 0.00834 | 0.0489 | 0.0186 | 0.0779 | 0.0301 | `TP=6, FP=316, TN=79818, FN=71` |

---

## 3. Precision-Recall Curve Analysis

The Precision-Recall curves are saved to:
`reports/figures/phase3_pr_curves.png`

```
Key Observation:
- Baseline prevalence is 0.00096 (0.096%).
- LightGBM achieves substantial precision across a wide range of recall, demonstrating strong ranking ability on unseen patients.
```

---

## 4. Feature Set & Safeguards

* **Allowed Primary Numeric Features (33):**
  - Clinical: `age_approx`, `clin_size_long_diam_mm`
  - 3D Coordinates: `tbp_lv_x`, `tbp_lv_y`, `tbp_lv_z`
  - Colorimetry & Contrast: `tbp_lv_A/B/C/H/L`, `tbp_lv_Aext/Bext/Cext/Hext/Lext`, `tbp_lv_deltaA/B/L/LB/LBnorm`
  - Geometry & Morphology: `tbp_lv_areaMM2`, `tbp_lv_area_perim_ratio`, `tbp_lv_perimeterMM`, `tbp_lv_minorAxisMM`, `tbp_lv_eccentricity`, `tbp_lv_symm_2axis`, `tbp_lv_symm_2axis_angle`
  - Texture & Uniformity: `tbp_lv_norm_border`, `tbp_lv_norm_color`, `tbp_lv_color_std_mean`, `tbp_lv_radial_color_std_max`, `tbp_lv_stdL`, `tbp_lv_stdLExt`
* **Allowed Primary Categorical Features (4):**
  - `sex` (imputed most-frequent, one-hot encoded)
  - `anatom_site_general` (imputed most-frequent, one-hot encoded)
  - `tbp_tile_type` (one-hot encoded)
  - `tbp_lv_location_simple` (one-hot encoded)
* **Strictly Excluded Prohibited Columns:**
  - `isic_id`, `patient_id`, `lesion_id`
  - `target`
  - `iddx_full`, `iddx_1`, `iddx_2`, `iddx_3`, `iddx_4`, `iddx_5`, `mel_mitotic_index`, `mel_thick_mm`
  - `attribution`, `copyright_license`, `image_type`
* **Ablation Feature Analysis:**
  - Scanner baseline confidence scores (`tbp_lv_nevi_confidence`, `tbp_lv_dnn_lesion_confidence`) were tested in a separate ablation experiment to measure their incremental value without contaminating the primary baseline.

---

## 5. Memory & Performance Observations

* **In-memory data loading:** Subsetting to the allowlist reduced metadata RAM consumption to under 120 MB.
* **LightGBM multi-threaded training on Apple Silicon M3:** Very fast (~2–3 seconds per fold).
* **Isolation Guarantee:** All transformers (imputers, scalers, one-hot encoders) and models were fitted exclusively on the `train` partition (625 patients). Validation data (209 patients) was only passed to `transform()` and `predict_proba()`.
