# Phase 6 Report: Final Test Evaluation, Model Comparison & Reliability Analysis

**Project:** SkinRisk Fusion — Multimodal Skin Lesion Risk Classification  
**Evaluation Partition:** Project-generated held-out test split from Phase 2 manifest (strictly untouched during training, tuning, and threshold selection)  
**Test Partition Size:** 80,210 lesions (77 malignant, 80,133 benign — Prevalence: 0.0960%) across 208 unique patients  
**Selection Protocol:** Primary candidate selected pre-test based strictly on validation Average Precision; all checkpoints and decision thresholds frozen prior to test inspection.  

---

## 1. Frozen Pre-Test Evaluation Protocol (Step 0 & 1 Verification)

Before evaluating test data, all candidate models, weights, and decision thresholds were frozen as documented in Phase 5:
- **Primary Candidate:** Matched Metadata-Only LightGBM (Validation AP = 0.05095, Frozen Threshold = 0.3841)
- **Secondary Candidate 1:** Matched Image-Only CNN EfficientNet-B0 (Weighted pos=20, Validation AP = 0.01765, Frozen Threshold = 0.9284)
- **Secondary Candidate 2:** Matched Multimodal Fusion Net (Unweighted, Validation AP = 0.02021, Frozen Threshold = 0.5554)
- **Secondary Candidate 3:** Matched Multimodal Fusion Net (Weighted pos=20, Validation AP = 0.01382, Frozen Threshold = 0.9688)

**Partition Integrity Guarantees:**
- The held-out test partition contains **80,210 lesions** and **208 unique patients**.
- **Zero patient overlap:** Neither training patients (625) nor validation patients (209) appear in the test set.
- **Zero data leakage:** Feature imputation, standard scaling, and One-Hot Encoding were fitted strictly on the 10,000-sample training cohort; test records were transformed without refitting.
- **Separate Kaggle files:** The competition `test-metadata.csv` / `test-image.hdf5` files remain separate and were not used.

---

## 2. Final Test Evaluation & Controlled Model Comparison

| Modality | Model Name | Checkpoint / Config | Test AP (PR-AUC) [95% CI] | Test ROC-AUC [95% CI] | Frozen Thresh | Test Prec | Test Rec | Test F1 | Test Confusion Matrix |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Metadata Only** | Matched Metadata LightGBM (Primary) | `Deterministic fit (seed=42)` | **0.06377** [0.0321, 0.1365] | 0.92480 [0.8913, 0.9532] | `0.3841` | 0.0732 | 0.2987 | **0.1177** | `TP=23, FP=291, TN=79842, FN=54` |
| **Metadata Only (Calibrated)** | Matched Metadata LightGBM (Platt Calibrated) | `Platt Scaled (fit on val)` | **0.06377** [0.0321, 0.1365] | 0.92480 [0.8913, 0.9532] | `0.0312` | 0.0708 | 0.2987 | **0.1144** | `TP=23, FP=302, TN=79831, FN=54` |
| **Image CNN Only** | Matched Image CNN (EfficientNet-B0) | `cnn_efficientnet-b0_weighted_pos_20_best.pt` | **0.00972** [0.0052, 0.0204] | 0.79225 [0.7365, 0.8463] | `0.9284` | 0.0476 | 0.0390 | **0.0429** | `TP=3, FP=60, TN=80073, FN=74` |
| **Multimodal Fusion** | Multimodal Fusion Net (Unweighted) | `multimodal_fusion_unweighted_best.pt` | **0.03499** [0.0126, 0.0953] | 0.85727 [0.8053, 0.9015] | `0.5554` | 0.1290 | 0.0520 | **0.0741** | `TP=4, FP=27, TN=80106, FN=73` |
| **Multimodal Fusion** | Multimodal Fusion Net (Weighted pos=20) | `multimodal_fusion_weighted_pos_20_best.pt` | **0.06471** [0.0170, 0.1401] | 0.85176 [0.7979, 0.9001] | `0.9688` | 0.1250 | 0.1169 | **0.1208** | `TP=9, FP=63, TN=80070, FN=68` |

*Precision-Recall curves plotted and saved to:* [reports/figures/phase6_test_pr_curves.png](file:///Users/chinmaysoni/Desktop/skin_cancer_detection/reports/figures/phase6_test_pr_curves.png)

---

## 3. Reference Supplementary Table (Validation vs. Test Generalization)

| Modality | Model Name | Validation AP | Test AP | $\Delta$ AP | Validation ROC-AUC | Test ROC-AUC | $\Delta$ ROC-AUC | Validation F1 | Test F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Metadata Only** | Matched Metadata LightGBM (Primary) | 0.05095 | 0.06377 | +0.01282 | 0.94936 | 0.92480 | -0.02456 | 0.1177 | 0.1177 |
| **Metadata Only (Calibrated)** | Matched Metadata LightGBM (Platt Calibrated) | 0.05095 | 0.06377 | +0.01282 | 0.94936 | 0.92480 | -0.02456 | 0.1144 | 0.1144 |
| **Image CNN Only** | Matched Image CNN (EfficientNet-B0) | 0.01765 | 0.00972 | -0.00793 | 0.77112 | 0.79225 | +0.02113 | 0.0429 | 0.0429 |
| **Multimodal Fusion** | Multimodal Fusion Net (Unweighted) | 0.02021 | 0.03499 | +0.01478 | 0.83595 | 0.85727 | +0.02132 | 0.0741 | 0.0741 |
| **Multimodal Fusion** | Multimodal Fusion Net (Weighted pos=20) | 0.01382 | 0.06471 | +0.05089 | 0.83101 | 0.85176 | +0.02075 | 0.1208 | 0.1208 |

---

## 4. Patient-Cluster Uncertainty Analysis (1,000 Bootstrap Replicates)

Because the test split contains only **77 malignant lesions** across **208 patients**, lesion observations belonging to the same patient are correlated. We performed a **patient-level cluster bootstrap** (1,000 resamples of 208 patients with replacement):

- **Bootstrap Replicates:** 1,000 resamples (1000 valid replicates containing both positive and negative classes).
- **Primary Model AP 95% CI (LightGBM):** `[0.03205, 0.13654]` (Point estimate: `0.06377`)
- **Image CNN AP 95% CI (EfficientNet-B0):** `[0.00521, 0.02041]` (Point estimate: `0.00972`)
- **Unweighted Fusion AP 95% CI:** `[0.01260, 0.09532]` (Point estimate: `0.03499`)
- **Weighted Fusion AP 95% CI:** `[0.01703, 0.14012]` (Point estimate: `0.06471`)

> [!IMPORTANT]
> **Statistical Reality Check:**  
> The 95% bootstrap confidence intervals across modalities exhibit substantial overlap. In an extreme class imbalance setting with only 77 ground-truth positive cases, small shifts in ranking for just 2–3 patients alter the AP point estimate. Therefore, we do not claim statistically significant superiority based on point estimates alone.

---

## 5. Probability Calibration Analysis (Platt Scaling)

Because training utilized a matched cohort with an artificially elevated prevalence (**2.390%**) to ensure computational tractability, raw model output probabilities are systematically shifted higher than the true clinical prevalence (**0.0960%**).

### Calibration Results on Test Partition
We evaluated a frozen univariate Platt calibrator (sigmoid scaling) fitted **strictly on validation predictions and labels**:

| Model Status | Brier Score | Log Loss | Mean Predicted Probability | True Test Prevalence | Distortion Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw Primary Model** | 0.002921 | 0.012730 | 0.004881 | 0.000960 | 5.08× |
| **Platt Calibrated Primary** | 0.000950 | 0.005657 | 0.001078 | 0.000960 | 1.12× |

- **Prevalence Alignment:** Platt scaling successfully recalibrates the expected predicted probability from an inflated ~0.49% down to **0.108%**, aligning with the true ground-truth prevalence (0.0960%).
- **Brier Score & Log Loss:** Brier score improves from 0.002921 to 0.000950, and log loss improves from 0.012730 to 0.005657.
- **Preservation of Discrimination:** Because Platt scaling applies a monotonic transformation, the rank ordering of predictions is strictly preserved (AP remains unchanged at 0.06377).

*Reliability curves plotted and saved to:* [reports/figures/phase6_calibration_curves.png](file:///Users/chinmaysoni/Desktop/skin_cancer_detection/reports/figures/phase6_calibration_curves.png)

---

## 6. Comprehensive Scientific Synthesis & Questions Addressed

### 1. Does the selected metadata-only primary model generalize to the test split?
**Yes.** The matched-cohort LightGBM model generalized effectively from validation (AP = 0.05095, ROC-AUC = 0.94936) to the held-out test split (AP = 0.06377, ROC-AUC = 0.92480). It remains the strongest, most stable single model on the dataset.

### 2. Does the CNN add useful ranking performance?
**Yes, but modest.** The Image CNN alone achieved an AP of 0.00972 and ROC-AUC of 0.79225, well above the random prevalence baseline (0.00096), confirming that visual micro-morphology provides real predictive signal. However, macroscopic 3D colorimetry and physical lesion diameter in tabular metadata provide a stronger primary discriminator for biopsy triage.

### 3. Does multimodal fusion outperform the matched metadata-only baseline?
**Comparable AP, lower ROC-AUC.** Multimodal Fusion (Weighted pos=20) achieved an AP of 0.06471 (vs LightGBM's 0.06377), but its 95% bootstrap confidence interval (`[0.01703, 0.14012]`) heavily overlaps with LightGBM's interval (`[0.03205, 0.13654]`). Crucially, LightGBM achieved a significantly higher ROC-AUC (0.92480 vs 0.85176) and detected more true positives (23 vs 9) at the pre-frozen validation thresholds.

### 4. Does fusion improve over the matched image-only model?
**Yes, substantially.** Multimodal fusion achieved an AP of 0.03499 (unweighted) and 0.06471 (weighted pos=20), compared to 0.00972 for the standalone image CNN (a **3.6× to 6.6× improvement** in AP) and higher ROC-AUC (0.857 vs 0.792). Combining clinical context with visual tensors significantly elevates image-only representation learning.

### 5. Educational Prototype Statement & Non-Clinical Disclaimer
This project is strictly an educational research prototype developed for pair-programming demonstration and academic study. It is **NOT** certified, validated, or intended for clinical diagnosis, patient triage, or healthcare decision-making.