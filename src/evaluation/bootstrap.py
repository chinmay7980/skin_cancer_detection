"""Patient-level cluster bootstrap for evaluation metrics with correlated records."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def patient_cluster_bootstrap(
    patient_ids: List[str] | np.ndarray | pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Perform patient-level cluster bootstrapping to compute robust confidence intervals.

    Resamples unique patients with replacement rather than treating individual lesions
    as independent observations, preserving intra-patient lesion correlation.

    Args:
        patient_ids: Identifier per record indicating patient cluster.
        y_true: Binary ground-truth targets (0 or 1).
        y_prob: Predicted malignancy probabilities.
        threshold: Frozen decision threshold for classification metrics.
        n_bootstraps: Number of bootstrap iterations (e.g., 500-1000).
        seed: Random seed for reproducibility.
        alpha: Significance level (default 0.05 for 95% two-sided CI).

    Returns:
        Dictionary containing point estimates, 95% CIs, standard errors, and valid replicate counts.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    patient_series = pd.Series(patient_ids)

    unique_patients = patient_series.unique()
    n_unique_patients = len(unique_patients)

    # Pre-index record locations for each patient for fast vectorized gathering
    patient_to_indices = patient_series.groupby(patient_series).indices

    rng = np.random.RandomState(seed)

    ap_list: List[float] = []
    roc_auc_list: List[float] = []
    prec_list: List[float] = []
    rec_list: List[float] = []
    f1_list: List[float] = []

    valid_replicates = 0
    failed_replicates = 0

    lower_p = (alpha / 2.0) * 100
    upper_p = (1.0 - alpha / 2.0) * 100

    for _ in range(n_bootstraps):
        sampled_patients = rng.choice(unique_patients, size=n_unique_patients, replace=True)
        sampled_idx = np.concatenate([patient_to_indices[p] for p in sampled_patients])

        b_true = y_true[sampled_idx]
        b_prob = y_prob[sampled_idx]

        # Valid replicate requires both classes to be present
        if b_true.sum() == 0 or (b_true == 0).sum() == 0:
            failed_replicates += 1
            continue

        try:
            ap = float(average_precision_score(b_true, b_prob))
            roc = float(roc_auc_score(b_true, b_prob))

            b_pred = (b_prob >= threshold).astype(int)
            prec = float(precision_score(b_true, b_pred, zero_division=0))
            rec = float(recall_score(b_true, b_pred, zero_division=0))
            f1 = float(f1_score(b_true, b_pred, zero_division=0))

            ap_list.append(ap)
            roc_auc_list.append(roc)
            prec_list.append(prec)
            rec_list.append(rec)
            f1_list.append(f1)
            valid_replicates += 1
        except Exception:
            failed_replicates += 1

    if valid_replicates == 0:
        raise RuntimeError("Patient cluster bootstrap failed: 0 valid replicates.")

    def calc_stats(metric_values: List[float]) -> Dict[str, Any]:
        arr = np.array(metric_values)
        return {
            "mean": round(float(np.mean(arr)), 5),
            "std": round(float(np.std(arr)), 5),
            "ci_lower": round(float(np.percentile(arr, lower_p)), 5),
            "ci_upper": round(float(np.percentile(arr, upper_p)), 5),
        }

    return {
        "n_bootstraps": n_bootstraps,
        "valid_replicates": valid_replicates,
        "failed_replicates": failed_replicates,
        "confidence_level": f"{int((1 - alpha) * 100)}%",
        "threshold": threshold,
        "ap": calc_stats(ap_list),
        "roc_auc": calc_stats(roc_auc_list),
        "precision": calc_stats(prec_list),
        "recall": calc_stats(rec_list),
        "f1": calc_stats(f1_list),
    }
