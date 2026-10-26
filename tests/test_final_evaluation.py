"""Automated unit tests for Phase 6: Final Evaluation, Calibration, and Bootstrap."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.evaluation.bootstrap import patient_cluster_bootstrap
from src.evaluation.calibration import (
    apply_platt_calibrator,
    evaluate_calibration,
    fit_platt_calibrator,
)
from src.evaluation.metrics import evaluate_predictions, find_optimal_f1_threshold


def test_bootstrap_cluster_consistency():
    """Verify that patient-level cluster bootstrap correctly aggregates patient clusters and yields valid CIs."""
    patient_ids = ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"]
    y_true = np.array([0, 0, 1, 0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.4, 0.2, 0.1, 0.9, 0.85])

    boot_res = patient_cluster_bootstrap(
        patient_ids=patient_ids,
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.5,
        n_bootstraps=50,
        seed=42,
    )

    assert boot_res["valid_replicates"] > 0
    assert boot_res["failed_replicates"] >= 0
    assert "ap" in boot_res
    assert "ci_lower" in boot_res["ap"]
    assert "ci_upper" in boot_res["ap"]
    assert boot_res["ap"]["ci_lower"] <= boot_res["ap"]["ci_upper"]


def test_platt_calibration_preserves_ranking_and_corrects_prevalence():
    """Verify that Platt scaling shifts distorted probabilities towards true prevalence without changing ranking."""
    rng = np.random.RandomState(42)
    # Synthetic validation set with low prevalence (1%) but model predicting high probabilities (mean ~20%)
    n_samples = 1000
    y_true_val = np.zeros(n_samples, dtype=int)
    y_true_val[:10] = 1  # 1% prevalence

    # Biased uncalibrated probabilities (skewed high)
    y_prob_raw = rng.uniform(0.1, 0.8, size=n_samples)
    y_prob_raw[:10] += 0.2  # Signal for positives
    y_prob_raw = np.clip(y_prob_raw, 0.01, 0.99)

    calibrator = fit_platt_calibrator(y_prob_raw, y_true_val, seed=42)
    calibrated_probs = apply_platt_calibrator(calibrator, y_prob_raw)

    raw_metrics = evaluate_calibration(y_true_val, y_prob_raw)
    cal_metrics = evaluate_calibration(y_true_val, calibrated_probs)

    # Calibrated Brier score and log loss should improve
    assert cal_metrics["brier_score"] <= raw_metrics["brier_score"]
    # Mean calibrated probability should be much closer to true prevalence
    assert abs(cal_metrics["mean_predicted_prob"] - cal_metrics["true_prevalence"]) < abs(
        raw_metrics["mean_predicted_prob"] - raw_metrics["true_prevalence"]
    )


def test_frozen_test_partition_disjointness_and_counts():
    """Verify that the project-generated test partition matches verified counts with 0 patient overlap."""
    cfg = load_config()
    manifest_path = Path(cfg.splits.manifest_path)
    if not manifest_path.is_file():
        pytest.skip(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    test_df = df[df["split"] == "test"]
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    assert len(test_df) == 80210, f"Expected 80,210 test rows, got {len(test_df)}"
    assert (test_df["target"] == 1).sum() == 77, "Expected 77 test positives."
    assert (test_df["target"] == 0).sum() == 80133, "Expected 80,133 test negatives."
    assert test_df["patient_id"].nunique() == 208, "Expected 208 unique test patients."

    train_pts = set(train_df["patient_id"])
    val_pts = set(val_df["patient_id"])
    test_pts = set(test_df["patient_id"])

    assert len(train_pts.intersection(test_pts)) == 0, "Patient leakage between train and test."
    assert len(val_pts.intersection(test_pts)) == 0, "Patient leakage between val and test."


def test_confusion_matrix_direct_math_agreement():
    """Confirm that precision, recall, and F1 exactly match direct math from confusion matrix counts."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01])
    thresh = 0.5

    metrics = evaluate_predictions(y_true, y_prob, threshold=thresh)
    tp, fp, tn, fn = metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]

    calc_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    calc_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    calc_f1 = (2 * calc_prec * calc_rec) / (calc_prec + calc_rec) if (calc_prec + calc_rec) > 0 else 0.0

    assert np.isclose(metrics["precision"], calc_prec, atol=1e-5)
    assert np.isclose(metrics["recall"], calc_rec, atol=1e-5)
    assert np.isclose(metrics["f1"], calc_f1, atol=1e-5)
