"""Probability calibration utilities, Platt scaling, and reliability analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logit, expit
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def fit_platt_calibrator(
    y_prob_val: np.ndarray,
    y_true_val: np.ndarray,
    seed: int = 42,
) -> LogisticRegression:
    """Fit a univariate Platt scaling (sigmoid) calibrator on validation data ONLY.

    Transforms raw model output probabilities through inverse-sigmoid (logits) and
    fits a standard logistic regression model to recalibrate probabilities to
    the actual validation prevalence without altering model ranking.
    """
    y_true_val = np.asarray(y_true_val).astype(int)
    y_prob_val = np.asarray(y_prob_val).astype(float)

    # Numerical clip to avoid infinite logits
    eps = 1e-7
    y_prob_clipped = np.clip(y_prob_val, eps, 1.0 - eps)
    logits_val = logit(y_prob_clipped).reshape(-1, 1)

    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=seed)
    calibrator.fit(logits_val, y_true_val)
    return calibrator


def apply_platt_calibrator(
    calibrator: LogisticRegression,
    y_prob: np.ndarray,
) -> np.ndarray:
    """Apply the fitted Platt calibrator to generate calibrated probabilities."""
    y_prob = np.asarray(y_prob).astype(float)
    eps = 1e-7
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    logits = logit(y_prob_clipped).reshape(-1, 1)
    calibrated_probs = calibrator.predict_proba(logits)[:, 1]
    return calibrated_probs


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Compute calibration metrics: Brier score, log loss, and prevalence gap."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    eps = 1e-7
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)

    brier = float(brier_score_loss(y_true, y_prob))
    loss = float(log_loss(y_true, y_prob_clipped))
    mean_prob = float(np.mean(y_prob))
    true_prev = float(np.mean(y_true))
    prev_ratio = float(mean_prob / max(1e-9, true_prev))

    return {
        "brier_score": round(brier, 6),
        "log_loss": round(loss, 6),
        "mean_predicted_prob": round(mean_prob, 6),
        "true_prevalence": round(true_prev, 6),
        "prevalence_distortion_ratio": round(prev_ratio, 3),
    }


def plot_calibration_curves(
    model_probabilities: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_path: Path | str,
    n_bins: int = 10,
) -> None:
    """Plot reliability calibration curves comparing raw and calibrated predictions."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).astype(int)

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)")

    for name, probs in model_probabilities.items():
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="quantile")
        plt.plot(prob_pred, prob_true, marker="o", lw=2, label=name)

    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of Positives (Empirical Risk)", fontsize=11)
    plt.title("Reliability Calibration Curves — Test Partition (SkinRisk Fusion)", fontsize=12, pad=12)
    plt.legend(loc="upper left", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
