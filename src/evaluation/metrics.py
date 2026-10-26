"""Evaluation metrics, threshold analysis, and curve plotting for SkinRisk Fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate discrimination metrics and threshold-dependent classification metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # 1. Discrimination metrics (threshold-independent)
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    # 2. Threshold-dependent metrics
    y_pred = (y_prob >= threshold).astype(int)

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "pr_auc": round(pr_auc, 5),
        "roc_auc": round(roc_auc, 5),
        "threshold": threshold,
        "precision": round(prec, 5),
        "recall": round(rec, 5),
        "f1": round(f1, 5),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "pos_count": int(y_true.sum()),
        "neg_count": int((y_true == 0).sum()),
    }


def find_optimal_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_thresholds: int = 100,
) -> Dict[str, Any]:
    """Sweep thresholds on validation data to identify the threshold maximizing F1 score."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    thresholds = np.linspace(0.001, 0.95, n_thresholds)
    best_thresh = 0.5
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)
            best_prec = float(precision_score(y_true, y_pred, zero_division=0))
            best_rec = float(recall_score(y_true, y_pred, zero_division=0))

    return {
        "optimal_threshold": round(best_thresh, 4),
        "f1_at_optimal": round(best_f1, 5),
        "precision_at_optimal": round(best_prec, 5),
        "recall_at_optimal": round(best_rec, 5),
    }


def plot_precision_recall_curves(
    model_probabilities: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_path: Path | str,
) -> None:
    """Plot Precision-Recall curves for all evaluated models and save to disk."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).astype(int)
    prevalence = float(y_true.sum() / len(y_true))

    plt.figure(figsize=(9, 6), dpi=150)

    for name, probs in model_probabilities.items():
        prec, rec, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        plt.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {ap:.4f})")

    # Baseline random prevalence line
    plt.axhline(
        y=prevalence,
        color="grey",
        linestyle="--",
        label=f"Random Prevalence Baseline ({prevalence:.4f})",
    )

    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.title("Precision-Recall Curves — Validation Set (SkinRisk Fusion)", fontsize=13, pad=12)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
