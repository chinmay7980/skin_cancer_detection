"""Automated unit tests for tabular preprocessing, feature validation, and baseline models."""

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    PROHIBITED_COLUMNS,
    get_feature_lists,
    validate_feature_allowlist,
)
from src.evaluation.metrics import evaluate_predictions, find_optimal_f1_threshold
from src.models.tabular_pipeline import (
    create_dummy_baseline,
    create_lightgbm_pipeline,
    create_logistic_regression_pipeline,
    create_preprocessor,
)


def test_feature_allowlist_guardrails():
    """Verify that validate_feature_allowlist strictly catches all prohibited fields."""
    num_cols, cat_cols = get_feature_lists()

    # Base allowlist must pass validation
    validate_feature_allowlist(num_cols + cat_cols)

    # Injected prohibited fields must raise ValueError
    for prohibited in PROHIBITED_COLUMNS:
        with pytest.raises(ValueError, match="CRITICAL LEAKAGE ERROR"):
            validate_feature_allowlist(num_cols + cat_cols + [prohibited])


def test_preprocessor_isolation_and_unseen_categories():
    """Verify preprocessor fits only on train and handles unseen categories in validation."""
    train_df = pd.DataFrame({
        "age": [50.0, np.nan, 30.0, 40.0],
        "site": ["arm", "leg", "arm", "torso"],
    })
    val_df = pd.DataFrame({
        "age": [np.nan, 60.0],
        "site": ["head", "leg"],  # 'head' is an unseen category in training!
    })

    preprocessor = create_preprocessor(["age"], ["site"], scale_numeric=True)

    # Fit on train
    preprocessor.fit(train_df)
    train_features = preprocessor.transform(train_df)

    # Record fitted imputer median
    num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    fitted_median = num_imputer.statistics_[0]
    assert fitted_median == 40.0, f"Expected median 40.0, got {fitted_median}"

    # Transform validation without refitting
    val_features = preprocessor.transform(val_df)

    # Assert shape (1 numeric + 3 one-hot columns from train categories)
    assert val_features.shape == (2, 4)
    # The unseen 'head' category should have zero across all one-hot columns
    assert (val_features[0, 1:] == 0).all()
    # Imputer statistics must remain unchanged
    assert num_imputer.statistics_[0] == 40.0


def test_metrics_evaluation_properties():
    """Verify evaluate_predictions outputs valid mathematical ranges and proper confusion matrix sums."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.8, 0.9])

    metrics = evaluate_predictions(y_true, y_prob, threshold=0.5)

    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"] == len(y_true)
    assert metrics["pos_count"] == 2
    assert metrics["neg_count"] == 3

    opt = find_optimal_f1_threshold(y_true, y_prob)
    assert 0.0 <= opt["optimal_threshold"] <= 1.0
    assert 0.0 <= opt["f1_at_optimal"] <= 1.0


def test_toy_baseline_model_training():
    """Verify that Dummy, LogisticRegression, and LightGBM pipelines fit and predict valid probabilities."""
    X_toy = pd.DataFrame({
        "age_approx": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, np.nan],
        "clin_size_long_diam_mm": [2.0, 3.5, 4.0, 1.5, 5.0, 3.0, 4.5, 2.5],
        "sex": ["male", "female", "male", "female", "male", np.nan, "female", "male"],
        "tbp_tile_type": ["3D: XP", "3D: white", "3D: XP", "3D: XP", "3D: white", "3D: XP", "3D: white", "3D: XP"],
    })
    y_toy = np.array([0, 0, 0, 0, 0, 0, 1, 1])

    num_cols = ["age_approx", "clin_size_long_diam_mm"]
    cat_cols = ["sex", "tbp_tile_type"]

    # 1. Dummy
    dummy = create_dummy_baseline()
    dummy.fit(X_toy, y_toy)
    dummy_probs = dummy.predict_proba(X_toy)[:, 1]
    assert len(dummy_probs) == len(y_toy)
    assert np.allclose(dummy_probs, 2 / 8)

    # 2. Logistic Regression
    lr = create_logistic_regression_pipeline(num_cols, cat_cols, class_weight="balanced")
    lr.fit(X_toy, y_toy)
    lr_probs = lr.predict_proba(X_toy)[:, 1]
    assert len(lr_probs) == len(y_toy)
    assert ((lr_probs >= 0.0) & (lr_probs <= 1.0)).all()

    # 3. LightGBM
    lgbm = create_lightgbm_pipeline(num_cols, cat_cols, scale_pos_weight=3.0, n_estimators=10)
    lgbm.fit(X_toy, y_toy)
    lgbm_probs = lgbm.predict_proba(X_toy)[:, 1]
    assert len(lgbm_probs) == len(y_toy)
    assert ((lgbm_probs >= 0.0) & (lgbm_probs <= 1.0)).all()
