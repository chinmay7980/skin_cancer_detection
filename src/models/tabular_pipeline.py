"""Preprocessing pipeline and tabular model factories for SkinRisk Fusion."""

from __future__ import annotations

from typing import List, Optional, Any
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb


def create_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes numerics and one-hot encodes categoricals.

    Numeric strategy: median imputation (+ optional standard scaling).
    Categorical strategy: most-frequent imputation + OneHotEncoder with handle_unknown='ignore'.
    """
    if scale_numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",  # Drop any column not explicitly allowlisted
    )
    return preprocessor


def create_dummy_baseline(seed: int = 42) -> DummyClassifier:
    """Baseline predicting class prior probability."""
    return DummyClassifier(strategy="prior", random_state=seed)


def create_logistic_regression_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    class_weight: Optional[Any] = None,
    C: float = 1.0,
    seed: int = 42,
) -> Pipeline:
    """Construct an end-to-end Logistic Regression pipeline."""
    preprocessor = create_preprocessor(numeric_features, categorical_features, scale_numeric=True)
    clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])


def create_lightgbm_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_pos_weight: float = 1.0,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    seed: int = 42,
    **extra_params: Any,
) -> Pipeline:
    """Construct an end-to-end LightGBM pipeline with preprocessor."""
    preprocessor = create_preprocessor(numeric_features, categorical_features, scale_numeric=False)
    clf = lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
        **extra_params,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])
