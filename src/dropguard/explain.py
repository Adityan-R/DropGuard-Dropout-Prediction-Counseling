"""
explain.py - SHAP-based model explanation utilities for DropGuard.

Provides functions to compute SHAP values, generate summary plots,
and extract per-student feature importance information.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_classifier(pipeline: Pipeline):
    """Return the final estimator from a sklearn Pipeline."""
    return pipeline[-1]


def _transform_X(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Transform X through all pipeline steps except the last (classifier)."""
    # Build a sub-pipeline of all steps except the classifier
    if len(pipeline.steps) > 1:
        preprocessor = Pipeline(pipeline.steps[:-1])
        X_transformed = preprocessor.transform(X)
    else:
        X_transformed = X.values
    # Ensure dense array
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    return np.asarray(X_transformed, dtype=float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values for the classifier in *pipeline*.

    Transforms *X* through the preprocessor steps first, then selects
    an appropriate SHAP explainer:

    * ``TreeExplainer`` for Random Forest / XGBoost estimators.
    * ``LinearExplainer`` for Logistic Regression estimators.

    Parameters
    ----------
    pipeline:
        A fitted sklearn :class:`~sklearn.pipeline.Pipeline` whose last step
        is the classifier.
    X:
        Raw (un-transformed) feature DataFrame.
    feature_names:
        List of feature names corresponding to the *transformed* feature
        space (after the preprocessor).

    Returns
    -------
    shap_values_class1 : np.ndarray, shape (n_samples, n_features)
        SHAP values for the positive class (at-risk = 1).
    expected_value : np.ndarray, shape (1,) or scalar
        The base/expected value from the explainer (for class 1).
    """
    clf = _get_classifier(pipeline)
    X_transformed = _transform_X(pipeline, X)

    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, X_transformed, feature_perturbation="interventional")
        shap_explanation = explainer(X_transformed)
        shap_values = shap_explanation.values
        expected_value = explainer.expected_value
    else:
        # TreeExplainer works for RandomForest, XGBoost, GradientBoosting, etc.
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)
        expected_value = explainer.expected_value

    # SHAP returns per-class values for binary classifiers, either as a list of
    # arrays (older versions) or as an (n_samples, n_features, n_classes)
    # array (newer versions). Keep the positive class (at-risk = 1) either way.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., -1]

    expected_arr = np.atleast_1d(np.asarray(expected_value, dtype=float))
    if expected_arr.size > 1:
        expected_arr = expected_arr[-1:]
    return shap_values, float(expected_arr.ravel()[0])


def plot_shap_summary(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list[str],
    save_path: Path,
) -> None:
    """Create and save a SHAP beeswarm summary plot (top 15 features).

    Parameters
    ----------
    shap_values:
        SHAP values array of shape ``(n_samples, n_features)``.
    X_transformed:
        Transformed feature matrix of shape ``(n_samples, n_features)``.
    feature_names:
        Feature names for the x-axis labels.
    save_path:
        Destination ``Path`` for the saved ``.png`` figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Trim to top-15 features by mean |SHAP|
    n_features = min(15, shap_values.shape[1])
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:n_features]

    shap_top = shap_values[:, top_idx]
    X_top = X_transformed[:, top_idx]
    names_top = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_top,
        X_top,
        feature_names=names_top,
        show=False,
        plot_type="dot",
        max_display=n_features,
    )
    plt.title("SHAP Feature Importance - Top 15 Drivers", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def get_top_drivers(
    shap_values_row: np.ndarray,
    feature_names: list[str],
    top_n: int = 3,
) -> list[tuple[str, float]]:
    """Return the *top_n* ``(feature_name, shap_value)`` pairs for one student.

    Pairs are sorted by absolute SHAP value in descending order so the most
    influential features appear first.

    Parameters
    ----------
    shap_values_row:
        1-D array of SHAP values for a single student.
    feature_names:
        Feature names aligned with *shap_values_row*.
    top_n:
        Number of top drivers to return (default 3).

    Returns
    -------
    list[tuple[str, float]]
        Each tuple is ``(feature_name, shap_value)``.
    """
    shap_values_row = np.asarray(shap_values_row, dtype=float).ravel()
    feature_names = list(feature_names)
    sorted_idx = np.argsort(np.abs(shap_values_row))[::-1][:top_n]
    return [(feature_names[int(i)], float(shap_values_row[int(i)])) for i in sorted_idx]


def explain_students(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    feature_names: list[str],
    config: dict,
) -> list[dict]:
    """Compute SHAP explanations for every student in *X_test*.

    Parameters
    ----------
    pipeline:
        Fitted sklearn Pipeline.
    X_test:
        Raw feature DataFrame for the test set.
    feature_names:
        Feature names for the transformed feature space.
    config:
        Project config dict (reserved for future use / logging).

    Returns
    -------
    list[dict]
        One dict per student with keys:

        * ``student_idx`` - original row index from *X_test*.
        * ``top_drivers`` - list of ``(feature_name, shap_value)`` tuples.
        * ``shap_values`` - full SHAP row array.
        * ``expected_value`` - scalar base value from the explainer.
    """
    shap_values, expected_value = compute_shap_values(pipeline, X_test, feature_names)

    results: list[dict] = []
    for local_i, student_idx in enumerate(X_test.index):
        row_shap = shap_values[local_i]
        top_drivers = get_top_drivers(row_shap, feature_names, top_n=3)
        results.append(
            {
                "student_idx": student_idx,
                "top_drivers": top_drivers,
                "shap_values": row_shap,
                "expected_value": float(expected_value),
            }
        )
    return results
