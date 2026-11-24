"""
train.py - DropGuard model training module.

Loads config from config.yaml, compares Logistic Regression, Random Forest,
and XGBoost, tunes RF/XGB via GridSearchCV, selects best model by recall,
evaluates on a held-out test set, and persists model + metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.dropguard.data import (
    clean_data,
    create_labels,
    get_features_and_target,
    load_config,
    load_raw_data,
)
from src.dropguard.features import build_pipeline, get_feature_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root resolution (config.yaml lives here)
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the project root (two levels above this file: src/dropguard/train.py)."""
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models(config: dict) -> dict[str, Any]:
    """Return a dict of model_name -> unfitted sklearn estimator.

    Models returned:
    - 'logistic'      : LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed)
    - 'random_forest' : RandomForestClassifier(class_weight='balanced', random_state=seed)
    - 'xgboost'       : XGBClassifier(eval_metric='logloss', random_state=seed, verbosity=0)

    Parameters
    ----------
    config : dict
        Parsed config.yaml dictionary.

    Returns
    -------
    dict[str, Any]
        Mapping of model name to estimator instance.
    """
    seed: int = config.get("training", {}).get("random_seed", 42)
    lr_max_iter: int = config.get("models", {}).get("logistic", {}).get("max_iter", 1000)

    return {
        "logistic": LogisticRegression(
            class_weight="balanced",
            max_iter=lr_max_iter,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=seed,
        ),
        "xgboost": XGBClassifier(
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        ),
    }


# ---------------------------------------------------------------------------
# Hyper-parameter grids
# ---------------------------------------------------------------------------

def get_param_grids(config: dict) -> dict[str, dict]:
    """Return GridSearchCV parameter grids for each model.

    Keys are prefixed with ``'classifier__'`` to match the pipeline step name.
    Search spaces are drawn from ``config.yaml``.

    Parameters
    ----------
    config : dict
        Parsed config.yaml dictionary.

    Returns
    -------
    dict[str, dict]
        Mapping of model name to its hyper-parameter grid.
    """
    models_cfg = config.get("models", {})
    lr_cfg = models_cfg.get("logistic", {})
    rf_cfg = models_cfg.get("random_forest", {})
    xgb_cfg = models_cfg.get("xgboost", {})

    return {
        "logistic": {
            "classifier__C": lr_cfg.get("C", [0.01, 0.1, 1, 10]),
        },
        "random_forest": {
            "classifier__n_estimators": rf_cfg.get("n_estimators", [100, 200]),
            "classifier__max_depth": rf_cfg.get("max_depth", [None, 5, 10]),
            "classifier__min_samples_split": rf_cfg.get("min_samples_split", [2, 5]),
        },
        "xgboost": {
            "classifier__n_estimators": xgb_cfg.get("n_estimators", [100, 200]),
            "classifier__max_depth": xgb_cfg.get("max_depth", [3, 5, 7]),
            "classifier__learning_rate": xgb_cfg.get("learning_rate", [0.05, 0.1, 0.2]),
        },
    }


# ---------------------------------------------------------------------------
# Cross-validation helper
# ---------------------------------------------------------------------------

def cross_validate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> dict:
    """Run StratifiedKFold cross-validation and return aggregated metrics.

    Scoring metrics: accuracy, precision (macro), recall (macro), f1 (macro),
    roc_auc (macro ovr).

    Parameters
    ----------
    name : str
        Human-readable model name (used for logging).
    pipeline : Pipeline
        Fitted-ready sklearn Pipeline containing a preprocessor and classifier.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    config : dict
        Parsed config.yaml dictionary.

    Returns
    -------
    dict
        Keys: ``accuracy_mean``, ``accuracy_std``, ``precision_mean``,
        ``precision_std``, ``recall_mean``, ``recall_std``, ``f1_mean``,
        ``f1_std``, ``roc_auc_mean``, ``roc_auc_std``.
    """
    cv_folds: int = config.get("training", {}).get("cv_folds", 5)
    seed: int = config.get("training", {}).get("random_seed", 42)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    logger.info("Cross-validating %s (%d folds)...", name, cv_folds)
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    result: dict = {}
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        result[f"{metric}_mean"] = float(np.mean(scores))
        result[f"{metric}_std"] = float(np.std(scores))

    logger.info(
        "%s CV -> recall=%.4f±%.4f  roc_auc=%.4f±%.4f",
        name,
        result["recall_mean"],
        result["recall_std"],
        result["roc_auc_mean"],
        result["roc_auc_std"],
    )
    return result


# ---------------------------------------------------------------------------
# Hyper-parameter tuning
# ---------------------------------------------------------------------------

def tune_model(
    name: str,
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> Pipeline:
    """Tune a pipeline with GridSearchCV and return the best estimator.

    Uses StratifiedKFold with ``scoring='recall'`` and ``refit=True``.

    Parameters
    ----------
    name : str
        Human-readable model name (used for logging).
    pipeline : Pipeline
        Fitted-ready sklearn Pipeline.
    param_grid : dict
        Hyper-parameter grid (keys prefixed with ``'classifier__'``).
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    config : dict
        Parsed config.yaml dictionary.

    Returns
    -------
    Pipeline
        Best estimator (already refitted on the full training set).
    """
    cv_folds: int = config.get("training", {}).get("cv_folds", 5)
    seed: int = config.get("training", {}).get("random_seed", 42)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    logger.info("Tuning %s with GridSearchCV...", name)
    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=skf,
        scoring="recall",
        refit=True,
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    logger.info("%s best params: %s  best recall: %.4f", name, gs.best_params_, gs.best_score_)
    return gs.best_estimator_


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> tuple[pd.DataFrame, str, Pipeline]:
    """Compare Logistic Regression, Random Forest, and XGBoost.

    Strategy:
    - Logistic Regression: cross-validate only (no tuning).
    - Random Forest / XGBoost: tune with GridSearchCV, then cross-validate.

    The best model is the one with the highest mean recall (macro) on CV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    config : dict
        Parsed config.yaml dictionary.

    Returns
    -------
    tuple[pd.DataFrame, str, Pipeline]
        - comparison_df  : DataFrame indexed by model name with CV metric columns.
        - best_model_name: Name of the best performing model.
        - best_pipeline  : Fitted Pipeline for the best model.
    """
    models = get_models(config)
    param_grids = get_param_grids(config)

    rows: dict[str, dict] = {}
    pipelines: dict[str, Pipeline] = {}

    for model_name, estimator in models.items():
        pipeline = build_pipeline(estimator, X_train)

        if model_name == "logistic":
            # Cross-validate only; refit on full training set afterwards
            cv_metrics = cross_validate_model(model_name, pipeline, X_train, y_train, config)
            pipeline.fit(X_train, y_train)
            final_pipeline = pipeline
        else:
            # Tune first, which refits on full training data
            tuned = tune_model(
                model_name,
                pipeline,
                param_grids[model_name],
                X_train,
                y_train,
                config,
            )
            cv_metrics = cross_validate_model(model_name, tuned, X_train, y_train, config)
            final_pipeline = tuned

        rows[model_name] = cv_metrics
        pipelines[model_name] = final_pipeline

    comparison_df = pd.DataFrame(rows).T
    comparison_df.index.name = "model"

    # Best = highest mean recall (macro)
    best_model_name: str = comparison_df["recall_mean"].idxmax()
    best_pipeline = pipelines[best_model_name]

    logger.info(
        "Best model: %s (recall_mean=%.4f)",
        best_model_name,
        comparison_df.loc[best_model_name, "recall_mean"],
    )
    return comparison_df, best_model_name, best_pipeline


# ---------------------------------------------------------------------------
# Full training + saving pipeline
# ---------------------------------------------------------------------------

def train_and_save(config: dict) -> None:
    """End-to-end training workflow.

    Steps
    -----
    1. Load and merge raw CSVs.
    2. Create ``at_risk`` label.
    3. Clean data.
    4. Split 80/20 stratified on target.
    5. Compare all three models.
    6. Evaluate best model on the held-out test set.
    7. Persist:
       - ``models/best_model.joblib``       - fitted Pipeline
       - ``reports/metrics.json``            - full metrics payload
       - ``reports/model_comparison.csv``    - per-model CV metrics

    Parameters
    ----------
    config : dict
        Parsed config.yaml dictionary.
    """
    root = _project_root()

    # --- Resolve output paths ---
    paths_cfg = config.get("paths", {})
    models_dir = root / paths_cfg.get("models_dir", "models")
    reports_dir = root / paths_cfg.get("reports_dir", "reports")
    figures_dir = reports_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    seed: int = config.get("training", {}).get("random_seed", 42)
    test_size: float = config.get("training", {}).get("test_size", 0.2)
    target_col: str = config.get("label", {}).get("target_col", "at_risk")

    # --- Load data ---
    logger.info("Loading raw data...")
    df = load_raw_data(config)
    df = create_labels(df, config)
    df = clean_data(df, config)

    X, y = get_features_and_target(df, config, include_grades=True)

    logger.info(
        "Dataset: %d rows, %d features | class distribution: %s",
        len(X),
        X.shape[1],
        y.value_counts().to_dict(),
    )

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    logger.info(
        "Train size: %d | Test size: %d", len(X_train), len(X_test)
    )

    # --- Compare and select best model ---
    comparison_df, best_name, best_pipeline = compare_models(X_train, y_train, config)

    # --- Evaluate on test set ---
    logger.info("Evaluating best model (%s) on test set...", best_name)
    y_pred = best_pipeline.predict(X_test)
    threshold = 0.5

    if hasattr(best_pipeline, "predict_proba"):
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_proba))
    else:
        y_proba = None
        roc_auc = None

    test_metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "roc_auc": roc_auc,
    }

    cm = confusion_matrix(y_test, y_pred).tolist()

    train_metrics_dict = comparison_df.loc[best_name].to_dict()

    # --- Build metrics payload ---
    metrics_payload: dict[str, Any] = {
        "model_name": best_name,
        "threshold": threshold,
        "train_metrics": train_metrics_dict,
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "class_distribution": {
            "train": y_train.value_counts().to_dict(),
            "test": y_test.value_counts().to_dict(),
            "overall": y.value_counts().to_dict(),
        },
    }

    # --- Save model ---
    model_path = models_dir / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)
    logger.info("Model saved -> %s", model_path)

    # --- Save metrics ---
    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2, default=str)
    logger.info("Metrics saved -> %s", metrics_path)

    # --- Save comparison CSV ---
    comparison_path = reports_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path)
    logger.info("Comparison CSV saved -> %s", comparison_path)

    # --- Generate evaluation plots ---
    from src.dropguard.evaluate import (
        plot_confusion_matrix,
        plot_model_comparison,
        plot_feature_importance,
    )

    plot_confusion_matrix(
        y_test,
        y_pred,
        title=f"Confusion Matrix - {best_name}",
        save_path=figures_dir / "confusion_matrix.png",
    )

    plot_model_comparison(
        comparison_df,
        save_path=figures_dir / "model_comparison.png",
    )

    try:
        feature_names = get_feature_names(best_pipeline.named_steps["preprocessor"], X_train)
        plot_feature_importance(
            best_pipeline,
            feature_names=feature_names,
            top_n=20,
            save_path=figures_dir / "feature_importance.png",
        )
    except Exception as exc:
        logger.warning("Feature importance plot skipped: %s", exc)

    # --- Auto-generated counseling notes (assignment deliverable) ---
    # Shares SHAP explanations with the per-student risk drivers, so the notes
    # are grounded in the same model output shown in the Streamlit app.
    logger.info("Generating sample counseling notes for the highest-risk students...")
    from src.dropguard.counseling import generate_batch_notes, save_sample_notes
    from src.dropguard.explain import explain_students

    feature_names = get_feature_names(best_pipeline.named_steps["preprocessor"], X_train)
    n_notes = min(10, len(X_test))
    highest_risk_order = np.argsort(-y_proba)[:n_notes]
    X_sample = X_test.iloc[highest_risk_order]
    sample_proba = y_proba[highest_risk_order]

    students = explain_students(best_pipeline, X_sample, feature_names, config)
    notes = generate_batch_notes(students, sample_proba, X_sample)
    notes_path = reports_dir / "sample_counseling_notes.md"
    save_sample_notes(notes, notes_path)
    logger.info("Sample counseling notes saved -> %s", notes_path)

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point: ``python -m src.dropguard.train``."""
    config = load_config()
    train_and_save(config)


if __name__ == "__main__":
    main()
