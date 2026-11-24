"""
dropguard.experiments
=====================
Runs all 6 supervised-learning experiments for the DropGuard project.

All outputs (CSV, JSON, PNG) are saved to paths derived from ``config.yaml``;
no paths are hardcoded.

Experiments
-----------
1. Model comparison        - bar chart + CSV of CV scores across classifiers.
2. Class imbalance         - no_weight / class_weight / SMOTE on XGBoost.
3. Learning curves         - train vs validation score over dataset sizes.
4. Threshold tuning        - precision-recall, F-beta optimisation, CMs.
5. Feature ablation        - WITH vs WITHOUT G1/G2 grades.
6. Regression variant      - predict G3 numerically (MAE, RMSE).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict, cross_val_score, learning_curve, train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier

from src.dropguard.data import (
    clean_data,
    create_labels,
    get_features_and_target,
    load_config,
    load_raw_data,
)
from src.dropguard.features import build_pipeline, build_preprocessor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_dirs(config: dict) -> tuple[Path, Path, Path]:
    """Return (project_root, reports_dir, figures_dir) as absolute Paths.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.

    Returns
    -------
    tuple[Path, Path, Path]
        ``(project_root, reports_dir, figures_dir)``
    """
    project_root = Path(__file__).resolve().parents[2]
    reports_dir = project_root / config["paths"]["reports_dir"]
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return project_root, reports_dir, figures_dir


def _load_data(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load, clean, label and split the student dataset.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` ready for model training / evaluation.
    """
    df_raw = load_raw_data(config)
    df_labelled = create_labels(df_raw, config)
    df_clean = clean_data(df_labelled, config)
    X, y = get_features_and_target(df_clean, config, include_grades=True)
    return X, y


def _train_test(
    X: pd.DataFrame, y: pd.Series, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split ``X`` / ``y`` into train and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    config : dict
        Loaded configuration dictionary.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        ``(X_train, X_test, y_train, y_test)``
    """
    return train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_seed"],
        stratify=y,
    )


# ---------------------------------------------------------------------------
# Experiment 1 - Model comparison
# ---------------------------------------------------------------------------

def experiment_1_model_comparison(
    config: dict,
    reports_dir: Path,
    figures_dir: Path,
) -> None:
    """Compare Logistic Regression, Random Forest, and XGBoost via cross-validation.

    Saves ``reports/model_comparison.csv`` and
    ``reports/figures/model_comparison_bar.png``.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.
    reports_dir : Path
        Directory for CSV output.
    figures_dir : Path
        Directory for PNG output.
    """
    print("[Exp 1] Model comparison ...")
    X, y = _load_data(config)
    seed = config["training"]["random_seed"]
    cv = config["training"]["cv_folds"]

    classifiers: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            max_iter=config["models"]["logistic"]["max_iter"],
            random_state=seed,
        ),
        "RandomForest": __import__(
            "sklearn.ensemble", fromlist=["RandomForestClassifier"]
        ).RandomForestClassifier(
            n_estimators=100, random_state=seed
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=seed, verbosity=0, eval_metric="logloss"
        ),
    }

    rows: list[dict[str, Any]] = []
    for name, clf in classifiers.items():
        pipeline = build_pipeline(clf, X)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")
        rows.append(
            {
                "model": name,
                "cv_f1_mean": round(float(scores.mean()), 4),
                "cv_f1_std": round(float(scores.std()), 4),
            }
        )
        print(f"  {name}: f1={scores.mean():.4f} ± {scores.std():.4f}")

    df_res = pd.DataFrame(rows)
    csv_path = reports_dir / "model_comparison_exp1.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"  -> {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df_res["model"], df_res["cv_f1_mean"], yerr=df_res["cv_f1_std"],
           capsize=5, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_xlabel("Model")
    ax.set_ylabel("CV F1-macro (mean ± std)")
    ax.set_title("Experiment 1 - Model Comparison")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    png_path = figures_dir / "model_comparison_bar.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  -> {png_path}")


# ---------------------------------------------------------------------------
# Experiment 2 - Class imbalance handling
# ---------------------------------------------------------------------------

def experiment_2_imbalance(
    config: dict,
    reports_dir: Path,
    figures_dir: Path,
) -> None:
    """Compare three class-imbalance strategies on XGBoost.

    Strategies: ``no_weight``, ``class_weight`` (scale_pos_weight), ``smote``.
    Metric: recall on the minority (at-risk) class.

    Saves ``reports/imbalance_comparison.csv`` and
    ``reports/figures/imbalance_comparison_bar.png``.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.
    reports_dir : Path
        Directory for CSV output.
    figures_dir : Path
        Directory for PNG output.
    """
    print("[Exp 2] Class imbalance handling ...")
    X, y = _load_data(config)
    seed = config["training"]["random_seed"]
    cv = config["training"]["cv_folds"]

    neg_count = int((y == 0).sum())
    pos_count = int((y == 1).sum())
    ratio = neg_count / max(pos_count, 1)

    preprocessor = build_preprocessor(X)

    # Strategy a: no weight
    clf_no = XGBClassifier(verbosity=0, random_state=seed, eval_metric="logloss")
    pipe_no = ImbPipeline(
        [("preprocessor", preprocessor), ("classifier", clf_no)]
    )

    # Strategy b: scale_pos_weight
    clf_cw = XGBClassifier(
        scale_pos_weight=ratio, verbosity=0, random_state=seed, eval_metric="logloss"
    )
    preprocessor_cw = build_preprocessor(X)
    pipe_cw = ImbPipeline(
        [("preprocessor", preprocessor_cw), ("classifier", clf_cw)]
    )

    # Strategy c: SMOTE
    clf_sm = XGBClassifier(verbosity=0, random_state=seed, eval_metric="logloss")
    preprocessor_sm = build_preprocessor(X)
    pipe_sm = ImbPipeline(
        [
            ("preprocessor", preprocessor_sm),
            ("smote", SMOTE(random_state=seed)),
            ("classifier", clf_sm),
        ]
    )

    strategies = {
        "no_weight": pipe_no,
        "class_weight": pipe_cw,
        "smote": pipe_sm,
    }

    rows: list[dict[str, Any]] = []
    for name, pipe in strategies.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="recall")
        rows.append(
            {
                "strategy": name,
                "cv_recall_mean": round(float(scores.mean()), 4),
                "cv_recall_std": round(float(scores.std()), 4),
            }
        )
        print(f"  {name}: recall={scores.mean():.4f} ± {scores.std():.4f}")

    df_res = pd.DataFrame(rows)
    csv_path = reports_dir / "imbalance_comparison.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"  -> {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        df_res["strategy"],
        df_res["cv_recall_mean"],
        yerr=df_res["cv_recall_std"],
        capsize=5,
        color=["#4C72B0", "#55A868", "#C44E52"],
    )
    ax.set_xlabel("Strategy")
    ax.set_ylabel("CV Recall (mean ± std)")
    ax.set_title("Experiment 2 - Class Imbalance Strategies")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    png_path = figures_dir / "imbalance_comparison_bar.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  -> {png_path}")


# ---------------------------------------------------------------------------
# Experiment 3 - Learning curves
# ---------------------------------------------------------------------------

def experiment_3_learning_curves(
    config: dict,
    reports_dir: Path,
    figures_dir: Path,
) -> None:
    """Plot training vs validation scores across increasing dataset sizes.

    Uses the XGBoost pipeline (best model) with ``sklearn.model_selection.learning_curve``.
    Saves ``reports/figures/learning_curve.png``.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.
    reports_dir : Path
        Directory (unused here, kept for API consistency).
    figures_dir : Path
        Directory for PNG output.
    """
    print("[Exp 3] Learning curves ...")
    X, y = _load_data(config)
    seed = config["training"]["random_seed"]
    cv = config["training"]["cv_folds"]

    clf = XGBClassifier(verbosity=0, random_state=seed, eval_metric="logloss")
    pipeline = build_pipeline(clf, X)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        pipeline,
        X,
        y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes_abs, train_mean, "o-", color="#4C72B0", label="Train score")
    ax.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color="#4C72B0",
    )
    ax.plot(train_sizes_abs, val_mean, "o-", color="#C44E52", label="Validation score")
    ax.fill_between(
        train_sizes_abs,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.2,
        color="#C44E52",
    )
    ax.set_xlabel("Training samples")
    ax.set_ylabel("F1-macro")
    ax.set_title("Experiment 3 - Learning Curves (XGBoost)")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    png_path = figures_dir / "learning_curve.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  -> {png_path}")


# ---------------------------------------------------------------------------
# Experiment 4 - Threshold tuning
# ---------------------------------------------------------------------------

def _fbeta_score(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    """Compute F-beta score element-wise.

    Parameters
    ----------
    precision : np.ndarray
        Precision values.
    recall : np.ndarray
        Recall values.
    beta : float
        Beta parameter (beta > 1 favours recall).

    Returns
    -------
    np.ndarray
        F-beta score for each (precision, recall) pair.
    """
    b2 = beta ** 2
    denom = b2 * precision + recall
    with np.errstate(invalid="ignore", divide="ignore"):
        fb = np.where(denom == 0, 0.0, (1 + b2) * precision * recall / denom)
    return fb


def experiment_4_threshold_tuning(
    config: dict,
    reports_dir: Path,
    figures_dir: Path,
) -> None:
    """Tune decision threshold on out-of-fold training predictions (beta=2).

    Loads the best model from ``models/best_model.joblib`` and tunes the
    threshold on out-of-fold (OOF) predictions over the training set, so the
    test set is never used for threshold selection. Final metrics are then
    computed on the held-out test set.

    Saves:
    - ``reports/figures/precision_recall_curve.png``
    - ``reports/figures/cm_default_threshold.png``
    - ``reports/figures/cm_optimal_threshold.png``
    - ``reports/threshold_results.json``

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.
    reports_dir : Path
        Directory for JSON output.
    figures_dir : Path
        Directory for PNG output.
    """
    print("[Exp 4] Threshold tuning ...")
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / config["paths"]["models_dir"] / "best_model.joblib"

    X, y = _load_data(config)
    X_train, X_test, y_train, y_test = _train_test(X, y, config)

    seed = config["training"]["random_seed"]
    cv = config["training"]["cv_folds"]

    # Load or fallback: train an XGBoost pipeline if no saved model yet
    if model_path.exists():
        best_pipeline = joblib.load(model_path)
    else:
        print("  [warn] best_model.joblib not found - training fresh XGBoost pipeline.")
        clf = XGBClassifier(verbosity=0, random_state=seed, eval_metric="logloss")
        best_pipeline = build_pipeline(clf, X_train)
        best_pipeline.fit(X_train, y_train)

    # Out-of-fold probabilities on the training set (no test leakage)
    oof_proba = cross_val_predict(
        best_pipeline, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]

    # Precision-Recall curve on OOF predictions
    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_proba)
    pr_auc = auc(recalls, precisions)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recalls, precisions, lw=2, color="#4C72B0",
            label=f"PR curve (AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Experiment 4 - Precision-Recall Curve (OOF train)")
    ax.legend()
    plt.tight_layout()
    pr_path = figures_dir / "precision_recall_curve.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    print(f"  -> {pr_path}")

    # Find threshold maximising F-beta (beta=2) on OOF predictions
    beta = 2.0
    p_cut = precisions[:-1]
    r_cut = recalls[:-1]
    fb_scores = _fbeta_score(p_cut, r_cut, beta)
    best_idx = int(np.argmax(fb_scores))
    best_threshold = float(thresholds[best_idx])
    best_fb = float(fb_scores[best_idx])
    print(f"  Optimal threshold: {best_threshold:.4f} (F{beta}={best_fb:.4f})")

    # Test-set probabilities from the deployed model
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    # Confusion matrix at default threshold (0.5)
    y_pred_default = (y_proba >= 0.5).astype(int)
    cm_default = confusion_matrix(y_test, y_pred_default)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_default,
                                  display_labels=["not_at_risk", "at_risk"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("CM - default threshold (0.5)")
    plt.tight_layout()
    cm_def_path = figures_dir / "cm_default_threshold.png"
    fig.savefig(cm_def_path, dpi=150)
    plt.close(fig)
    print(f"  -> {cm_def_path}")

    # Confusion matrix at optimal threshold
    y_pred_opt = (y_proba >= best_threshold).astype(int)
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_opt,
                                  display_labels=["not_at_risk", "at_risk"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"CM - optimal threshold ({best_threshold:.3f})")
    plt.tight_layout()
    cm_opt_path = figures_dir / "cm_optimal_threshold.png"
    fig.savefig(cm_opt_path, dpi=150)
    plt.close(fig)
    print(f"  -> {cm_opt_path}")

    # JSON results
    precision_default = float(precision_score(y_test, y_pred_default, zero_division=0))
    results = {
        "best_threshold": round(best_threshold, 6),
        f"f{int(beta)}_at_best_threshold": round(best_fb, 6),
        "pr_auc": round(float(pr_auc), 6),
        "recall_at_default": round(float(recall_score(y_test, y_pred_default)), 6),
        "recall_at_optimal": round(float(recall_score(y_test, y_pred_opt)), 6),
        "precision_at_default": round(precision_default, 6),
        "tuned_on": "out_of_fold_train",
    }
    json_path = reports_dir / "threshold_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  -> {json_path}")


# ---------------------------------------------------------------------------
# Experiment 5 - Feature ablation (G1 / G2)
# ---------------------------------------------------------------------------

def experiment_5_ablation(
    config: dict,
    reports_dir: Path,
    figures_dir: Path,
) -> None:
    """Compare model performance with and without early-grade features (G1, G2).

    Saves ``reports/ablation_results.json`` and
    ``reports/figures/ablation_bar.png``.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.
    reports_dir : Path
        Directory for JSON output.
    figures_dir : Path
        Directory for PNG output.
    """
    print("[Exp 5] Feature ablation (G1/G2) ...")
    seed = config["training"]["random_seed"]

    df_raw = load_raw_data(config)
    df_labelled = create_labels(df_raw, config)
    df_clean = clean_data(df_labelled, config)

    X_full, y = get_features_and_target(df_clean, config, include_grades=True)
    X_train_f, X_test_f, y_train, y_test = _train_test(X_full, y, config)

    grade_cols = [c for c in ["G1", "G2"] if c in X_full.columns]

    X_no = X_full.drop(columns=grade_cols, errors="ignore")
    X_train_n, X_test_n, _, _ = train_test_split(
        X_no,
        y,
        test_size=config["training"]["test_size"],
        random_state=seed,
        stratify=y,
    )

    def _fit_eval(X_tr: pd.DataFrame, X_te: pd.DataFrame,
                  y_tr: pd.Series, y_te: pd.Series) -> dict[str, float]:
        clf = XGBClassifier(verbosity=0, random_state=seed, eval_metric="logloss")
        pipe = build_pipeline(clf, X_tr)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_proba = pipe.predict_proba(X_te)[:, 1]
        return {
            "recall": round(float(recall_score(y_te, y_pred)), 6),
            "f1": round(float(f1_score(y_te, y_pred)), 6),
            "roc_auc": round(float(roc_auc_score(y_te, y_proba)), 6),
        }

    with_grades = _fit_eval(X_train_f, X_test_f, y_train, y_test)
    without_grades = _fit_eval(X_train_n, X_test_n, y_train, y_test)

    print(f"  With G1/G2:    {with_grades}")
    print(f"  Without G1/G2: {without_grades}")

    results = {"with_G1_G2": with_grades, "without_G1_G2": without_grades}
    json_path = reports_dir / "ablation_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  -> {json_path}")

    # Bar chart
    metrics = ["recall", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2,
                   [with_grades[m] for m in metrics],
                   width, label="With G1/G2", color="#4C72B0")
    bars2 = ax.bar(x + width / 2,
                   [without_grades[m] for m in metrics],
                   width, label="Without G1/G2", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Experiment 5 - Feature Ablation (G1/G2)")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    png_path = figures_dir / "ablation_bar.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  -> {png_path}")


# ---------------------------------------------------------------------------
# Experiment 6 - Regression variant
# ---------------------------------------------------------------------------

def experiment_6_regression(
    config: dict,
    reports_dir: Path,
    figures_dir: Path,
) -> None:
    """Predict G3 numerically, comparing LinearRegression and RandomForestRegressor.

    Features: all columns except G3 (and not the ``at_risk`` label).
    Metric: MAE, RMSE.

    Saves ``reports/regression_results.json`` and
    ``reports/figures/regression_bar.png``.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.
    reports_dir : Path
        Directory for JSON output.
    figures_dir : Path
        Directory for PNG output.
    """
    print("[Exp 6] Regression variant (predict G3) ...")
    seed = config["training"]["random_seed"]

    df_raw = load_raw_data(config)
    df_labelled = create_labels(df_raw, config)
    df_clean = clean_data(df_labelled, config)

    # For regression, target = G3; drop at_risk label column
    target_col = "G3"
    drop_cols = [config["label"]["target_col"]]  # 'at_risk'
    if target_col in df_clean.columns:
        y_reg = df_clean[target_col].copy()
        X_reg = df_clean.drop(columns=[target_col] + drop_cols, errors="ignore")
    else:
        raise ValueError("G3 column not found in cleaned data for regression experiment.")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_reg, y_reg,
        test_size=config["training"]["test_size"],
        random_state=seed,
    )

    regressors: dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100, random_state=seed
        ),
    }

    rows: list[dict[str, Any]] = []
    for name, reg in regressors.items():
        from src.dropguard.features import build_preprocessor as _bp
        pre = _bp(X_tr)
        from sklearn.pipeline import Pipeline as SkPipeline
        pipe = SkPipeline([("preprocessor", pre), ("regressor", reg)])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        mae = float(mean_absolute_error(y_te, y_pred))
        rmse = float(np.sqrt(np.mean((y_te - y_pred) ** 2)))
        rows.append({"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4)})
        print(f"  {name}: MAE={mae:.4f}  RMSE={rmse:.4f}")

    results = {r["model"]: {"MAE": r["MAE"], "RMSE": r["RMSE"]} for r in rows}
    json_path = reports_dir / "regression_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  -> {json_path}")

    # Bar chart: grouped MAE / RMSE
    models_list = [r["model"] for r in rows]
    mae_vals = [r["MAE"] for r in rows]
    rmse_vals = [r["RMSE"] for r in rows]
    x = np.arange(len(models_list))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, mae_vals, width, label="MAE", color="#4C72B0")
    ax.bar(x + width / 2, rmse_vals, width, label="RMSE", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=15, ha="right")
    ax.set_ylabel("Error (grade points)")
    ax.set_title("Experiment 6 - Regression Variant (G3 prediction)")
    ax.legend()
    plt.tight_layout()
    png_path = figures_dir / "regression_bar.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  -> {png_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_all_experiments(config: dict) -> None:
    """Run experiments 1-6 in sequence, printing progress for each.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary (from ``load_config()``).
    """
    _, reports_dir, figures_dir = _resolve_dirs(config)

    print("=" * 60)
    print("DropGuard - Running all 6 experiments")
    print("=" * 60)

    experiment_1_model_comparison(config, reports_dir, figures_dir)
    experiment_2_imbalance(config, reports_dir, figures_dir)
    experiment_3_learning_curves(config, reports_dir, figures_dir)
    experiment_4_threshold_tuning(config, reports_dir, figures_dir)
    experiment_5_ablation(config, reports_dir, figures_dir)
    experiment_6_regression(config, reports_dir, figures_dir)

    print("=" * 60)
    print("All experiments complete. Results saved to:", reports_dir)
    print("=" * 60)


if __name__ == "__main__":
    from src.dropguard.data import load_config  # noqa: F811 - re-import for __main__

    config = load_config()
    run_all_experiments(config)
