"""
evaluate.py - DropGuard evaluation and visualisation utilities.

All plot functions save PNGs to ``reports/figures/`` by default (via the
``save_path`` argument). Every function is fully self-contained and can be
called independently without running the full training pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend - safe for scripts

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

_PALETTE = "Blues"
_FIG_DPI = 150


def _ensure_dir(path: Path) -> None:
    """Create parent directories for *path* if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    title: str,
    save_path: Path,
) -> None:
    """Plot and save a labelled confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    title : str
        Figure title.
    save_path : Path
        Destination ``.png`` file (parent directories are created as needed).
    """
    save_path = Path(save_path)
    _ensure_dir(save_path)

    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(set(y_true))

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=_PALETTE,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        linewidths=0.5,
        linecolor="grey",
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=_FIG_DPI)
    plt.close(fig)
    logger.info("Confusion matrix saved -> %s", save_path)


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    models_data: list[dict],
    save_path: Path,
) -> None:
    """Plot ROC curves for multiple models on the same axes and save as PNG.

    Parameters
    ----------
    models_data : list[dict]
        Each element must contain:
        - ``'name'``    : str  - model label for the legend.
        - ``'y_true'``  : array-like - ground-truth binary labels.
        - ``'y_proba'`` : array-like - predicted probabilities for the positive class.
    save_path : Path
        Destination ``.png`` file.
    """
    save_path = Path(save_path)
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(7, 6))

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for idx, model_info in enumerate(models_data):
        name = model_info["name"]
        y_true = model_info["y_true"]
        y_proba = model_info["y_proba"]

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc_val:.3f})",
            color=colors[idx % len(colors)],
            lw=2,
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves - Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_FIG_DPI)
    plt.close(fig)
    logger.info("ROC curves saved -> %s", save_path)


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Bar chart comparing accuracy, precision, recall, f1, and roc_auc across models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame indexed by model name. Expected columns (at minimum):
        ``accuracy_mean``, ``precision_mean``, ``recall_mean``,
        ``f1_mean``, ``roc_auc_mean``.
    save_path : Path
        Destination ``.png`` file.
    """
    save_path = Path(save_path)
    _ensure_dir(save_path)

    metrics = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    # Keep only the mean columns that actually exist
    present = [m for m in metrics if m in comparison_df.columns]
    present_labels = [metric_labels[metrics.index(m)] for m in present]

    plot_df = comparison_df[present].copy()
    plot_df.columns = present_labels
    plot_df.index.name = "Model"

    n_metrics = len(present_labels)
    n_models = len(plot_df)
    bar_width = 0.6 / n_metrics
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(8, n_models * 2.5), 5))
    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]

    for i, metric_label in enumerate(present_labels):
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            plot_df[metric_label],
            width=bar_width,
            label=metric_label,
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.005,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (macro average)", fontsize=11)
    ax.set_title("Cross-Validation Metric Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=n_metrics)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_FIG_DPI)
    plt.close(fig)
    logger.info("Model comparison chart saved -> %s", save_path)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    pipeline: Pipeline,
    feature_names: list[str],
    top_n: int,
    save_path: Path,
) -> None:
    """Extract and plot top-N feature importances from an RF or XGBoost classifier.

    For models that expose ``feature_importances_`` (Random Forest, XGBoost)
    the importances are extracted directly. For logistic regression the
    absolute coefficient values are used.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn Pipeline whose last step is named ``'classifier'``.
    feature_names : list[str]
        Names of features after preprocessing (must match the length of the
        importance array).
    top_n : int
        Number of top features to display.
    save_path : Path
        Destination ``.png`` file.
    """
    save_path = Path(save_path)
    _ensure_dir(save_path)

    classifier = pipeline.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        importance_label = "Feature Importance"
    elif hasattr(classifier, "coef_"):
        coef = classifier.coef_
        # For multi-class the shape is (n_classes, n_features); take mean of abs
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef[0])
        importance_label = "|Coefficient|"
    else:
        raise ValueError(
            f"Classifier {type(classifier).__name__!r} does not expose "
            "feature_importances_ or coef_."
        )

    # Align feature names with importance array
    n = min(len(feature_names), len(importances))
    names = list(feature_names)[:n]
    importances = importances[:n]

    # Select top-N
    top_n = min(top_n, n)
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [names[i] for i in indices]
    top_imps = importances[indices]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    bars = ax.barh(
        range(top_n),
        top_imps[::-1],
        color=sns.color_palette("Blues_r", top_n),
        edgecolor="white",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel(importance_label, fontsize=11)
    ax.set_title(
        f"Top {top_n} Feature Importances - {type(classifier).__name__}",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_FIG_DPI)
    plt.close(fig)
    logger.info("Feature importance chart saved -> %s", save_path)


# ---------------------------------------------------------------------------
# Metric computation helper
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: Any = None,
) -> dict:
    """Compute a standard set of classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    y_proba : array-like, optional
        Predicted probabilities for the positive class. When provided,
        ``roc_auc`` is included in the output dict.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``
        (all using macro averaging), and optionally ``roc_auc``.
    """
    result: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if y_proba is not None:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError as exc:
            logger.warning("roc_auc_score failed: %s", exc)
            result["roc_auc"] = None

    return result
