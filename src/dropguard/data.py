"""
src/dropguard/data.py
---------------------
Data loading, labelling, cleaning, and feature-extraction pipeline for the
DropGuard student-performance project.

Public API
----------
load_config          - Load ``config.yaml`` with auto-discovery of project root.
load_raw_data        - Read both raw CSVs and return a combined DataFrame.
create_labels        - Add binary ``at_risk`` column.
clean_data           - Drop bad rows, encode categoricals.
get_features_and_target - Split into X, y.
prepare_data         - Full end-to-end pipeline convenience wrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Columns whose raw values are "yes"/"no" strings that should be 0/1 ints.
_BINARY_YES_NO_COLS: list[str] = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]

#: Mapping from each binary yes/no column to the value that becomes 1.
#: For most columns "yes" -> 1; for a few the encoding follows the UCI spec.
_BINARY_POSITIVE_VALUE: dict[str, str] = {
    "school": "GP",          # GP -> 1, MS -> 0
    "sex": "F",              # F -> 1, M -> 0
    "address": "U",          # Urban -> 1, Rural -> 0
    "famsize": "GT3",        # >3 -> 1, ≤3 (LE3) -> 0
    "Pstatus": "T",          # Together -> 1, Apart (A) -> 0
    "schoolsup": "yes",
    "famsup": "yes",
    "paid": "yes",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "yes",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_project_root(start: Path) -> Path:
    """Walk up the directory tree from *start* until ``config.yaml`` is found.

    Parameters
    ----------
    start:
        Starting directory (typically ``Path(__file__).parent``).

    Returns
    -------
    Path
        The first ancestor directory that contains ``config.yaml``.

    Raises
    ------
    FileNotFoundError
        If ``config.yaml`` is not found before the filesystem root.
    """
    current: Path = start.resolve()
    for ancestor in [current, *current.parents]:
        if (ancestor / "config.yaml").exists():
            return ancestor
    raise FileNotFoundError(
        f"Could not find config.yaml starting from {start}. "
        "Make sure the project has been scaffolded correctly."
    )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def load_config(config_path: Union[str, Path, None] = None) -> dict:
    """Load ``config.yaml`` from the project root.

    If *config_path* is ``None``, the project root is auto-discovered by
    walking up the directory tree from this module's location until a
    ``config.yaml`` file is found.

    Parameters
    ----------
    config_path:
        Explicit path to a YAML config file, or ``None`` for auto-discovery.

    Returns
    -------
    dict
        Parsed YAML configuration mapping.

    Raises
    ------
    FileNotFoundError
        If the config file cannot be found.
    """
    if config_path is None:
        project_root: Path = _find_project_root(Path(__file__).parent)
        resolved: Path = project_root / "config.yaml"
    else:
        resolved = Path(config_path).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as fh:
        config: dict = yaml.safe_load(fh)

    return config


def load_raw_data(config: dict) -> pd.DataFrame:
    """Load both raw CSVs and return a combined DataFrame.

    Both ``student-mat.csv`` and ``student-por.csv`` are semicolon-separated.
    A ``'course'`` column (string values ``'math'`` / ``'portuguese'``) is
    added to each before concatenation so the source can be identified later.

    Parameters
    ----------
    config:
        Parsed configuration dictionary (as returned by :func:`load_config`).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with a reset integer index and an extra ``'course'``
        column.

    Raises
    ------
    FileNotFoundError
        If either CSV file cannot be found in the configured raw directory.
    """
    # Resolve the raw data directory
    project_root: Path = _find_project_root(Path(__file__).parent)
    raw_dir_rel: str = config.get("data", {}).get("raw_dir", "data/raw")
    raw_dir: Path = project_root / raw_dir_rel

    math_path: Path = raw_dir / "student-mat.csv"
    por_path: Path = raw_dir / "student-por.csv"

    for p in (math_path, por_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Raw data file not found: {p}. "
                "Run `python scripts/download_data.py` first."
            )

    # Load each file, then tag with course name before concatenating
    df_math: pd.DataFrame = pd.read_csv(math_path, sep=";")
    df_math["course"] = "math"

    df_por: pd.DataFrame = pd.read_csv(por_path, sep=";")
    df_por["course"] = "portuguese"

    combined: pd.DataFrame = pd.concat(
        [df_math, df_por], ignore_index=True
    )
    return combined


def create_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add a binary ``at_risk`` column derived from the final grade ``G3``.

    A student is considered at-risk if their ``G3`` score is strictly less than
    the threshold defined in ``config['label']['at_risk_threshold']``.

    This function **never** modifies the input DataFrame in place.

    Parameters
    ----------
    df:
        Input DataFrame (must contain a ``'G3'`` column).
    config:
        Parsed configuration dictionary.

    Returns
    -------
    pd.DataFrame
        New DataFrame identical to *df* but with an additional ``'at_risk'``
        column (``int``, values ``0`` or ``1``).
    """
    threshold: int = int(
        config.get("label", {}).get("at_risk_threshold", 10)
    )
    result: pd.DataFrame = df.copy()
    result["at_risk"] = (result["G3"] < threshold).astype(int)
    return result


def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:  # noqa: ARG001
    """Clean the raw (labelled) DataFrame.

    Steps performed (in order):

    1. Drop rows where ``G3`` is ``NaN``.
    2. Encode all binary yes/no string columns (see :data:`_BINARY_YES_NO_COLS`)
       as ``int`` (0 or 1).
    3. Encode the ``'course'`` column as ``int`` (``math=0``,
       ``portuguese=1``).

    Parameters
    ----------
    df:
        Input DataFrame (should already have an ``at_risk`` column from
        :func:`create_labels`).
    config:
        Parsed configuration dictionary (currently unused but kept for a
        consistent interface).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a reset index.
    """
    result: pd.DataFrame = df.copy()

    # 1. Drop rows with missing G3
    result = result.dropna(subset=["G3"])

    # 2. Encode binary yes/no columns
    for col in _BINARY_YES_NO_COLS:
        if col not in result.columns:
            continue
        positive_val: str = _BINARY_POSITIVE_VALUE.get(col, "yes")
        result[col] = (result[col] == positive_val).astype(int)

    # 3. Encode 'course' column: math -> 0, portuguese -> 1
    if "course" in result.columns:
        result["course"] = (result["course"] == "portuguese").astype(int)

    result = result.reset_index(drop=True)
    return result


def get_features_and_target(
    df: pd.DataFrame,
    config: dict,
    include_grades: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned DataFrame into feature matrix *X* and target series *y*.

    ``G3`` is **always** excluded from *X* to prevent label leakage.
    When *include_grades* is ``False``, the intermediate grade columns ``G1``
    and ``G2`` are also excluded, enabling an early-warning ablation study.

    Parameters
    ----------
    df:
        Cleaned DataFrame (output of :func:`clean_data`).
    config:
        Parsed configuration dictionary.  The key
        ``config['label']['target_col']`` (default ``'at_risk'``) specifies
        the target column name.
    include_grades:
        If ``True`` (default), include ``G1`` and ``G2`` in features.
        If ``False``, exclude both intermediate grade columns.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` where *X* is the feature matrix and *y* is the target
        series.

    Raises
    ------
    KeyError
        If the target column is not present in *df*.
    """
    target_col: str = config.get("label", {}).get("target_col", "at_risk")

    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in DataFrame. "
            "Make sure create_labels() has been called first."
        )

    # Columns to exclude from X
    exclude: set[str] = {"G3", target_col}
    if not include_grades:
        exclude.update({"G1", "G2"})

    feature_cols: list[str] = [c for c in df.columns if c not in exclude]

    X: pd.DataFrame = df[feature_cols].copy()
    y: pd.Series = df[target_col].copy()

    return X, y


def prepare_data(
    config: dict,
    include_grades: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run the full data preparation pipeline and return ``(X, y)``.

    Internally chains: :func:`load_raw_data` -> :func:`create_labels` ->
    :func:`clean_data` -> :func:`get_features_and_target`.

    Parameters
    ----------
    config:
        Parsed configuration dictionary (as returned by :func:`load_config`).
    include_grades:
        Forwarded to :func:`get_features_and_target`. Set to ``False`` to
        exclude ``G1`` / ``G2`` for an early-warning ablation study.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` ready for model training.
    """
    df_raw: pd.DataFrame = load_raw_data(config)
    df_labelled: pd.DataFrame = create_labels(df_raw, config)
    df_clean: pd.DataFrame = clean_data(df_labelled, config)
    X, y = get_features_and_target(df_clean, config, include_grades=include_grades)
    return X, y
