"""
dropguard.features
==================
Sklearn preprocessing pipeline utilities for the DropGuard project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_numeric_features(X: pd.DataFrame) -> list[str]:
    """Return list of numeric column names (int64/float64).

    Parameters
    ----------
    X : pd.DataFrame
        Input feature DataFrame.

    Returns
    -------
    list[str]
        Column names whose dtype is int64 or float64.
    """
    return X.select_dtypes(include=["int64", "float64"]).columns.tolist()


def get_categorical_features(X: pd.DataFrame) -> list[str]:
    """Return list of object dtype columns.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature DataFrame.

    Returns
    -------
    list[str]
        Column names whose dtype is object.
    """
    return X.select_dtypes(include=["object"]).columns.tolist()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build an sklearn ColumnTransformer for numeric and categorical features.

    Numeric pipeline  : SimpleImputer(strategy='median') -> StandardScaler
    Categorical pipeline: SimpleImputer(strategy='most_frequent') ->
                          OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    The returned transformer is *not* yet fitted; call ``fit`` / ``fit_transform``
    on training data before use.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature DataFrame used to detect column dtypes.

    Returns
    -------
    ColumnTransformer
        Fitted-ready (unfitted) preprocessing transformer.
    """
    numeric_features = get_numeric_features(X)
    categorical_features = get_categorical_features(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def build_pipeline(classifier, X: pd.DataFrame) -> Pipeline:
    """Return an sklearn Pipeline combining the preprocessor and a classifier.

    Parameters
    ----------
    classifier : sklearn-compatible estimator
        Any estimator that implements ``fit`` / ``predict``.
    X : pd.DataFrame
        Input feature DataFrame used to detect column dtypes for the preprocessor.

    Returns
    -------
    Pipeline
        An unfitted Pipeline with steps:
        ``[('preprocessor', ColumnTransformer), ('classifier', classifier)]``.
    """
    preprocessor = build_preprocessor(X)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def get_feature_names(preprocessor: ColumnTransformer, X: pd.DataFrame) -> list[str]:
    """Get output feature names after the ColumnTransformer has been fitted.

    Handles the expanded column names produced by OneHotEncoder for categorical
    features.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        A *fitted* ColumnTransformer (e.g. from ``build_preprocessor``).
    X : pd.DataFrame
        The DataFrame that was used to fit the preprocessor (used as a fallback
        for transformers that do not expose ``get_feature_names_out``).

    Returns
    -------
    list[str]
        Flat list of output feature names in transformer order.
    """
    feature_names: list[str] = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            # Skip remainder columns (dropped by default)
            continue

        if hasattr(transformer, "get_feature_names_out"):
            # Pipeline or single transformer that supports the sklearn API
            names = transformer.get_feature_names_out(columns)
            feature_names.extend(names.tolist())
        elif isinstance(transformer, Pipeline):
            # Walk to the last step and try there
            last_step = transformer.steps[-1][1]
            if hasattr(last_step, "get_feature_names_out"):
                names = last_step.get_feature_names_out(columns)
                feature_names.extend(names.tolist())
            else:
                feature_names.extend(list(columns))
        else:
            feature_names.extend(list(columns))

    return feature_names
