"""
tests/test_features.py
Tests for src.dropguard.features: preprocessor pipeline building.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.dropguard.data import clean_data, create_labels, get_features_and_target
from src.dropguard.features import build_pipeline, build_preprocessor, get_numeric_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_xy(sample_config, synthetic_df):
    """Return (X_train, X_test, y_train, y_test) from the synthetic dataset."""
    from sklearn.model_selection import train_test_split

    df = clean_data(synthetic_df, sample_config)
    df = create_labels(df, sample_config)
    X, y = get_features_and_target(df, sample_config)

    seed = sample_config["training"]["random_seed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPreprocessor:
    def test_preprocessor_no_nans(self, clean_xy):
        """After fit_transform on X_train the output must have no NaN values."""
        X_train, X_test, y_train, _ = clean_xy
        preprocessor = build_preprocessor(X_train)
        X_transformed = preprocessor.fit_transform(X_train)

        # Result may be a dense numpy array or sparse matrix
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        assert not np.isnan(X_transformed).any(), "NaNs found in preprocessor output"

    def test_consistent_columns(self, clean_xy):
        """X_train and X_test must produce the same number of columns after transform."""
        X_train, X_test, y_train, _ = clean_xy
        preprocessor = build_preprocessor(X_train)
        preprocessor.fit(X_train)

        out_train = preprocessor.transform(X_train)
        out_test = preprocessor.transform(X_test)

        if hasattr(out_train, "toarray"):
            out_train = out_train.toarray()
        if hasattr(out_test, "toarray"):
            out_test = out_test.toarray()

        assert out_train.shape[1] == out_test.shape[1], (
            f"Column count mismatch: train={out_train.shape[1]}, test={out_test.shape[1]}"
        )


class TestBuildPipeline:
    def test_pipeline_builds(self, sample_config, clean_xy):
        """build_pipeline must return a scikit-learn Pipeline instance."""
        X_train, _, _, _ = clean_xy
        pipeline = build_pipeline(LogisticRegression(max_iter=1000), X_train)
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_classifier_step(self, sample_config, clean_xy):
        """The pipeline must expose a classifier step (last step)."""
        X_train, _, _, _ = clean_xy
        pipeline = build_pipeline(LogisticRegression(max_iter=1000), X_train)
        # Last named step should have predict / predict_proba
        last_step = pipeline.steps[-1][1]
        assert hasattr(last_step, "predict")


class TestGetNumericFeatures:
    def test_numeric_features_detected(self, sample_config, synthetic_df):
        """get_numeric_features must return a non-empty list of strings."""
        df = clean_data(synthetic_df, sample_config)
        df = create_labels(df, sample_config)
        X, _ = get_features_and_target(df, sample_config)

        numeric_feats = get_numeric_features(X)
        assert isinstance(numeric_feats, list)
        assert len(numeric_feats) > 0
        assert all(isinstance(f, str) for f in numeric_feats)

    def test_numeric_features_subset_of_columns(self, sample_config, synthetic_df):
        """Every detected numeric feature must be an actual column of X."""
        df = clean_data(synthetic_df, sample_config)
        df = create_labels(df, sample_config)
        X, _ = get_features_and_target(df, sample_config)

        numeric_feats = get_numeric_features(X)
        for feat in numeric_feats:
            assert feat in X.columns, f"'{feat}' not in X.columns"
