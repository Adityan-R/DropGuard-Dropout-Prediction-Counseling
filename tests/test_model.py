"""
tests/test_model.py
Tests for the trained model: prediction shape, probability range, and fitted state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.dropguard.data import clean_data, create_labels, get_features_and_target
from src.dropguard.features import build_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trained_pipeline(sample_config, synthetic_df):
    """Return a small LogisticRegression pipeline fitted on synthetic data."""
    df = clean_data(synthetic_df, sample_config)
    df = create_labels(df, sample_config)
    X, y = get_features_and_target(df, sample_config)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = build_pipeline(clf, X)
    pipeline.fit(X, y)
    return pipeline, X, y


# ---------------------------------------------------------------------------
# Prediction shape tests
# ---------------------------------------------------------------------------

class TestPredictionShape:
    def test_predict_shape_matches_input(self, trained_pipeline):
        """model.predict(X) must return an array of length n_samples."""
        pipeline, X, _ = trained_pipeline
        preds = pipeline.predict(X)
        assert preds.shape == (len(X),), (
            f"Expected shape ({len(X)},), got {preds.shape}"
        )

    def test_predict_proba_shape(self, trained_pipeline):
        """predict_proba must return shape (n_samples, 2) for binary classification."""
        pipeline, X, _ = trained_pipeline
        proba = pipeline.predict_proba(X)
        assert proba.shape == (len(X), 2), (
            f"Expected shape ({len(X)}, 2), got {proba.shape}"
        )

    def test_single_row_predict_shape(self, trained_pipeline):
        """Prediction on a single row must return shape (1,)."""
        pipeline, X, _ = trained_pipeline
        single_row = X.iloc[:1]
        preds = pipeline.predict(single_row)
        assert preds.shape == (1,)


# ---------------------------------------------------------------------------
# Probability range tests
# ---------------------------------------------------------------------------

class TestProbabilityRange:
    def test_probabilities_in_unit_interval(self, trained_pipeline):
        """All predict_proba values must be in [0, 1]."""
        pipeline, X, _ = trained_pipeline
        proba = pipeline.predict_proba(X)
        assert np.all(proba >= 0.0), "Some probabilities are negative."
        assert np.all(proba <= 1.0), "Some probabilities exceed 1."

    def test_probabilities_sum_to_one(self, trained_pipeline):
        """Each row's class probabilities must sum to 1."""
        pipeline, X, _ = trained_pipeline
        proba = pipeline.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, atol=1e-6,
            err_msg="predict_proba rows do not sum to 1."
        )

    def test_at_risk_proba_column_in_range(self, trained_pipeline):
        """The at-risk class probability column ([:,1]) must be in [0, 1]."""
        pipeline, X, _ = trained_pipeline
        at_risk_proba = pipeline.predict_proba(X)[:, 1]
        assert at_risk_proba.min() >= 0.0
        assert at_risk_proba.max() <= 1.0


# ---------------------------------------------------------------------------
# Fitted-model tests
# ---------------------------------------------------------------------------

class TestModelFitted:
    def test_model_has_predict(self, trained_pipeline):
        """The returned pipeline must expose a predict method."""
        pipeline, _, _ = trained_pipeline
        assert hasattr(pipeline, "predict"), "Pipeline missing predict method."

    def test_model_has_predict_proba(self, trained_pipeline):
        """The returned pipeline must expose a predict_proba method."""
        pipeline, _, _ = trained_pipeline
        assert hasattr(pipeline, "predict_proba"), "Pipeline missing predict_proba method."

    def test_binary_predictions(self, trained_pipeline):
        """All predictions must be 0 or 1."""
        pipeline, X, _ = trained_pipeline
        preds = pipeline.predict(X)
        unique = set(np.unique(preds))
        assert unique.issubset({0, 1}), f"Unexpected prediction values: {unique}"

    def test_pipeline_has_preprocessor_step(self, trained_pipeline):
        """Pipeline must contain a 'preprocessor' step."""
        pipeline, _, _ = trained_pipeline
        assert "preprocessor" in pipeline.named_steps

    def test_pipeline_has_classifier_step(self, trained_pipeline):
        """Pipeline must contain a 'classifier' step."""
        pipeline, _, _ = trained_pipeline
        assert "classifier" in pipeline.named_steps

    def test_joblib_roundtrip(self, trained_pipeline, tmp_path):
        """Model must survive a joblib save/load cycle with identical predictions."""
        import joblib
        pipeline, X, _ = trained_pipeline
        model_path = tmp_path / "test_model.joblib"
        joblib.dump(pipeline, model_path)
        loaded = joblib.load(model_path)
        np.testing.assert_array_equal(
            pipeline.predict(X),
            loaded.predict(X),
            err_msg="Predictions differ after joblib round-trip."
        )
