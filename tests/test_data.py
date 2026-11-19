"""
tests/test_data.py
Tests for src.dropguard.data: label creation, cleaning, feature extraction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dropguard.data import clean_data, create_labels, get_features_and_target


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(**overrides) -> dict:
    """Return a single-row dict with valid student data, allowing overrides."""
    base = {
        "school": "GP",
        "sex": "F",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 3,
        "Fedu": 3,
        "traveltime": 1,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "yes",
        "famsup": "no",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 1,
        "health": 3,
        "absences": 4,
        "G1": 12,
        "G2": 11,
        "G3": 10,
        "course": "math",
    }
    base.update(overrides)
    return base


def _df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------------------
# Label creation tests
# ---------------------------------------------------------------------------

class TestCreateLabels:
    def test_label_creation_at_risk(self, sample_config):
        """G3=8 → at_risk=1 (below threshold of 10)."""
        df = _df(_make_row(G3=8))
        result = create_labels(df, sample_config)
        assert result["at_risk"].iloc[0] == 1

    def test_label_creation_not_at_risk(self, sample_config):
        """G3=12 → at_risk=0 (above threshold)."""
        df = _df(_make_row(G3=12))
        result = create_labels(df, sample_config)
        assert result["at_risk"].iloc[0] == 0

    def test_label_threshold_exact(self, sample_config):
        """G3=10 → at_risk=0 (exactly at threshold, NOT at risk; <10 is at risk)."""
        df = _df(_make_row(G3=10))
        result = create_labels(df, sample_config)
        assert result["at_risk"].iloc[0] == 0

    def test_original_df_not_mutated(self, sample_config, synthetic_df):
        """create_labels must not modify the input DataFrame."""
        original_cols = set(synthetic_df.columns)
        create_labels(synthetic_df, sample_config)
        assert set(synthetic_df.columns) == original_cols

    def test_label_column_added(self, sample_config, synthetic_df):
        """at_risk column present in result."""
        result = create_labels(synthetic_df, sample_config)
        assert "at_risk" in result.columns

    def test_all_labels_binary(self, sample_config, synthetic_df):
        """All at_risk values are 0 or 1."""
        result = create_labels(synthetic_df, sample_config)
        assert set(result["at_risk"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Feature / target extraction tests
# ---------------------------------------------------------------------------

class TestGetFeaturesAndTarget:
    def test_g3_not_in_features(self, sample_config, synthetic_df):
        """get_features_and_target must exclude G3 from X."""
        df = create_labels(synthetic_df, sample_config)
        X, y = get_features_and_target(df, sample_config)
        assert "G3" not in X.columns

    def test_g1_g2_excluded_in_early_warning(self, sample_config, synthetic_df):
        """include_grades=False → G1 and G2 must not appear in X."""
        df = create_labels(synthetic_df, sample_config)
        X, y = get_features_and_target(df, sample_config, include_grades=False)
        assert "G1" not in X.columns
        assert "G2" not in X.columns

    def test_g1_g2_present_by_default(self, sample_config, synthetic_df):
        """Default call (include_grades=True) includes G1 and G2."""
        df = create_labels(synthetic_df, sample_config)
        X, y = get_features_and_target(df, sample_config, include_grades=True)
        assert "G1" in X.columns
        assert "G2" in X.columns

    def test_target_matches_at_risk(self, sample_config, synthetic_df):
        """y Series must equal the at_risk column."""
        df = create_labels(synthetic_df, sample_config)
        X, y = get_features_and_target(df, sample_config)
        pd.testing.assert_series_equal(
            y.reset_index(drop=True),
            df["at_risk"].reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Data cleaning tests
# ---------------------------------------------------------------------------

class TestCleanData:
    def test_no_missing_g3_after_clean(self, sample_config, synthetic_df):
        """clean_data drops rows where G3 is NaN."""
        dirty = synthetic_df.copy()
        dirty.loc[0, "G3"] = np.nan
        dirty.loc[3, "G3"] = np.nan
        result = clean_data(dirty, sample_config)
        assert result["G3"].isna().sum() == 0
        assert len(result) == len(synthetic_df) - 2

    def test_binary_encoding(self, sample_config):
        """'yes'/'no' columns are encoded as 1/0."""
        df = _df(_make_row(schoolsup="yes", famsup="no"))
        result = clean_data(df, sample_config)
        # At least one binary column should be numeric after cleaning
        binary_cols = [
            c for c in result.columns
            if result[c].dropna().isin([0, 1]).all()
            and c not in ("Medu", "Fedu", "traveltime", "studytime", "failures",
                          "famrel", "freetime", "goout", "Dalc", "Walc", "health",
                          "absences", "age", "G1", "G2", "G3")
        ]
        assert len(binary_cols) > 0, "Expected at least one binary-encoded column"

    def test_yes_encoded_as_1(self, sample_config):
        """'yes' in schoolsup should become 1."""
        df = _df(_make_row(schoolsup="yes"))
        result = clean_data(df, sample_config)
        assert result["schoolsup"].iloc[0] == 1

    def test_no_encoded_as_0(self, sample_config):
        """'no' in schoolsup should become 0."""
        df = _df(_make_row(schoolsup="no"))
        result = clean_data(df, sample_config)
        assert result["schoolsup"].iloc[0] == 0
