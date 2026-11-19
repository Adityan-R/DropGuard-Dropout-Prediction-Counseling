"""
Shared pytest fixtures for the DropGuard test suite.
"""
from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Minimal config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config() -> dict:
    """Return a minimal config dict sufficient for all unit tests."""
    return {
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
        },
        "label": {
            "at_risk_threshold": 10,
            "target_col": "at_risk",
            "exclude_features": ["G3"],
        },
        "training": {
            "random_seed": 42,
            "test_size": 0.2,
            "cv_folds": 5,
        },
        "paths": {
            "models_dir": "models",
            "reports_dir": "reports",
        },
        "models": {
            "logistic": {"C": [1], "max_iter": 1000},
            "random_forest": {
                "n_estimators": [100],
                "max_depth": [None],
                "min_samples_split": [2],
            },
            "xgboost": {
                "n_estimators": [100],
                "max_depth": [3],
                "learning_rate": [0.1],
            },
        },
    }


# ---------------------------------------------------------------------------
# Synthetic student DataFrame fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Return a 10-row synthetic student DataFrame with real column names."""
    n = 10
    data = {
        "school": ["GP"] * n,
        "sex": ["F", "M", "F", "M", "F", "M", "F", "M", "F", "M"],
        "age": [17, 18, 16, 17, 18, 16, 17, 18, 16, 17],
        "address": ["U", "R", "U", "U", "R", "U", "R", "U", "U", "R"],
        "famsize": ["GT3"] * n,
        "Pstatus": ["T", "A", "T", "T", "A", "T", "T", "A", "T", "T"],
        "Medu": [4, 3, 2, 1, 4, 3, 2, 1, 4, 3],
        "Fedu": [4, 3, 2, 1, 4, 3, 2, 1, 4, 3],
        "traveltime": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "studytime": [2, 1, 3, 2, 1, 3, 2, 1, 3, 2],
        "failures": [0, 1, 0, 2, 0, 1, 0, 1, 0, 0],
        "schoolsup": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"],
        "famsup": ["no", "yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes"],
        "paid": ["no", "yes", "no", "no", "yes", "no", "yes", "no", "no", "yes"],
        "activities": ["yes", "no", "yes", "yes", "no", "yes", "no", "yes", "yes", "no"],
        "nursery": ["yes"] * n,
        "higher": ["yes", "yes", "yes", "no", "yes", "yes", "yes", "no", "yes", "yes"],
        "internet": ["yes", "no", "yes", "yes", "no", "yes", "yes", "no", "yes", "yes"],
        "romantic": ["no", "no", "yes", "no", "yes", "no", "no", "yes", "no", "no"],
        "famrel": [4, 3, 5, 4, 3, 4, 5, 3, 4, 4],
        "freetime": [3, 4, 3, 2, 4, 3, 2, 4, 3, 3],
        "goout": [3, 4, 2, 3, 5, 2, 3, 4, 2, 3],
        "Dalc": [1, 2, 1, 1, 2, 1, 1, 2, 1, 1],
        "Walc": [1, 3, 2, 1, 3, 2, 1, 3, 2, 1],
        "health": [3, 4, 3, 2, 5, 3, 2, 4, 3, 3],
        "absences": [4, 0, 2, 6, 0, 2, 4, 8, 0, 2],
        "G1": [15, 8, 12, 7, 14, 11, 9, 6, 13, 10],
        "G2": [14, 7, 11, 8, 13, 10, 8, 7, 12, 11],
        "G3": [14, 8, 12, 7, 13, 11, 9, 6, 13, 10],
        "course": [
            "math", "portuguese", "math", "portuguese", "math",
            "portuguese", "math", "portuguese", "math", "portuguese",
        ],
    }
    return pd.DataFrame(data)
