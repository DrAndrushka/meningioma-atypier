"""Tests for model_validation.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_calculator import calculator_meta_to_streamlit_artifact
from model_validation import bootstrap_internal_validation, enrich_streamlit_artifact
from schema_infer import ColSpec


@pytest.fixture
def tiny_model_df() -> tuple[pd.DataFrame, list[str], dict]:
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "event": rng.integers(0, 2, n),
        "age": rng.normal(60, 10, n),
        "flag": rng.integers(0, 2, n),
    })
    design_cols = ["age", "flag"]
    coefficients = {"const": -0.5, "age": 0.05, "flag": 0.4}
    return df, design_cols, coefficients


def test_bootstrap_internal_validation(tiny_model_df):
    df, design_cols, coefficients = tiny_model_df
    out = bootstrap_internal_validation(
        df, "event", design_cols, coefficients, n_bootstrap=30,
    )
    assert out["method"] == "bootstrap internal validation"
    assert len(out["metrics"]) == 3
    assert "roc_curves" in out
    assert out["roc_curves"]["curves"][0]["fpr"]


def test_enrich_streamlit_artifact_with_ordinal_predictor():
    df = pd.DataFrame({
        "event": [0, 1, 0, 1, 0, 1],
        "age_bins": [0.0, 1.0, 2.0, 3.0, 4.0, 2.0],
    })
    design_cols = ["age_bins"]
    meta = {
        "target": "event",
        "intercept": -0.3,
        "terms": [
            {
                "name": "age_bins",
                "kind": "ordinal",
                "coef": 0.2,
                "levels": ["<50", "50-59", "60-69", "70-79", "80+"],
            },
        ],
    }
    artifact = calculator_meta_to_streamlit_artifact(meta, n=len(df), events=int(df["event"].sum()))
    enriched = enrich_streamlit_artifact(artifact, df, design_cols, n_bootstrap=20)
    assert enriched["coefficients"]["age_bins"] == pytest.approx(0.2 * enriched["coefficient_processing"]["shrinkage_factor"])


def test_enrich_streamlit_artifact_adds_validation(tiny_model_df):
    df, design_cols, coefficients = tiny_model_df
    meta = {
        "target": "event",
        "intercept": coefficients["const"],
        "terms": [
            {"name": "age", "kind": "continuous", "coef": 0.05, "z_mu": 60.0, "z_sd": 10.0},
            {"name": "flag", "kind": "binary", "coef": 0.4},
        ],
    }
    artifact = calculator_meta_to_streamlit_artifact(meta, n=len(df), events=int(df["event"].sum()))
    enriched = enrich_streamlit_artifact(artifact, df, design_cols, n_bootstrap=30)
    assert "validation" in enriched
    assert "coefficient_processing" in enriched
    assert enriched["coefficient_processing"]["shrinkage_applied"] is True
    assert "missing_data_policy" in enriched
