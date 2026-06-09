"""Tests for diagnostic_accuracy.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from diagnostic_accuracy import (
    binary_diagnostic_metrics,
    screen_diagnostic_accuracy,
)
from schema_infer import ColSpec


def _tiny_binary_df() -> pd.DataFrame:
    return pd.DataFrame({
        "high_grade": [True, True, False, False, True, False],
        "necrosis": [True, False, True, False, False, False],
        "age": [50.0, 60.0, 70.0, 80.0, 55.0, 65.0],
        "tumor_location": [
            "non_skull_base", "skull_base", "non_skull_base",
            "skull_base", "non_skull_base", "skull_base",
        ],
        "sex": ["male", "female", "male", "female", "male", "female"],
    })


def _tiny_schema() -> dict[str, ColSpec]:
    return {
        "high_grade": ColSpec("high_grade", "binary"),
        "necrosis": ColSpec("necrosis", "binary"),
        "age": ColSpec("age", "continuous"),
        "tumor_location": ColSpec("tumor_location", "nominal"),
        "sex": ColSpec("sex", "nominal"),
    }


def test_binary_diagnostic_metrics_counts():
    df = _tiny_binary_df()
    m = binary_diagnostic_metrics(df, "high_grade", "necrosis", positive_class=True)
    assert m["n_used"] == 6
    assert m["TP"] == 1
    assert m["FP"] == 1
    assert m["FN"] == 2
    assert m["TN"] == 2
    assert m["sensitivity"] == 1 / 3
    assert m["specificity"] == 2 / 3
    assert m["accuracy"] == 0.5
    assert abs(m["AUC"] - 0.5) < 1e-9
    assert not np.isnan(m["p"])
    assert 0 <= m["sensitivity_lo"] <= m["sensitivity_hi"] <= 1
    assert 0 <= m["accuracy_lo"] <= m["accuracy_hi"] <= 1


def test_binary_diagnostic_metrics_dropna():
    df = _tiny_binary_df()
    df["necrosis"] = df["necrosis"].astype(object)
    df.loc[0, "necrosis"] = np.nan
    m = binary_diagnostic_metrics(df, "high_grade", "necrosis", positive_class=True)
    assert m["n_used"] == 5


def test_derived_nominal_contrast():
    df = _tiny_binary_df()
    from diagnostic_accuracy import _derived_series

    series = _derived_series(df, "tumor_location_non_skull_base")
    m = binary_diagnostic_metrics(
        df, "high_grade", "tumor_location_non_skull_base",
        positive_class=True, predictor_series=series,
    )
    assert m["n_used"] == 6
    assert m["note"] == ""


def test_screen_diagnostic_accuracy_writes_csv(tmp_output):
    df = _tiny_binary_df()
    schema = _tiny_schema()
    out = screen_diagnostic_accuracy(
        df,
        schema,
        targets=["high_grade"],
        predictors=["necrosis", "age", "tumor_location", "sex"],
        positive_class={"high_grade": True},
        output_root=tmp_output,
    )
    assert (tmp_output / "eda" / "tables" / "diagnostic_accuracy.csv").exists()
    necrosis = out[(out["predictor"] == "necrosis") & (out["note"].eq(""))]
    assert len(necrosis) == 1
    age = out[out["predictor"] == "age"].iloc[0]
    assert age["note"] == "Skipped: requires predefined cutoff"
    derived = out[out["predictor"] == "tumor_location_non_skull_base"]
    assert len(derived) == 1
    assert derived.iloc[0]["note"] == ""
    sex_male = out[out["predictor"] == "sex_male"]
    assert len(sex_male) == 1


def test_screen_skips_non_binary_target(tmp_output):
    df = pd.DataFrame({"grade": [1, 2, 3], "necrosis": [True, False, True]})
    schema = {
        "grade": ColSpec("grade", "ordinal", ordered_levels=[1, 2, 3]),
        "necrosis": ColSpec("necrosis", "binary"),
    }
    out = screen_diagnostic_accuracy(
        df, schema, targets=["grade"], predictors=["necrosis"],
        output_root=tmp_output,
    )
    assert out.empty
