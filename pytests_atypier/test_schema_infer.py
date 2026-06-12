"""Tests for schema_infer.py — type inference and categorical order."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import schema_infer as si
from schema_infer import ColSpec, export_schema_summary, infer_schema, print_schema_template, schema_summary


def test_looks_binary():
    assert si._looks_binary(pd.Series([True, False, True]))
    assert not si._looks_binary(pd.Series([1, 2, 3]))


def test_looks_datetime():
    s = pd.Series(pd.date_range("2018-01-01", periods=20, freq="ME"))
    assert si._looks_datetime(s)


def test_looks_id():
    s = pd.Series([f"id{i}" for i in range(100)], name="patient_id")
    assert si._looks_id(s, 100)
    assert not si._looks_id(pd.Series([1, 1, 2, 2]), 4)


def test_infer_one():
    assert si._infer_one(pd.Series(np.arange(20, dtype=float)), 20, 15) == "continuous"
    assert si._infer_one(pd.Series([True, False]), 2, 15) == "binary"


def test_infer_schema_preserves_ordered_categorical_levels():
    df = pd.DataFrame({
        "age_bins": pd.Categorical(
            ["60-69", "<50", "80+"],
            categories=["<50", "50-59", "60-69", "70-79", "80+"],
            ordered=True,
        ),
        "ki67_group": pd.Categorical(
            ["high_ge_10", "low_le_4", "low_le_4"],
            categories=["low_le_4", "intermediate_5_9", "high_ge_10"],
            ordered=True,
        ),
    })
    schema = infer_schema(df)
    assert schema["age_bins"].ordered_levels == ["<50", "50-59", "60-69", "70-79", "80+"]
    assert schema["ki67_group"].ordered_levels == [
        "low_le_4", "intermediate_5_9", "high_ge_10",
    ]


def test_infer_schema(tiny_df):
    schema = infer_schema(tiny_df)
    assert "age" in schema
    assert schema["event"].kind == "binary"


def test_print_schema_template(tiny_df, capsys):
    schema = infer_schema(tiny_df)
    print_schema_template(schema)
    text = capsys.readouterr().out
    assert "ColSpec" in text
    assert "schema_overrides" in text


def test_schema_summary(tiny_df):
    schema = infer_schema(tiny_df)
    tbl = schema_summary(schema)
    assert list(tbl.columns) == ["column", "kind", "keep", "datetime_bin", "levels", "nulls", "note"]
    grade = tbl.loc[tbl["column"] == "grade", "levels"].iloc[0]
    assert grade == [1, 2]


def test_schema_summary_nominal_levels_from_replace():
    schema = {
        "episode": ColSpec(
            "episode", "nominal",
            replace={"0": "primary", "1": "recurrent"},
        ),
    }
    tbl = schema_summary(schema)
    assert tbl.loc[0, "levels"] == ["primary", "recurrent"]


def test_print_column_uniques(tiny_df, capsys):
    schema = infer_schema(tiny_df)
    si.print_column_uniques(tiny_df, schema)
    text = capsys.readouterr().out
    assert "Column uniques" in text
    assert "event" in text


def test_export_schema_summary(tiny_df, tmp_output):
    schema = infer_schema(tiny_df)
    path = export_schema_summary(schema, tmp_output)
    assert path.exists()
    assert path.name == "schema_summary.csv"
