"""Tests for schema_infer.py — one test per function."""

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


def test_infer_schema(tiny_df):
    schema = infer_schema(tiny_df)
    assert "age" in schema
    assert schema["event"].kind == "binary"


def test_print_schema_template(tiny_df):
    schema = infer_schema(tiny_df)
    text = print_schema_template(schema)
    assert "ColSpec" in text
    assert "schema_overrides" in text


def test_schema_summary(tiny_df):
    schema = infer_schema(tiny_df)
    tbl = schema_summary(schema)
    assert list(tbl.columns) == ["column", "kind", "keep", "ordered_levels", "nulls", "note"]


def test_export_schema_summary(tiny_df, tmp_output):
    schema = infer_schema(tiny_df)
    path = export_schema_summary(schema, tmp_output)
    assert path.exists()
    assert path.name == "schema_summary.csv"
