"""Tests for missingness_resolution.py — one test per function."""

from __future__ import annotations

import numpy as np
import pandas as pd

import missingness_resolution as mr
from missingness_resolution import (
    add_missing_flags,
    analyze_missingness,
    drop_rows,
    mark_structural_missing,
    mice_impute,
    simple_impute,
)
from schema_infer import ColSpec


def test_ensure_dirs(tmp_output):
    figs, tabs = mr._ensure_dirs(tmp_output)
    assert figs.is_dir() and tabs.is_dir()


def test_analyze_missingness(tiny_df, tmp_output):
    tbl = analyze_missingness(tiny_df, output_root=tmp_output)
    assert "pct_missing" in tbl.columns
    assert (tmp_output / "missingness" / "tables" / "missing_per_column.csv").exists()


def test_mark_structural_missing(tiny_df):
    schema = {"a": ColSpec("a", "continuous"), "b": ColSpec("b", "continuous")}
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, np.nan, np.nan]})
    groups = {
        "n_items": {
            "cols": ["a", "b"],
            "derive_count": True,
            "derive_max": False,
            "skip_after": ["b"],
        }
    }
    out = mark_structural_missing(df, schema, groups)
    assert "n_items" in out.columns
    assert schema["b"].kind == "skip"


def test_add_missing_flags(tiny_df, tiny_schema):
    out = add_missing_flags(tiny_df, ["age"], schema=tiny_schema)
    assert "age_missing" in out.columns
    assert "age_missing" in tiny_schema


def test_drop_rows_mask():
    df = pd.DataFrame({"x": [1, 2, 3]})
    log = []
    out = drop_rows(df, mask=pd.Series([False, True, False]), reason="drop 2", log=log)
    assert len(out) == 2
    assert log[0]["n_dropped"] == 1


def test_drop_rows_where():
    df = pd.DataFrame({"age": [10, 20, 30]})
    out = drop_rows(df, where="age < 15", reason="young")
    assert len(out) == 2


def test_encode_for_impute(tiny_df, tiny_schema):
    work, decoders, cat_cols, dropped = mr._encode_for_impute(tiny_df, tiny_schema)
    assert "sex" in cat_cols or "event" in cat_cols


def test_decode_after_impute(tiny_df, tiny_schema):
    work, decoders, cat_cols, _ = mr._encode_for_impute(tiny_df, tiny_schema)
    work = work.fillna(work.median(numeric_only=True))
    out = mr._decode_after_impute(work, decoders, cat_cols, tiny_schema)
    assert len(out) == len(tiny_df)


def test_mice_impute(tiny_df, tiny_schema, tmp_output):
    frames = mice_impute(tiny_df, tiny_schema, m=2, max_iter=2, random_state=0, output_root=tmp_output)
    assert len(frames) == 2
    assert len(frames[0]) == len(tiny_df)


def test_simple_impute(tiny_df, tiny_schema):
    out = simple_impute(tiny_df, tiny_schema)
    assert out["age"].isna().sum() == 0
