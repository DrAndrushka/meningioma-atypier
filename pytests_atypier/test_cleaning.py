"""Tests for cleaning.py — one test per function."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cleaning as cl
from cleaning import (
    apply_schema,
    audit_duplicates,
    bin_datetime,
    bin_numeric,
    combine_categories,
    export_cleaning_artifacts,
    write_cleaned_csv,
    format_number,
    format_table_for_csv,
    make_missing_flag,
    zscore,
)
from schema_infer import ColSpec


def test_classify_column():
    assert cl._classify_column("p_value") == "pvalue"
    assert cl._classify_column("n_rows") == "count"
    assert cl._classify_column("missing_pct") == "percent"
    assert cl._classify_column("mean_age") == "central"
    assert cl._classify_column("other") == "default"


def test_format_value():
    assert cl._format_value(0.0005, "pvalue") == "<0.001"
    assert cl._format_value(5.0, "count") == 5


def test_format_number():
    assert format_number(1.23456, "default") == 1.235


def test_format_table_for_csv():
    df = pd.DataFrame({"p": [0.0001, 0.05], "n": [10.0, 20.0]})
    out = format_table_for_csv(df)
    assert out.loc[0, "p"] == "<0.001"
    assert out.loc[0, "n"] == 10


def test_apply_schema(tiny_df, tiny_schema):
    log: list[dict] = []
    out = apply_schema(tiny_df, tiny_schema, log=log)
    assert "note" in out.columns
    assert pd.api.types.is_float_dtype(out["age"])
    id_row = next(r for r in log if r["column"] == "id")
    assert "excluded_keep_false" in id_row["action"]


def test_coerce_binary():
    s = pd.Series(["yes", "no", "1", "0"])
    out = cl._coerce_binary(s)
    assert out.tolist() == [True, False, True, False]


def test_audit_duplicates():
    df = pd.DataFrame({"id": ["A", "A", "B"], "v": [1, 2, 3]})
    audit, cleaned = audit_duplicates(df, ["id"])
    assert len(audit) == 2
    assert len(cleaned) == 3


def test_bin_numeric():
    s = pd.Series([25, 55, 75])
    out = bin_numeric(s, [0, 50, 70, 100], labels=["<50", "50-69", "70+"])
    assert str(out[1]) == "50-69"


def test_bin_datetime():
    s = pd.Series(pd.to_datetime(["2018-03-01", "2019-07-15"]))
    assert bin_datetime(s, unit="year").tolist() == [2018, 2019]


def test_make_missing_flag():
    s = pd.Series([1.0, np.nan, 3.0], name="age")
    flag = make_missing_flag(s)
    assert flag.name == "age_missing"
    assert flag.tolist() == [False, True, False]


def test_combine_categories():
    s = pd.Series(["a", "b", "c"])
    out = combine_categories(s, {"a": "A", "b": "A"}, other="other")
    assert set(out.astype(str)) == {"A", "other"}


def test_zscore():
    s = pd.Series([1.0, 2.0, 3.0])
    z = zscore(s)
    assert abs(z.mean()) < 1e-10


def test_build_cleaning_summary(tiny_schema):
    tbl = cl._build_cleaning_summary(
        n_rows_raw=10,
        n_rows_after_schema=10,
        n_rows_final=9,
        schema=tiny_schema,
        drop_log=[{"reason": "test", "criterion": "x", "n_remaining": 9, "n_dropped": 1}],
        dupes=None,
    )
    assert tbl.iloc[-1]["step"] == "final"


def test_build_cleaning_log():
    log = cl._build_cleaning_log(
        schema_log=[{"step": "apply_schema", "column": "age", "action": "coerce", "kind": "continuous"}],
        drop_log=None,
        dupes=None,
    )
    assert not log.empty


def test_columns_for_cleaned_export(tiny_df, tiny_schema):
    cols = cl.columns_for_cleaned_export(tiny_df, tiny_schema)
    assert "id" not in cols
    assert "age" in cols


def test_write_cleaned_csv(tiny_df, tiny_schema, tmp_output):
    tiny_df = tiny_df.copy()
    tiny_df["derived"] = 1
    tiny_schema["derived"] = ColSpec("derived", "binary")
    path = write_cleaned_csv(tmp_output, tiny_df, tiny_schema)
    saved = pd.read_csv(path)
    assert "derived" in saved.columns


def test_export_cleaning_artifacts(tiny_df, tiny_schema, tmp_output):
    paths = export_cleaning_artifacts(
        tmp_output,
        df=tiny_df,
        n_rows_raw=10,
        n_rows_after_schema=10,
        n_rows_final=10,
        schema=tiny_schema,
    )
    assert paths["summary"].exists()
    assert paths["cleaned"].exists()
    saved = pd.read_csv(paths["cleaned"])
    assert "id" not in saved.columns
    assert "age" in saved.columns
