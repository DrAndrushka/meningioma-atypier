"""Tests for cleaning.py — schema application, derivations, export."""

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


def test_format_table_for_display():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": ["x", None]})
    out = cl.format_table_for_display(df)
    assert out.loc[1, "a"] == ""
    assert out.loc[1, "b"] == ""


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


def test_schema_coercion_table(tmp_output):
    df = pd.DataFrame({
        "vol": ["1.5", "NAV SECTRA - NOSŪTĪTS", "2,1", None, "3"],
        "flag": ["yes", "no", "maybe", "1", "0"],
    })
    schema = {
        "vol": ColSpec("vol", "continuous"),
        "flag": ColSpec("flag", "binary"),
    }
    coercion: list[dict] = []
    out = apply_schema(
        df, schema, output_root=tmp_output, coercion_log=coercion,
    )
    path = tmp_output / "cleaning" / "schema_coercion.csv"
    assert path.exists()
    table = pd.read_csv(path)
    assert list(table.columns) == [
        "column", "kind", "value_before", "value_after", "n", "n_after",
    ]
    # sentinel → missing (kept as its own row)
    nav = table[
        (table["column"] == "vol")
        & (table["value_before"] == "NAV SECTRA - NOSŪTĪTS")
        & (table["value_after"] == "(missing)")
    ]
    assert len(nav) == 1 and int(nav.iloc[0]["n"]) == 1
    # comma decimal folds into a style summary row
    folded = table[table["value_before"].astype(str).str.startswith("comma decimal")]
    assert len(folded) == 1
    assert "vol" in str(folded.iloc[0]["column"])
    assert "dot decimal" in str(folded.iloc[0]["value_after"])
    assert int(folded.iloc[0]["n"]) >= 1
    # binary unknown → missing
    assert any(
        r["column"] == "flag" and r["value_before"] == "maybe" and r["value_after"] == "(missing)"
        for r in coercion
    )
    assert out["vol"].isna().sum() == 2  # sentinel + None


def test_schema_coercion_datetime_style_collapse(tmp_output):
    df = pd.DataFrame({
        "mri_date": [
            "05.09.2024.", "05.09.2024.", "13.02.2018", "NAV MRI", None,
        ],
    })
    schema = {"mri_date": ColSpec("mri_date", "datetime", datetime_bin="full")}
    coercion: list[dict] = []
    apply_schema(df, schema, output_root=tmp_output, coercion_log=coercion)
    table = pd.read_csv(tmp_output / "cleaning" / "schema_coercion.csv")
    # Many concrete dates collapse to style → style
    dotted = table[
        (table["value_before"] == "DD.MM.YYYY.")
        & (table["value_after"] == "YYYY-MM-DD 00:00:00")
    ]
    assert len(dotted) == 1 and int(dotted.iloc[0]["n"]) == 2
    plain = table[
        (table["value_before"] == "DD.MM.YYYY")
        & (table["value_after"] == "YYYY-MM-DD 00:00:00")
    ]
    assert len(plain) == 1 and int(plain.iloc[0]["n"]) == 1
    # Sentinels stay literal
    nav = table[
        (table["value_before"] == "NAV MRI")
        & (table["value_after"] == "(missing)")
    ]
    assert len(nav) == 1
    # Not one row per concrete timestamp
    assert len(table) <= 3


def test_schema_coercion_numeric_format_collapse(tmp_output):
    df = pd.DataFrame({
        "vol": ["01", "1.10", "2,50", "bad", "3.0"],
        "n_lesions": ["02", "3", "04", "1", "05"],
    })
    schema = {
        "vol": ColSpec("vol", "continuous"),
        "n_lesions": ColSpec("n_lesions", "count"),
    }
    apply_schema(df, schema, output_root=tmp_output)
    table = pd.read_csv(tmp_output / "cleaning" / "schema_coercion.csv")
    miss = table[
        (table["column"] == "vol") & (table["value_after"] == "(missing)")
    ]
    assert len(miss) == 1
    assert miss.iloc[0]["value_before"] == "bad"

    leading = table[table["value_before"].astype(str).str.startswith("leading-zero integer")]
    assert len(leading) == 1
    assert "integer" in str(leading.iloc[0]["value_after"])
    assert "e.g." in str(leading.iloc[0]["value_before"])
    assert int(leading.iloc[0]["n"]) >= 2

    trailing = table[table["value_before"].astype(str).str.startswith("trailing-zero decimal")]
    assert len(trailing) >= 1
    assert "e.g." in str(trailing.iloc[0]["value_after"])

    comma = table[table["value_before"].astype(str).str.startswith("comma decimal")]
    assert len(comma) == 1
    assert "dot decimal" in str(comma.iloc[0]["value_after"])

    # No leftover raw per-value format rows for these columns
    leftover = table[
        table["column"].isin(["vol", "n_lesions"])
        & ~table["value_after"].eq("(missing)")
        & ~table["value_before"].astype(str).str.contains(r"e\.g\.", regex=True)
    ]
    assert leftover.empty


def test_schema_coercion_id_collapse(tmp_output):
    df = pd.DataFrame({
        "pid": [1.0, 2.0, "NAV", 3.0],
        "sid": [10.0, 20.0, 30.0, 40.0],
    })
    schema = {
        "pid": ColSpec("pid", "id", nulls=["NAV"]),
        "sid": ColSpec("sid", "id"),
    }
    apply_schema(df, schema, output_root=tmp_output)
    table = pd.read_csv(tmp_output / "cleaning" / "schema_coercion.csv")
    id_miss = table[
        (table["kind"] == "id") & (table["value_after"] == "(missing)")
    ]
    assert len(id_miss) == 1
    assert id_miss.iloc[0]["column"] == "pid"
    assert id_miss.iloc[0]["value_before"] == "NAV"
    folded = table[
        (table["kind"] == "id")
        & (table["value_before"] == "(various)")
        & (table["value_after"] == "(string)")
    ]
    assert len(folded) == 1
    assert "pid" in folded.iloc[0]["column"]
    assert "sid" in folded.iloc[0]["column"]
    assert int(folded.iloc[0]["n"]) >= 1
    # No per-id value rows left (only missing + one fold row)
    assert len(table[table["kind"] == "id"]) == 2


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
    assert bin_datetime(s, unit="day").tolist() == ["2018-03-01", "2019-07-15"]


def test_apply_schema_datetime_bin(tiny_df):
    schema = {
        "entry_year": ColSpec("entry_year", "datetime", keep=False, datetime_bin="year"),
    }
    out = apply_schema(tiny_df[["entry_year"]].copy(), schema)
    assert list(out.columns) == ["entry_year"]
    assert out["entry_year"].tolist() == [2018, 2019, 2020, 2021]
    assert schema["entry_year"].kind == "ordinal"
    assert schema["entry_year"].ordered_levels == [2018, 2019, 2020, 2021]


def test_apply_schema_datetime_bin_full(tiny_df):
    schema = {
        "entry_year": ColSpec("entry_year", "datetime", datetime_bin="full"),
    }
    out = apply_schema(tiny_df[["entry_year"]].copy(), schema)
    assert list(out.columns) == ["entry_year"]
    # Dates-only 'full' keeps a true datetime dtype (analysed as datetime,
    # not demoted to ordinal date categories); time-of-day is stripped.
    assert schema["entry_year"].kind == "datetime"
    assert pd.api.types.is_datetime64_any_dtype(out["entry_year"])
    assert out["entry_year"].dt.strftime("%Y-%m-%d").tolist() == [
        "2018-01-01", "2019-06-01", "2020-03-01", "2021-01-01",
    ]
    assert (out["entry_year"].dt.normalize() == out["entry_year"]).all()


def test_apply_schema_datetime_bin_full_with_time():
    df = pd.DataFrame({"mri_date": ["2018-03-01 14:30:00", "2019-07-15 09:00:00"]})
    schema = {"mri_date": ColSpec("mri_date", "datetime", datetime_bin="full")}
    out = apply_schema(df, schema)
    assert pd.api.types.is_datetime64_any_dtype(out["mri_date"])
    assert schema["mri_date"].kind == "datetime"
    assert out["mri_date"].dt.hour.tolist() == [14, 9]


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
        n_columns_raw=5,
        n_columns_after_schema=4,
        schema=tiny_schema,
        drop_log=[{"reason": "test", "criterion": "x", "n_before": 10, "n_remaining": 9, "n_dropped": 1}],
        dupes=None,
    )
    assert list(tbl.columns) == ["step", "detail", "n_rows", "n_columns", "criterion"]
    assert tbl.iloc[-1]["step"] == "final"
    drop_row = tbl.loc[tbl["step"] == "drop_rows"].iloc[0]
    assert drop_row["n_rows"] == 9
    assert drop_row["n_columns"] == 4
    assert drop_row["criterion"] == "x"
    assert "n_rows_before" not in tbl.columns
    assert "n_dropped" not in tbl.columns


@pytest.mark.parametrize("dupes, expected", [
    (None, "no duplicates found"),
    (pd.DataFrame({"id": [1, 1]}), "2 row(s) in duplicate ID groups (flagged, not removed)"),
])
def test_build_cleaning_summary_duplicate_row_always_present(tiny_schema, dupes, expected):
    tbl = cl._build_cleaning_summary(
        n_rows_raw=10,
        n_rows_after_schema=10,
        n_rows_final=10,
        n_columns_raw=5,
        n_columns_after_schema=4,
        schema=tiny_schema,
        drop_log=None,
        dupes=dupes,
    )
    dup_row = tbl.loc[tbl["step"] == "duplicate_audit"]
    assert len(dup_row) == 1
    assert dup_row.iloc[0]["detail"] == expected


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
