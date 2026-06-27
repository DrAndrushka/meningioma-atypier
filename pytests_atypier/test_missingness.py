"""Tests for missingness_resolution.py — MICE and dtype restore."""

from __future__ import annotations

import numpy as np
import pandas as pd

import missingness_resolution as mr
from missingness_resolution import (
    add_missing_flags,
    analyze_missingness,
    drop_rows,
    imputation_audit,
    mark_structural_missing,
    mice_impute,
    prepare_datasets_dir,
    simple_impute,
    simple_impute_stage,
    stage_unimputed_dataset,
    load_unimputed_dataset,
    load_modeling_frames,
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
    out = mr._decode_after_impute(
        work, decoders, cat_cols, tiny_schema,
        original=tiny_df, rng=np.random.default_rng(0),
    )
    assert len(out) == len(tiny_df)


def test_mice_impute(tiny_df, tiny_schema, tmp_output):
    frames = mice_impute(tiny_df, tiny_schema, m=2, max_iter=2, random_state=0, output_root=tmp_output)
    assert len(frames) == 2
    assert len(frames[0]) == len(tiny_df)
    mice_dir = tmp_output / "missingness" / "mice"
    assert (mice_dir / "manifest.json").exists()
    assert (mice_dir / "imputed_001.parquet").exists()
    assert (mice_dir / "imputed_002.parquet").exists()
    datasets_dir = tmp_output / "datasets"
    assert (datasets_dir / "unimputed_df.parquet").exists()
    assert (datasets_dir / "mice_imputed_df.parquet").exists()
    assert not (datasets_dir / "simple_imputed_df.parquet").exists()
    assert mr.load_unimputed_dataset(tmp_output).shape == tiny_df.shape
    assert len(mr.load_modeling_frames(tmp_output)) == 2


def test_save_load_imputed_frames_roundtrip(tiny_df, tmp_output):
    df = tiny_df.copy()
    df["age"] = df["age"].astype("Float64")
    df["sex"] = pd.Categorical(df["sex"], categories=["F", "M"], ordered=False)
    df["grade"] = pd.Categorical(df["grade"], categories=[1, 2, 3], ordered=True)
    df["event"] = df["event"].astype("boolean")
    frames = [df.copy(), df.copy()]
    mr.save_imputed_frames(frames, tmp_output, source_df=df)
    loaded = mr.load_imputed_frames(tmp_output)
    assert len(loaded) == 2
    mr._assert_frame_dtypes_match(df, loaded[0], context="load roundtrip")
    assert list(loaded[0].columns) == list(df.columns)
    assert len(loaded[0]) == len(df)


def test_imputed_parquet_preserves_pipeline_dtypes(tmp_output):
    df = pd.DataFrame({
        "age": pd.array([45.0, 55.0, np.nan], dtype="Float64"),
        "sex": pd.Categorical(["M", "F", "M"], categories=["F", "M"], ordered=False),
        "grade": pd.Categorical([1, 2, 1], categories=[1, 2, 3], ordered=True),
        "event": pd.Series([True, False, True], dtype="boolean"),
        "entry_year": pd.to_datetime(["2018-01-01", "2019-06-01", "2020-03-01"]),
        "note": pd.array(["x", "y", "z"], dtype="string"),
    })
    mr.save_imputed_frames([df], tmp_output, source_df=df)
    loaded = mr.load_imputed_frames(tmp_output)[0]
    mr._assert_frame_dtypes_match(df, loaded, context="pipeline dtypes")


def test_prepare_datasets_dir_overwrites(tmp_output):
    datasets_dir = tmp_output / "datasets"
    datasets_dir.mkdir(parents=True)
    stale = datasets_dir / "stale.txt"
    stale.write_text("old", encoding="utf-8")
    mr.prepare_datasets_dir(tmp_output)
    assert datasets_dir.is_dir()
    assert not stale.exists()


def test_simple_impute_stage(tiny_df, tiny_schema, tmp_output):
    out = simple_impute_stage(
        tiny_df, tiny_schema, tmp_output, impute_binary=False,
    )
    assert out["age"].isna().sum() == 0
    datasets_dir = tmp_output / "datasets"
    assert (datasets_dir / "unimputed_df.parquet").exists()
    assert (datasets_dir / "simple_imputed_df.parquet").exists()
    assert not (datasets_dir / "mice_imputed_df.parquet").exists()
    assert not (tmp_output / "missingness" / "mice").exists()
    assert len(mr.load_modeling_frames(tmp_output)) == 1


def test_simple_impute(tiny_df, tiny_schema):
    out = simple_impute(tiny_df, tiny_schema)
    assert out["age"].isna().sum() == 0


def test_simple_impute_binary_left_nan_by_default(tiny_df, tiny_schema):
    df = tiny_df.copy()
    df["event"] = df["event"].astype(object)
    df.loc[0, "event"] = np.nan
    out = simple_impute(df, tiny_schema)
    assert out["event"].isna().sum() == 1


def test_simple_impute_binary_mode_when_requested(tiny_df, tiny_schema):
    df = tiny_df.copy()
    df["event"] = df["event"].astype(object)
    df.loc[0, "event"] = np.nan
    out = simple_impute(df, tiny_schema, impute_binary=True)
    assert out["event"].isna().sum() == 0


def test_imputation_audit(tiny_df, tiny_schema):
    df = tiny_df.copy()
    df["event"] = df["event"].astype(object)
    df.loc[0, "event"] = np.nan
    out = simple_impute(df, tiny_schema)
    audit = imputation_audit(df, out, tiny_schema, ["age", "event"])
    assert audit.loc[audit["predictor"] == "event", "missing_after"].iloc[0] == 1
    assert "left NaN" in audit.loc[audit["predictor"] == "event", "imputation_method"].iloc[0]
