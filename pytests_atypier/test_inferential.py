"""Tests for inferential.py — one test per function."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import inferential as inf
from inferential import (
    _empty_inferential_df,
    _safe_z_denominator,
    fit_multivariable_logistic,
    run_inferential,
)
from schema_infer import ColSpec


def test_pool_df_for_display():
    assert inf._pool_df_for_display(float("inf")) == "∞"
    assert inf._pool_df_for_display(12.0) == 12


def test_format_inferential_table():
    df = pd.DataFrame({"df": [5.0, float("inf")], "p": [0.01, 0.05]})
    out = inf._format_inferential_table(df)
    assert "df" in out.columns


def test_safe_z_denominator():
    assert _safe_z_denominator(0.0) == 1.0


def test_ensure_dirs(tmp_output):
    figs, tabs = inf._ensure_dirs(tmp_output)
    assert figs.is_dir() and tabs.is_dir()


def test_build_design(tiny_df, tiny_schema):
    X, mapping, z_params = inf._build_design(tiny_df, tiny_schema, ["age", "sex"])
    assert "age" in mapping
    assert not X.empty
    assert "age" in z_params
    assert "mu" in z_params["age"] and "sd" in z_params["age"]


def test_prune_by_vif(tiny_df, tiny_schema):
    X, _, _ = inf._build_design(tiny_df, tiny_schema, ["age"])
    pruned, vif_df = inf._prune_by_vif(X, threshold=5.0)
    assert list(pruned.columns) == list(X.columns)
    assert "vif" in vif_df.columns


def test_rubin_pool():
    thetas = np.array([0.1, 0.2, 0.15])
    ses = np.array([0.05, 0.05, 0.05])
    pooled = inf._rubin_pool(thetas, ses)
    assert np.isfinite(pooled["coef"])
    assert "or" in pooled


def test_target_is_binary():
    y = pd.Series([True, False, True], dtype="boolean")
    assert inf._target_is_binary(y, ColSpec("event", "binary"))


def test_encode_target():
    y = pd.Series([True, False, True])
    enc, pos = inf._encode_target(y, True)
    assert pos is True
    assert enc.iloc[0] == 1.0


def _make_imputed(tiny_df, tiny_schema):
    df = tiny_df.copy()
    df["age"] = df["age"].fillna(df["age"].median())
    schema = {k: v for k, v in tiny_schema.items() if k in df.columns}
    return [df], schema


def test_fit_multivariable_logistic(tiny_df, tiny_schema):
    frames, schema = _make_imputed(tiny_df, tiny_schema)
    pooled, vif = fit_multivariable_logistic(
        frames, schema, "event", ["age"],
        positive_class=True,
    )
    assert "or" in pooled.columns
    assert "vif" in vif.columns
    assert np.isfinite(pooled.loc[0, "z_mu"])
    assert np.isfinite(pooled.loc[0, "z_sd"])


def test_forest_plot(tmp_path):
    pooled = pd.DataFrame({
        "predictor_col": ["age"],
        "or": [1.5],
        "or_ci_lo": [0.8],
        "or_ci_hi": [2.5],
    })
    figs = tmp_path / "figs"
    figs.mkdir()
    inf._forest_plot(pooled, "event", figs)
    assert (figs / "event__forest.svg").exists()


def test_empty_inferential_df():
    df = _empty_inferential_df()
    assert list(df.columns)


def test_run_inferential(tiny_df, tiny_schema, tmp_output):
    frames, schema = _make_imputed(tiny_df, tiny_schema)
    out = run_inferential(
        frames, schema,
        targets=["event"], predictors=["age"],
        positive_class={"event": True},
        output_root=tmp_output,
    )
    assert "target" in out.columns
