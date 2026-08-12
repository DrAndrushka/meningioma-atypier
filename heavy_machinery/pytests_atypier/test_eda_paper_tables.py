"""Tests for paper-style EDA table builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
from schema_infer import ColSpec

from eda_paper_tables import build_eda_paper_tables


def test_build_eda_paper_tables_binary_and_continuous(tmp_path):
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "high_grade": pd.array(rng.integers(0, 2, n), dtype="boolean"),
        "sign_a": pd.array(rng.integers(0, 2, n), dtype="boolean"),
        "age": rng.normal(60, 12, n),
    })
    schema = {
        "high_grade": ColSpec("high_grade", "binary"),
        "sign_a": ColSpec("sign_a", "binary"),
        "age": ColSpec("age", "continuous"),
    }
    assoc = pd.DataFrame({
        "target": ["high_grade", "high_grade"],
        "predictor": ["sign_a", "age"],
        "kind": ["binary", "continuous"],
        "test": ["chi2", "mann_whitney_u"],
        "p": [0.01, 0.02],
        "p_fdr": [0.02, 0.03],
        "in_fdr_family": [True, True],
        "positive_class": [True, True],
    })
    out = build_eda_paper_tables(df, schema, assoc, output_root=tmp_path)
    assert (tmp_path / "eda" / "tables" / "eda_paper_tables.csv").exists()
    assert set(out["table_kind"]) == {"binary", "continuous"}
    binary = out[out["predictor"] == "sign_a"].iloc[0]
    assert "/" in binary["grade1"] and "%" in binary["grade1"]
    assert binary["auc"] == ""
    cont = out[out["predictor"] == "age"].iloc[0]
    assert "[" in cont["grade1"]
    assert "OR" not in cont["effect"] or "(" in cont["effect"]


def test_build_eda_paper_tables_categorical_reference_coding(tmp_path):
    df = pd.DataFrame({
        "high_grade": [False, False, False, True, True, True] * 10,
        "side": ["left", "right", "midline", "left", "right", "midline"] * 10,
    })
    schema = {
        "high_grade": ColSpec("high_grade", "binary"),
        "side": ColSpec("side", "nominal"),
    }
    assoc = pd.DataFrame({
        "target": ["high_grade"],
        "predictor": ["side"],
        "kind": ["nominal"],
        "test": ["chi2"],
        "p": [0.04],
        "p_fdr": [0.05],
        "in_fdr_family": [True],
        "positive_class": [True],
    })
    out = build_eda_paper_tables(df, schema, assoc, output_root=tmp_path)
    cat = out[out["predictor"] == "side"]
    assert set(cat["table_kind"]) == {"nominal"}
    assert (cat["row_role"] == "variable").sum() == 1
    assert (cat["row_role"] == "reference").sum() == 1
    assert (cat["row_role"] == "level").sum() >= 1
    # parent first
    assert cat.iloc[0]["row_role"] == "variable"


def test_build_eda_paper_tables_skips_excluded(tmp_path):
    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame({
        "high_grade": pd.array(rng.integers(0, 2, n), dtype="boolean"),
        "keep_me": pd.array(rng.integers(0, 2, n), dtype="boolean"),
        "hide_me": pd.array(rng.integers(0, 2, n), dtype="boolean"),
    })
    schema = {
        "high_grade": ColSpec("high_grade", "binary"),
        "keep_me": ColSpec("keep_me", "binary"),
        "hide_me": ColSpec("hide_me", "binary"),
    }
    assoc = pd.DataFrame({
        "target": ["high_grade", "high_grade"],
        "predictor": ["keep_me", "hide_me"],
        "kind": ["binary", "binary"],
        "test": ["chi2", "chi2"],
        "p": [0.01, 0.02],
        "p_fdr": [0.02, 0.03],
        "in_fdr_family": [True, True],
        "positive_class": [True, True],
    })
    cleaning = tmp_path / "cleaning"
    cleaning.mkdir(parents=True)
    pd.DataFrame({"column": ["hide_me"]}).to_csv(
        cleaning / "eda_excluded_columns.csv", index=False,
    )
    out = build_eda_paper_tables(df, schema, assoc, output_root=tmp_path)
    assert set(out["predictor"]) == {"keep_me"}
