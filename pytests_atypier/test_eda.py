"""Tests for eda.py — one test per function."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eda
from eda import benjamini_hochberg, screen_associations
from schema_infer import ColSpec


def test_ensure_dirs(tmp_output):
    figs, tabs = eda._ensure_dirs(tmp_output)
    assert figs.is_dir() and tabs.is_dir()


def test_benjamini_hochberg():
    p = pd.Series([0.01, 0.04, 0.03])
    q = benjamini_hochberg(p)
    assert (q <= 1).all()
    assert q.notna().all()


def test_encode_binary_target():
    y = pd.Series([False, True, False])
    enc, pos = eda._encode_binary_target(y, True)
    assert pos is True
    assert enc.tolist() == [0.0, 1.0, 0.0]


def test_cramers_v():
    table = np.array([[10, 5], [8, 12]])
    v = eda._cramers_v(table)
    assert 0 <= v <= 1


def test_mwu_with_effect():
    x1 = np.array([1.0, 2.0, 3.0, 4.0])
    x0 = np.array([5.0, 6.0, 7.0, 8.0])
    u, p, r, n = eda._mwu_with_effect(x1, x0)
    assert n == 8
    assert 0 <= p <= 1


def test_infer_target_kind():
    y = pd.Series([True, False, True, False])
    assert eda._infer_target_kind(y, ColSpec("event", "binary")) == "binary"


def test_prepare_target():
    y = pd.Series([True, False, True])
    enc, _ = eda._prepare_target(y, "binary", ColSpec("event", "binary"), True)
    assert enc.sum() == 2


def test_predictor_values(tiny_df, tiny_schema):
    pair = tiny_df[["event", "age"]].dropna()
    x, kind = eda._predictor_values(pair, "age", tiny_schema["age"])
    assert kind == "continuous"


def test_kruskal_with_effect():
    H, p, eps2 = eda._kruskal_with_effect([np.array([1, 2]), np.array([3, 4])])
    assert not np.isnan(H)


def test_chi2_row():
    table = np.array([[10, 5], [8, 12]])
    row = eda._chi2_row(table)
    assert "p" in row and "test" in row


def test_spearman_row():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    x = np.array([1.0, 2.0, 3.0, 5.0])
    row = eda._spearman_row(y, x)
    assert row["test"] == "spearman"


def test_association_test(tiny_df, tiny_schema):
    pair = tiny_df[["event", "age"]].dropna().assign(_y=[1.0, 0.0, 1.0])
    row = eda._association_test("binary", pair, "event", "age", tiny_schema["age"])
    assert "test" in row


def test_polish_ax():
    fig, ax = plt.subplots()
    eda._polish_ax(ax)
    plt.close(fig)


def test_categorical_fig_width():
    assert eda._categorical_fig_width(3) >= 3.2


def test_level_order():
    s = pd.Categorical(["b", "a"], categories=["a", "b"], ordered=True)
    assert eda._level_order(s, None) == ["a", "b"]


def test_errorbar_yerr():
    err = eda._errorbar_yerr([0.5], [0.3], [0.7])
    assert err.shape == (2, 1)


def test_annotate_above():
    fig, ax = plt.subplots()
    eda._annotate_above(ax, np.array([0]), np.array([0.5]), ["hi"])
    plt.close(fig)


def test_plot_binary_target_rates(tiny_df):
    sub = tiny_df[["event", "sex"]].dropna()
    fig, ax = plt.subplots()
    eda._plot_binary_target_rates(ax, sub, "event", "sex", True, pred_levels=["M", "F"])
    plt.close(fig)


def test_plot_pair(tiny_df, tiny_schema, tmp_path):
    figs = tmp_path / "figs"
    figs.mkdir()
    eda._plot_pair(
        tiny_df, "event", "age",
        target_mode="binary", pred_kind="continuous",
        target_spec=tiny_schema["event"], pred_spec=tiny_schema["age"],
        positive_class=True, figs_dir=figs,
    )
    assert any(figs.glob("*.svg"))


def test_screen_associations(tiny_df, tiny_schema, tmp_output):
    out = screen_associations(
        tiny_df, tiny_schema,
        targets=["event"], predictors=["age", "sex"],
        output_root=tmp_output,
    )
    assert "target" in out.columns
    assert (tmp_output / "eda" / "tables" / "associations.csv").exists()
