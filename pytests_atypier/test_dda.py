"""Tests for dda.py — one test per function."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dda
from dda import plot_distribution_by_year, run_dda


def test_ensure_dirs(tmp_output):
    figs, tabs = dda._ensure_dirs(tmp_output)
    assert figs.is_dir() and tabs.is_dir()


def test_save_fig(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    p = tmp_path / "x.svg"
    dda._save_fig(fig, p)
    assert p.exists()


def test_stats_continuous():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = dda._stats_continuous(s)
    assert stats["n"] == 5
    assert stats["mean"] == 3.0


def test_stats_categorical():
    s = pd.Series(["a", "a", "b"])
    stats = dda._stats_categorical(s, ordered=False)
    assert stats["first_mode"] == "a"


def test_stats_binary():
    s = pd.Series([True, False, True], dtype="boolean")
    stats = dda._stats_binary(s)
    assert stats["n"] == 3


def test_stats_datetime():
    s = pd.to_datetime(["2018-01-01", "2019-01-01"])
    stats = dda._stats_datetime(s)
    assert stats["n"] == 2


def test_stats_id():
    s = pd.Series(["a", "b", "c"])
    stats = dda._stats_id(s)
    assert stats["n_unique"] == 3


def test_plot_continuous(tmp_path):
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    paths = dda._plot_continuous(s, "age", tmp_path)
    assert len(paths) == 2


def test_ordinal_bar_order():
    s = pd.Series(pd.Categorical(["b", "a"], categories=["a", "b"], ordered=True))
    assert dda._ordinal_bar_order(s, None) == ["a", "b"]


def test_plot_ordinal(tmp_path):
    s = pd.Series(pd.Categorical(["a", "b", "a"], categories=["a", "b"], ordered=True))
    paths = dda._plot_ordinal(s, "grade", tmp_path, ordered_levels=["a", "b"])
    assert len(paths) == 1


def test_plot_distribution_by_year(tiny_df, tmp_path):
    df = tiny_df.copy()
    df["age_bins"] = pd.Categorical(
        ["<50", "50-59", "60-69", "70+"], categories=["<50", "50-59", "60-69", "70+"],
    )
    p = plot_distribution_by_year(df, "age_bins", "entry_year", tmp_path, min_years=2)
    assert p is not None and p.exists()


def test_plot_nominal(tmp_path):
    s = pd.Series(["x", "y", "x"])
    paths = dda._plot_nominal(s, "sex", tmp_path)
    assert len(paths) == 1


def test_plot_binary(tmp_path):
    s = pd.Series([True, False, True], dtype="boolean")
    paths = dda._plot_binary(s, "event", tmp_path)
    assert len(paths) == 1


def test_plot_datetime(tmp_path):
    s = pd.Series(pd.to_datetime(["2018-01-01", "2018-02-01", "2019-01-01"]))
    paths = dda._plot_datetime(s, "entry_year", tmp_path)
    assert len(paths) == 1


def test_run_dda(tiny_df, tiny_schema, tmp_output):
    tables = run_dda(tiny_df, tiny_schema, output_root=tmp_output)
    assert "overall" in tables
    assert (tmp_output / "dda" / "tables" / "dda_overall.csv").exists()
