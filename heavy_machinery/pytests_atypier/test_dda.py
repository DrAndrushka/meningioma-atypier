"""Tests for dda.py — descriptive tables and figure outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dda
from dda import run_dda


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
    assert pd.isna(stats["median_category"])
    assert pd.isna(stats["second_mode"])
    assert pd.isna(stats["second_mode_pct"])
    assert stats["rarest_pct"] == round(100 / 3, 2)


def test_stats_categorical_three_levels():
    s = pd.Series(["a", "a", "a", "b", "c"])
    stats = dda._stats_categorical(s, ordered=False)
    assert stats["second_mode"] == "b"
    assert stats["second_mode_pct"] == 20.0
    assert stats["rarest_pct"] == 20.0


def test_stats_categorical_ordinal():
    s = pd.Series(pd.Categorical(["a", "b", "a"], categories=["a", "b"], ordered=True))
    stats = dda._stats_categorical(s, ordered=True)
    assert stats["median_category"] == "a"


def test_stats_binary():
    s = pd.Series([True, False, True], dtype="boolean")
    stats = dda._stats_binary(s)
    assert stats["n"] == 3
    assert stats["mode"] == True
    assert stats["mode_pct"] == round(200 / 3, 2)
    assert stats["rarest"] == False
    assert stats["rarest_pct"] == round(100 / 3, 2)
    assert "median_category" not in stats
    assert "second_mode" not in stats
    assert "first_mode" not in stats


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
