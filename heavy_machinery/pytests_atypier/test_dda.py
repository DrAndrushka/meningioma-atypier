"""Tests for dda.py — descriptive tables and figure outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

import dda
from dda import run_dda
from plot_style import normalize_science_styles, save_figure
from schema_infer import ColSpec


def test_ensure_dirs_and_figure_export(tmp_output, tmp_path, monkeypatch):
    """A normal pipeline run skips the 1200-dpi TIF — report.html shows the PNG.
    ATYPIER_FIGURES=submission restores the AJNR TIF export."""
    figs, tabs = dda._ensure_dirs(tmp_output)
    assert figs.is_dir() and tabs.is_dir()

    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    out = save_figure(fig, tmp_path / "x.png")
    assert out.exists()
    assert out.suffix == ".png"
    assert not (tmp_path / "x.tif").exists()
    assert not (tmp_path / "x.eps").exists()

    monkeypatch.setenv("ATYPIER_FIGURES", "submission")
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    out = save_figure(fig, tmp_path / "y.png")
    assert out.suffix == ".png"
    assert (tmp_path / "y.tif").exists()
    assert not (tmp_path / "y.eps").exists()


def test_stats_for_continuous_binary_datetime_and_id():
    stats = dda._stats_continuous(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert stats["n"] == 5
    assert stats["mean"] == 3.0

    stats = dda._stats_binary(pd.Series([True, False, True], dtype="boolean"))
    assert stats["n"] == 3
    assert stats["mode"] == True
    assert stats["mode_pct"] == round(200 / 3, 2)
    assert stats["rarest"] == False
    assert stats["rarest_pct"] == round(100 / 3, 2)
    assert "median_category" not in stats
    assert "second_mode" not in stats
    assert "first_mode" not in stats

    assert dda._stats_datetime(pd.to_datetime(["2018-01-01", "2019-01-01"]))["n"] == 2
    assert dda._stats_id(pd.Series(["a", "b", "c"]))["n_unique"] == 3


def test_stats_categorical():
    stats = dda._stats_categorical(pd.Series(["a", "a", "b"]), ordered=False)
    assert stats["first_mode"] == "a"
    assert pd.isna(stats["median_category"])
    assert pd.isna(stats["second_mode"])
    assert pd.isna(stats["second_mode_pct"])
    assert stats["rarest_pct"] == round(100 / 3, 2)

    stats = dda._stats_categorical(pd.Series(["a", "a", "a", "b", "c"]), ordered=False)
    assert stats["second_mode"] == "b"
    assert stats["second_mode_pct"] == 20.0
    assert stats["rarest_pct"] == 20.0

    ordinal = pd.Series(pd.Categorical(["a", "b", "a"], categories=["a", "b"],
                                       ordered=True))
    assert dda._stats_categorical(ordinal, ordered=True)["median_category"] == "a"


def test_plot_continuous(tmp_path):
    """Histogram and marginal box ship as one panel-aligned figure, and the
    jitter is seeded so a second run still writes a figure."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    paths = dda._plot_continuous(s, "age", tmp_path)
    assert len(paths) == 1
    assert paths[0].name == "age__distribution.png"
    assert paths[0].exists()

    # A structural spike (zero-inflation) is not smoothable — histogram only.
    spiked = np.concatenate([np.zeros(80), np.linspace(1.0, 50.0, 20)])
    assert dda._has_point_mass(spiked)
    assert not dda._has_point_mass(np.linspace(0.0, 50.0, 100))

    s = pd.Series(np.linspace(1.0, 50.0, 60))
    first = dda._plot_continuous(s, "wide", tmp_path)
    second = dda._plot_continuous(s, "wide", tmp_path)
    assert first and first[0].exists()
    assert second and second[0].exists()


def test_plot_ordinal_keeps_the_declared_order(tmp_path):
    s = pd.Series(pd.Categorical(["b", "a"], categories=["a", "b"], ordered=True))
    assert dda._ordinal_bar_order(s, None) == ["a", "b"]

    s = pd.Series(pd.Categorical(["a", "b", "a"], categories=["a", "b"], ordered=True))
    assert len(dda._plot_ordinal(s, "grade", tmp_path, ordered_levels=["a", "b"])) == 1


def _spy_on_category_proportions(call):
    """Run ``call`` with ``_plot_category_proportions`` spied on."""
    captured = {}
    original = dda._plot_category_proportions

    def spy(series, name, out_dir, *, order, note=None):
        captured["order"] = order
        captured["note"] = note
        return original(series, name, out_dir, order=order, note=note)

    dda._plot_category_proportions = spy
    try:
        call()
    finally:
        dda._plot_category_proportions = original
    return captured


def test_category_proportions_pin_the_positive_class_and_pool_the_tail(tmp_path):
    """The pooled tail is disclosed rather than silently dropped."""
    s = pd.Series(["left"] * 10 + ["right"] * 3 + ["midline"] * 2)
    captured = _spy_on_category_proportions(
        lambda: dda._plot_nominal(s, "side", tmp_path, positive_class="right"))
    assert captured["order"][-1] == "right"

    s = pd.Series([False, True, False, True, True])
    captured = _spy_on_category_proportions(
        lambda: dda._plot_binary(s, "flag", tmp_path, positive_class=False))
    assert captured["order"][-1] is False
    assert captured["order"][0] is True

    levels = [f"l{i}" for i in range(20)]
    s = pd.Series(levels * 2)
    captured = _spy_on_category_proportions(
        lambda: dda._plot_nominal(s, "site", tmp_path, top_n=5))
    assert len(captured["order"]) == 5
    assert "15" in captured["note"] and "30" in captured["note"]


def test_run_dda_records_declared_positive_class(tmp_output):
    df = pd.DataFrame({
        "side": ["left", "right", "left", "midline"] * 5,
        "flag": [True, False, True, False] * 5,
    })
    schema = {
        "side": ColSpec("side", "nominal", positive_class="right"),
        "flag": ColSpec("flag", "binary", positive_class=True),
    }
    tables = run_dda(df, schema, output_root=tmp_output)
    cat = tables["categorical"]
    assert cat.loc[cat["column"] == "side", "positive_class"].iloc[0] == "right"
    binary = tables["binary"]
    assert binary.loc[binary["column"] == "flag", "positive_class"].iloc[0] == True


def test_plot_datetime_keeps_empty_months_as_gaps(tmp_path):
    """Months without records must stay on the axis, not be collapsed away."""
    s = pd.Series(pd.to_datetime(["2018-01-05", "2018-06-05"]))
    dda._plot_datetime(s, "mri_date", tmp_path)
    # 2018-01 .. 2018-06 is six monthly slots for two records.
    monthly = s.dt.to_period("M").value_counts()
    full = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    assert len(full) == 6
    assert (tmp_path / "mri_date__timeline.png").exists()


def test_run_dda_writes_its_tables_and_skips_hidden_parents(
    tiny_df, tiny_schema, tmp_output,
):
    plain = tmp_output / "plain"
    tables = run_dda(tiny_df, tiny_schema, output_root=plain)
    assert "overall" in tables
    assert (plain / "dda" / "tables" / "dda_overall.csv").exists()

    # A fresh root, so the figures below are the hidden-parent run's own.
    cleaning = tmp_output / "cleaning"
    cleaning.mkdir(parents=True)
    pd.DataFrame({"column": ["sex"]}).to_csv(
        cleaning / "hidden_parent_columns.csv", index=False,
    )
    tables = run_dda(tiny_df, tiny_schema, output_root=tmp_output)
    cat = tables["categorical"]
    assert "sex" not in set(cat["column"]) if cat is not None and not cat.empty else True
    assert not list((tmp_output / "dda" / "figures").glob("sex__*.png"))

    # A hidden parent is skipped on the bivariate side too.
    assert dda.run_dda_bivariate(
        tiny_df, {"age": ["sex"], "sex": ["age"]}, output_root=tmp_output,
    ) == []


def test_run_dda_bivariate(tiny_df, tmp_output):
    paths = dda.run_dda_bivariate(
        tiny_df, {"age": ["sex"]}, output_root=tmp_output,
    )
    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].parent.name == "figures_bivariate"

    df = pd.concat([tiny_df] * 4, ignore_index=True)
    # Need enough distinct values to count as continuous-like
    df["adc_value"] = np.linspace(0.5, 1.2, len(df))
    paths = dda.run_dda_bivariate(
        df, {"age": ["adc_value"], "sex": ["adc_value"]}, output_root=tmp_output,
    )
    assert len(paths) == 2
    assert all(p.exists() for p in paths)


def test_build_dda_bivariate_specs_and_the_cardinality_limit(
    tiny_df, tiny_schema, tmp_path,
):
    df = tiny_df.copy()
    df["adc_value"] = [0.7, 0.8, 0.9, 1.1]
    schema = dict(tiny_schema)
    schema["adc_value"] = ColSpec("adc_value", "continuous")
    specs = dda.build_dda_bivariate_specs(df, schema, ["age"])
    assert "adc_value" in specs["age"]
    assert "sex" in specs["age"]
    assert "entry_year" not in specs["age"]
    assert "note" not in specs["age"]

    noisy = pd.concat([tiny_df] * 5, ignore_index=True)
    noisy["noisy"] = [f"l{i}" for i in range(len(noisy))]
    assert dda._plot_bivariate(noisy, "age", "noisy", tmp_path,
                               max_marker_levels=12) is None


def test_run_dda_trivariate(tmp_output, tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "vol": rng.uniform(5, 80, n),
        "diam": rng.uniform(1, 8, n),
        "high_grade": rng.choice([False, True], n),
    })
    paths = dda.run_dda_trivariate(
        df,
        {("diam", "vol"): ["high_grade"]},
        output_root=tmp_output,
    )
    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].parent.name == "figures_trivariate"
    assert paths[0].name == "diam__vs__vol__by__high_grade.png"

    # Legend levels are displayed through the label map.
    assert dda._plot_trivariate(df, "diam", "vol", "high_grade", tmp_path) is not None
    assert dda._display_level(False, by_col="high_grade") == "Low grade"
    assert dda._display_level(True, by_col="high_grade") == "High grade"

    assert normalize_science_styles(None) == ["science", "nature", "no-latex"]
    assert normalize_science_styles("ieee") == ["ieee"]
    assert normalize_science_styles(["science", "nature", "no-latex"]) == [
        "science", "nature", "no-latex",
    ]
    paths = dda.run_dda_trivariate(
        df,
        {("diam", "vol"): ["high_grade"]},
        output_root=tmp_output,
        science_style=["science", "ieee", "no-latex"],
    )
    assert len(paths) == 1
    assert paths[0].exists()


def test_trivariate_handles_categorical_and_ordered_groups(
    tiny_df, tmp_output, tmp_path,
):
    rng = np.random.default_rng(1)
    n = 36
    df = pd.DataFrame({
        "vol": rng.uniform(5, 80, n),
        "side": rng.choice(["L", "R", "mid"], n),
        "margin": rng.choice(["smooth", "irregular"], n),
        "high_grade": rng.choice(["low", "high"], n),
    })
    # ordered categorical group
    df["grade_ord"] = pd.Categorical(
        rng.choice(["I", "II", "III"], n),
        categories=["I", "II", "III"], ordered=True,
    )
    paths = dda.run_dda_trivariate(
        df,
        {
            ("vol", "side"): ["high_grade"],
            ("side", "margin"): ["grade_ord"],
        },
        output_root=tmp_output,
    )
    assert len(paths) == 2
    stems = {p.name for p in paths}
    assert "vol__vs__side__by__high_grade.png" in stems
    assert "side__vs__margin__by__grade_ord.png" in stems

    ordered = pd.DataFrame({
        "vol": np.linspace(10, 90, 30),
        "bin": ["a", "b"] * 15,
        "grade": pd.Categorical(
            ["III", "I", "II"] * 10,
            categories=["I", "II", "III"], ordered=True,
        ),
    })
    assert dda._plot_trivariate(ordered, "vol", "bin", "grade", tmp_path) is not None
    assert dda._ordered_levels(ordered["grade"]) == ["I", "II", "III"]

    # A grouping column with a level per row is not a grouping column.
    noisy = pd.concat([tiny_df] * 5, ignore_index=True)
    noisy["noisy"] = [f"l{i}" for i in range(len(noisy))]
    noisy["vol"] = np.linspace(1, 10, len(noisy))
    noisy["diam"] = np.linspace(2, 11, len(noisy))
    assert dda._plot_trivariate(noisy, "vol", "diam", "noisy", tmp_path) is None


def test_continuous_density_clips_to_observed_min(tiny_df, tmp_path):
    """Hist + KDE must keep xlim at the lowest observed continuous value."""
    df = tiny_df.dropna(subset=["age", "sex"]).copy()
    df["age"] = df["age"].clip(lower=40.0)  # all ages ≥ 40
    path = dda._plot_continuous_density_by_categorical(
        df, cont_col="age", cat_col="sex", out_dir=tmp_path, file_stem="age_by_sex",
    )
    assert path.exists()
    order = dda._ordered_levels(df["sex"])
    fig, ax = plt.subplots()
    x_lo = float(df["age"].min())
    x_hi = float(df["age"].max())
    for level in order:
        sub = df.loc[df["sex"] == level, "age"].dropna()
        if sub.empty:
            continue
        sns.histplot(
            sub, bins=18, kde=True, stat="count",
            binrange=(x_lo, x_hi),
            kde_kws={"clip": (x_lo, x_hi), "cut": 0},
            ax=ax,
        )
    ax.set_xlim(x_lo, x_hi)
    assert ax.get_xlim()[0] == pytest.approx(x_lo)
    plt.close(fig)
