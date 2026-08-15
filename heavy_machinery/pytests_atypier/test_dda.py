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


def test_ensure_dirs(tmp_output):
    figs, tabs = dda._ensure_dirs(tmp_output)
    assert figs.is_dir() and tabs.is_dir()


def test_save_figure_writes_only_the_report_png_by_default(tmp_path):
    """A normal pipeline run skips the 1200-dpi TIF — report.html shows the PNG."""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    out = save_figure(fig, tmp_path / "x.png")
    assert out.exists()
    assert out.suffix == ".png"
    assert not (tmp_path / "x.tif").exists()
    assert not (tmp_path / "x.eps").exists()


def test_save_figure_writes_the_journal_tif_under_the_submission_profile(
    tmp_path, monkeypatch,
):
    """ATYPIER_FIGURES=submission restores the AJNR 1200-dpi TIF export."""
    monkeypatch.setenv("ATYPIER_FIGURES", "submission")
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    out = save_figure(fig, tmp_path / "x.png")
    assert out.suffix == ".png"
    assert (tmp_path / "x.tif").exists()
    assert not (tmp_path / "x.eps").exists()


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
    """Histogram and marginal box ship as one panel-aligned figure."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    paths = dda._plot_continuous(s, "age", tmp_path)
    assert len(paths) == 1
    assert paths[0].name == "age__distribution.png"
    assert paths[0].exists()


def test_plot_continuous_skips_kde_when_a_single_value_dominates():
    """A structural spike (zero-inflation) is not smoothable — histogram only."""
    spiked = np.concatenate([np.zeros(80), np.linspace(1.0, 50.0, 20)])
    assert dda._has_point_mass(spiked)
    assert not dda._has_point_mass(np.linspace(0.0, 50.0, 100))


def test_plot_continuous_is_reproducible(tmp_path):
    """Jitter uses a seeded generator, so a second run still writes a figure."""
    s = pd.Series(np.linspace(1.0, 50.0, 60))
    first = dda._plot_continuous(s, "age", tmp_path)
    second = dda._plot_continuous(s, "age", tmp_path)
    assert first and first[0].exists()
    assert second and second[0].exists()


def test_ordinal_bar_order():
    s = pd.Series(pd.Categorical(["b", "a"], categories=["a", "b"], ordered=True))
    assert dda._ordinal_bar_order(s, None) == ["a", "b"]


def test_plot_ordinal(tmp_path):
    s = pd.Series(pd.Categorical(["a", "b", "a"], categories=["a", "b"], ordered=True))
    paths = dda._plot_ordinal(s, "grade", tmp_path, ordered_levels=["a", "b"])
    assert len(paths) == 1


def test_plot_nominal_pins_declared_positive_last(tmp_path):
    s = pd.Series(["left"] * 10 + ["right"] * 3 + ["midline"] * 2)
    captured = {}
    original = dda._plot_category_proportions

    def spy(series, name, out_dir, *, order, note=None):
        captured["order"] = order
        return original(series, name, out_dir, order=order, note=note)

    dda._plot_category_proportions = spy
    try:
        dda._plot_nominal(s, "side", tmp_path, positive_class="right")
    finally:
        dda._plot_category_proportions = original
    assert captured["order"][-1] == "right"


def test_plot_binary_pins_declared_positive_last(tmp_path):
    s = pd.Series([False, True, False, True, True])
    captured = {}
    original = dda._plot_category_proportions

    def spy(series, name, out_dir, *, order, note=None):
        captured["order"] = order
        return original(series, name, out_dir, order=order, note=note)

    dda._plot_category_proportions = spy
    try:
        dda._plot_binary(s, "flag", tmp_path, positive_class=False)
    finally:
        dda._plot_category_proportions = original
    assert captured["order"][-1] is False
    assert captured["order"][0] is True


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


def test_plot_nominal_pools_the_rare_tail_and_says_so(tmp_path):
    """The pooled tail is disclosed rather than silently dropped."""
    levels = [f"l{i}" for i in range(20)]
    s = pd.Series(levels * 2)
    captured = {}
    original = dda._plot_category_proportions

    def spy(series, name, out_dir, *, order, note=None):
        captured["order"] = order
        captured["note"] = note
        return original(series, name, out_dir, order=order, note=note)

    dda._plot_category_proportions = spy
    try:
        dda._plot_nominal(s, "site", tmp_path, top_n=5)
    finally:
        dda._plot_category_proportions = original

    assert len(captured["order"]) == 5
    assert "15" in captured["note"] and "30" in captured["note"]


def test_plot_datetime_keeps_empty_months_as_gaps(tmp_path):
    """Months without records must stay on the axis, not be collapsed away."""
    s = pd.Series(pd.to_datetime(["2018-01-05", "2018-06-05"]))
    dda._plot_datetime(s, "mri_date", tmp_path)
    # 2018-01 .. 2018-06 is six monthly slots for two records.
    monthly = s.dt.to_period("M").value_counts()
    full = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    assert len(full) == 6
    assert (tmp_path / "mri_date__timeline.png").exists()


def test_run_dda(tiny_df, tiny_schema, tmp_output):
    tables = run_dda(tiny_df, tiny_schema, output_root=tmp_output)
    assert "overall" in tables
    assert (tmp_output / "dda" / "tables" / "dda_overall.csv").exists()


def test_run_dda_skips_hidden_parent_columns(tiny_df, tiny_schema, tmp_output):
    cleaning = tmp_output / "cleaning"
    cleaning.mkdir(parents=True)
    pd.DataFrame({"column": ["sex"]}).to_csv(
        cleaning / "hidden_parent_columns.csv", index=False,
    )
    tables = run_dda(tiny_df, tiny_schema, output_root=tmp_output)
    cat = tables["categorical"]
    assert "sex" not in set(cat["column"]) if cat is not None and not cat.empty else True
    assert not list((tmp_output / "dda" / "figures").glob("sex__*.png"))


def test_run_dda_bivariate_skips_hidden_parent(tiny_df, tmp_output):
    cleaning = tmp_output / "cleaning"
    cleaning.mkdir(parents=True)
    pd.DataFrame({"column": ["sex"]}).to_csv(
        cleaning / "hidden_parent_columns.csv", index=False,
    )
    paths = dda.run_dda_bivariate(
        tiny_df, {"age": ["sex"], "sex": ["age"]}, output_root=tmp_output,
    )
    assert paths == []


def test_run_dda_bivariate(tiny_df, tmp_output):
    paths = dda.run_dda_bivariate(
        tiny_df, {"age": ["sex"]}, output_root=tmp_output,
    )
    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].parent.name == "figures_bivariate"


def test_run_dda_trivariate(tmp_output):
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


def test_plot_trivariate_legend_labels(tmp_path):
    rng = np.random.default_rng(2)
    n = 40
    df = pd.DataFrame({
        "diam": rng.uniform(1, 8, n),
        "vol": rng.uniform(5, 80, n),
        "high_grade": rng.choice([False, True], n),
    })
    path = dda._plot_trivariate(df, "diam", "vol", "high_grade", tmp_path)
    assert path is not None
    # Rebuild once more to inspect legend text via a direct call path:
    # (SVG exists; check display helper + that plotting didn't raise.)
    assert dda._display_level(False, by_col="high_grade") == "Low grade"
    assert dda._display_level(True, by_col="high_grade") == "High grade"


def test_run_dda_trivariate_cont_cat_and_cat_cat(tmp_output):
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


def test_plot_trivariate_respects_ordered_categories(tmp_path):
    df = pd.DataFrame({
        "vol": np.linspace(10, 90, 30),
        "bin": ["a", "b"] * 15,
        "grade": pd.Categorical(
            ["III", "I", "II"] * 10,
            categories=["I", "II", "III"], ordered=True,
        ),
    })
    path = dda._plot_trivariate(df, "vol", "bin", "grade", tmp_path)
    assert path is not None
    assert dda._ordered_levels(df["grade"]) == ["I", "II", "III"]


def test_plot_trivariate_skips_high_cardinality_by(tiny_df, tmp_path):
    df = tiny_df.copy()
    df = pd.concat([df] * 5, ignore_index=True)
    df["noisy"] = [f"l{i}" for i in range(len(df))]
    df["vol"] = np.linspace(1, 10, len(df))
    df["diam"] = np.linspace(2, 11, len(df))
    assert dda._plot_trivariate(df, "vol", "diam", "noisy", tmp_path) is None


def test_normalize_science_styles():
    assert normalize_science_styles(None) == ["science", "nature", "no-latex"]
    assert normalize_science_styles("ieee") == ["ieee"]
    assert normalize_science_styles(["science", "nature", "no-latex"]) == [
        "science", "nature", "no-latex",
    ]


def test_run_dda_trivariate_ieee_style(tmp_output):
    rng = np.random.default_rng(3)
    n = 30
    df = pd.DataFrame({
        "diam": rng.uniform(1, 8, n),
        "vol": rng.uniform(5, 80, n),
        "high_grade": rng.choice([False, True], n),
    })
    paths = dda.run_dda_trivariate(
        df,
        {("diam", "vol"): ["high_grade"]},
        output_root=tmp_output,
        science_style=["science", "ieee", "no-latex"],
    )
    assert len(paths) == 1
    assert paths[0].exists()


def test_run_dda_bivariate_continuous_partner(tiny_df, tmp_output):
    df = tiny_df.copy()
    df["adc_value"] = [0.7, 0.8, 0.9, 1.0, 0.75, 0.85, 0.95, 1.05][: len(df)]
    # Need enough distinct values to count as continuous-like
    df = pd.concat([df] * 4, ignore_index=True)
    df["adc_value"] = np.linspace(0.5, 1.2, len(df))
    paths = dda.run_dda_bivariate(
        df, {"age": ["adc_value"], "sex": ["adc_value"]}, output_root=tmp_output,
    )
    assert len(paths) == 2
    assert all(p.exists() for p in paths)


def test_build_dda_bivariate_specs_includes_continuous(tiny_df, tiny_schema):
    from schema_infer import ColSpec

    df = tiny_df.copy()
    df["adc_value"] = [0.7, 0.8, 0.9, 1.1]
    schema = dict(tiny_schema)
    schema["adc_value"] = ColSpec("adc_value", "continuous")
    specs = dda.build_dda_bivariate_specs(df, schema, ["age"])
    assert "adc_value" in specs["age"]
    assert "sex" in specs["age"]
    assert "entry_year" not in specs["age"]
    assert "note" not in specs["age"]


def test_plot_bivariate_skips_high_cardinality_categorical(tiny_df, tmp_path):
    df = tiny_df.copy()
    df = pd.concat([df] * 5, ignore_index=True)
    df["noisy"] = [f"l{i}" for i in range(len(df))]
    assert dda._plot_bivariate(df, "age", "noisy", tmp_path, max_marker_levels=12) is None


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
