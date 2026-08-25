"""The two pattern figures of the missingness story.

What they must never get wrong: the counts they draw have to be the cohort's
own counts, a truncated figure has to say so rather than quietly shrink the
cohort, and the words belong in the legend sidecar rather than in the pixels.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import missingness_resolution as mr
from plot_style import figure_legend


def _totals(fig) -> list[str]:
    """The per-column cohort totals the matrix prints above its columns."""
    ax = next(a for a in fig.axes[0].child_axes if a.get_label() == "column-totals")
    return [t.get_text() for t in ax.get_xticklabels()]


@pytest.fixture
def gappy_df() -> pd.DataFrame:
    """Six patients, four patterns, one column that is never missing.

    adc_value is missing three times: once alone, twice beside another gap —
    which is exactly the split the second figure draws.
    """
    return pd.DataFrame({
        "adc_value":    [1.0, np.nan, np.nan, np.nan, 2.0, 3.0],
        "tumor_volume": [4.0, np.nan, 5.0, 6.0, 7.0, 8.0],
        "hemorrhage":   [0.0, 1.0, 1.0, np.nan, 0.0, 1.0],
        "sex":          ["M", "F", "M", "F", "M", "F"],
    })


# ---------------------------------------------------------------------------
# The tables the figures are drawn from
# ---------------------------------------------------------------------------

def test_patterns_are_one_row_per_distinct_pattern_largest_first(gappy_df):
    patterns = mr.missingness_patterns(gappy_df)

    assert list(patterns["n_patients"]) == [3, 1, 1, 1]
    assert patterns["n_patients"].sum() == len(gappy_df)
    assert list(patterns["n_gaps"]) == [0, 1, 2, 2]


def test_patterns_keep_only_columns_that_have_a_gap(gappy_df):
    patterns = mr.missingness_patterns(gappy_df)

    assert [c for c in patterns.columns if c not in ("n_patients", "n_gaps")] == [
        "adc_value", "tumor_volume", "hemorrhage",
    ]


def test_gap_sharing_splits_each_column_into_alone_and_shared(gappy_df):
    sharing = mr.gap_sharing(gappy_df).set_index("column")

    assert sharing.loc["adc_value", "n_missing"] == 3
    assert sharing.loc["adc_value", "n_only_gap"] == 1
    assert sharing.loc["adc_value", "n_shared"] == 2
    assert sharing.loc["tumor_volume", "n_only_gap"] == 0
    assert sharing.loc["tumor_volume", "n_shared"] == 1


def test_gap_sharing_percentages_are_of_the_whole_cohort(gappy_df):
    sharing = mr.gap_sharing(gappy_df).set_index("column")

    assert sharing.loc["adc_value", "pct_missing"] == pytest.approx(3 / 6 * 100)


# ---------------------------------------------------------------------------
# Figure A — the pattern matrix
# ---------------------------------------------------------------------------

def test_pattern_matrix_puts_no_title_in_the_picture(gappy_df):
    fig = mr.pattern_matrix_figure(mr.missingness_patterns(gappy_df))

    assert fig.axes[0].get_title() == ""
    assert fig._suptitle is None


def test_pattern_matrix_rows_are_the_patient_counts(gappy_df):
    fig = mr.pattern_matrix_figure(mr.missingness_patterns(gappy_df))

    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["3", "1", "1", "1"]


def test_pattern_matrix_columns_carry_display_names_and_cohort_totals(gappy_df):
    fig = mr.pattern_matrix_figure(mr.missingness_patterns(gappy_df))
    ax = fig.axes[0]

    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "ADC value", "Tumor volume (cm\u00b3)", "Hemorrhage",
    ]
    assert _totals(fig) == ["3", "1", "1"]


def test_pattern_matrix_legend_counts_come_from_the_data(gappy_df):
    fig = mr.pattern_matrix_figure(mr.missingness_patterns(gappy_df))
    legend = figure_legend(fig)

    assert legend["title"]
    assert legend["plain"]
    assert "4 patterns" in legend["note"]
    assert "6 patients" in legend["note"]
    assert "3 (50.0%)" in legend["note"]        # complete cases


def test_pattern_matrix_draws_at_most_max_patterns_rows(gappy_df):
    fig = mr.pattern_matrix_figure(mr.missingness_patterns(gappy_df), max_patterns=2)

    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["3", "1"]


def test_truncated_pattern_matrix_says_what_it_left_out(gappy_df):
    fig = mr.pattern_matrix_figure(mr.missingness_patterns(gappy_df), max_patterns=2)

    assert "2 rarer patterns (2 patients) are not drawn" in figure_legend(fig)["note"]


def test_truncated_pattern_matrix_still_totals_the_whole_cohort(gappy_df):
    fig = mr.pattern_matrix_figure(mr.missingness_patterns(gappy_df), max_patterns=2)

    assert _totals(fig) == ["3", "1", "1"]


# ---------------------------------------------------------------------------
# Figure D — gaps alone or shared
# ---------------------------------------------------------------------------

def test_gap_overlap_puts_no_title_in_the_picture(gappy_df):
    fig = mr.gap_overlap_figure(mr.gap_sharing(gappy_df), n_patients=len(gappy_df))

    assert fig.axes[0].get_title() == ""
    assert fig._suptitle is None


def test_gap_overlap_rows_are_display_names_worst_first(gappy_df):
    fig = mr.gap_overlap_figure(mr.gap_sharing(gappy_df), n_patients=len(gappy_df))
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]

    assert labels[0] == "ADC value"
    assert set(labels) == {"ADC value", "Tumor volume (cm\u00b3)", "Hemorrhage"}


def test_gap_overlap_separates_the_two_kinds_of_gap_in_its_key(gappy_df):
    fig = mr.gap_overlap_figure(mr.gap_sharing(gappy_df), n_patients=len(gappy_df))
    keys = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]

    assert len(keys) == 2
    assert any("only" in k.lower() for k in keys)
    assert any("another" in k.lower() for k in keys)


def test_gap_overlap_legend_reports_the_shared_share(gappy_df):
    fig = mr.gap_overlap_figure(mr.gap_sharing(gappy_df), n_patients=len(gappy_df))
    note = figure_legend(fig)["note"]

    # 4 of the 5 gaps sit beside another gap in the same patient.
    assert "4 of 5" in note


# ---------------------------------------------------------------------------
# Wired into the pipeline
# ---------------------------------------------------------------------------

def test_analyze_missingness_writes_both_figures_with_their_legends(gappy_df, tmp_path):
    mr.analyze_missingness(gappy_df, output_root=tmp_path)

    figs = tmp_path / "missingness" / "figures"
    tabs = tmp_path / "missingness" / "tables"
    for stem in ("missingness_patterns", "gap_sharing"):
        assert (figs / f"{stem}.png").is_file()
        legend = json.loads((figs / f"{stem}.legend.json").read_text(encoding="utf-8"))
        assert legend["title"] and legend["plain"] and legend["note"]
    assert (tabs / "missingness_patterns.csv").is_file()
    assert (tabs / "gap_sharing.csv").is_file()


def test_analyze_missingness_no_longer_draws_the_superseded_figures(gappy_df, tmp_path):
    """The bar chart and the overlap heatmap were retired by the two above.

    The per-column table stays: the report reads it for the "Top missing"
    block and for its count of missing cells.
    """
    mr.analyze_missingness(gappy_df, output_root=tmp_path)

    figs = tmp_path / "missingness" / "figures"
    tabs = tmp_path / "missingness" / "tables"
    assert not (figs / "missing_per_column.png").exists()
    assert not (figs / "co_missingness_heatmap.png").exists()
    assert not (tabs / "co_missingness_jaccard.csv").exists()
    assert (tabs / "missing_per_column.csv").is_file()


def test_analyze_missingness_draws_no_pattern_figure_without_gaps(tmp_path):
    whole = pd.DataFrame({"age": [40.0, 50.0], "sex": ["M", "F"]})

    mr.analyze_missingness(whole, output_root=tmp_path)

    figs = tmp_path / "missingness" / "figures"
    assert not (figs / "missingness_patterns.png").exists()
    assert not (figs / "gap_sharing.png").exists()
