"""Marker rules: the ROC table and the five cut-point selection rules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import marker_rules as th

TARGET = "high_grade"

HIGHER_METRIC = th.Metric("vol", "Volume", "cc", "higher", log_x=True)
LOWER_METRIC = th.Metric("adc", "ADC", "unit", "lower")


def separable_frame(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Two clearly separated groups: cut-point behaviour should be unambiguous."""
    rng = np.random.default_rng(seed)
    benign = rng.normal(10.0, 1.0, n)
    high = rng.normal(20.0, 1.0, n)
    return pd.DataFrame({
        "vol": np.concatenate([benign, high]),
        "adc": np.concatenate([-benign, -high]),  # mirrored: low = suspicious
        TARGET: pd.array([False] * n + [True] * n, dtype="boolean"),
    })


# --------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------
def test_the_operator_and_the_flag_follow_the_declared_direction():
    assert HIGHER_METRIC.op == "≥"
    assert LOWER_METRIC.op == "≤"

    values = pd.Series([5.0, 15.0, None], dtype="Float64")
    high = HIGHER_METRIC.flag(values, 10.0)
    low = LOWER_METRIC.flag(values, 10.0)
    assert high.tolist()[:2] == [False, True]
    assert low.tolist()[:2] == [True, False]
    assert pd.isna(high.iloc[2]) and pd.isna(low.iloc[2])


def test_prose_unit_drops_mathtext_and_descriptive_pseudo_units():
    """"edema ÷ tumor" belongs on an axis, not in the middle of a sentence."""
    mathtext = th.Metric("a", "A", r"$\times 10^{-3}$", "lower")
    assert mathtext.prose_unit == ""
    spelled = th.Metric("a", "A", r"$\times 10^{-3}$", "lower", unit_plain="x10-3")
    assert spelled.prose_unit == "x10-3"
    assert th.Metric("v", "V", "cc", "higher").prose_unit == "cc"

    ratio = th.Metric("i", "Index", "edema ÷ tumor", "higher", unit_plain="")
    assert ratio.axis_label == "Index (edema ÷ tumor)"
    assert ratio.prose_unit == ""


def test_validate_metrics_rejects_a_bad_direction_and_drops_absent_columns():
    df = separable_frame()
    with pytest.raises(ValueError, match="direction"):
        th.validate_metrics([th.Metric("vol", "V", "cc", "up")], df)

    kept = th.validate_metrics([HIGHER_METRIC, th.Metric("nope", "N", "", "higher")], df)
    assert [m.col for m in kept] == ["vol"]


# --------------------------------------------------------------------------
# ROC table
# --------------------------------------------------------------------------
def test_the_roc_table_keeps_only_usable_cutpoints_on_the_metric_scale():
    df = separable_frame()
    x, y = th.metric_arrays(df, HIGHER_METRIC, TARGET)
    tab = th.roc_table(x, y, "higher")
    assert len(tab) > 0
    # Every retained cut-point flags somebody and spares somebody.
    assert (tab["n_flagged"] > 0).all()
    assert (tab["n_flagged"] < len(y)).all()
    assert np.isfinite(tab["cutoff"]).all()
    assert np.allclose(tab["youden_j"],
                       tab["sensitivity"] + tab["specificity"] - 1.0)

    for metric in (HIGHER_METRIC, LOWER_METRIC):
        x, y = th.metric_arrays(df, metric, TARGET)
        tab = th.roc_table(x, y, metric.direction)
        assert tab["cutoff"].min() >= x.min() - 1e-9
        assert tab["cutoff"].max() <= x.max() + 1e-9

    assert th.roc_table(np.arange(20, dtype=float), np.ones(20, dtype=int),
                        "higher").empty


# --------------------------------------------------------------------------
# Selection rules
# --------------------------------------------------------------------------
def test_youden_finds_the_gap_between_separated_groups():
    df = separable_frame()
    x, y = th.metric_arrays(df, HIGHER_METRIC, TARGET)
    cutoff = th.chosen_cutoff(x, y, "higher", "youden")
    assert 13.0 < cutoff < 17.0  # the empty band between the two clusters

    # Flipping the direction mirrors the cut-point.
    lo = th.chosen_cutoff(*th.metric_arrays(df, LOWER_METRIC, TARGET),
                          "lower", "youden")
    assert lo == pytest.approx(-cutoff, abs=1e-9)


def test_every_rule_returns_a_row_of_the_table_and_respects_its_floor():
    df = separable_frame()
    x, y = th.metric_arrays(df, HIGHER_METRIC, TARGET)
    tab = th.roc_table(x, y, "higher")
    for rule in th.RULES:
        idx = th.select_cutoff(tab, rule)
        assert idx is None or idx in tab.index

    i_spec = th.select_cutoff(tab, "spec_ge_90")
    i_sens = th.select_cutoff(tab, "sens_ge_90")
    assert tab.loc[i_spec, "specificity"] >= 0.90
    assert tab.loc[i_sens, "sensitivity"] >= 0.90


def test_an_unknown_rule_or_an_empty_table_yields_nothing():
    # Pure noise: no cut-point reaches 90% specificity *and* any sensitivity.
    rng = np.random.default_rng(0)
    tab = th.roc_table(rng.normal(size=60), np.array([0, 1] * 30), "higher")
    with pytest.raises(KeyError):
        th.select_cutoff(tab, "not_a_rule")

    assert th.select_cutoff(th._empty_roc_table(), "youden") is None


# --------------------------------------------------------------------------
# Table formatting
# --------------------------------------------------------------------------
def test_percentages_and_odds_ratios_print_with_their_intervals():
    assert th.format_pct_ci({"sensitivity": np.nan}, "sensitivity") == ""
    assert th.format_pct_ci(
        {"sensitivity": 0.5, "sensitivity_lo": 0.4, "sensitivity_hi": 0.6},
        "sensitivity",
    ) == "50% (40–60)"

    assert th.format_or_ci({"OR": 3.014, "OR_lo": 1.9, "OR_hi": 4.77}) == "3.01 (1.90–4.77)"
    # Two decimals on a large OR is false precision on cells this size.
    assert th.format_or_ci({"OR": 12.4, "OR_lo": 3.2, "OR_hi": 48.1}) == "12.4 (3.2–48.1)"
    assert th.format_or_ci({"OR": float("nan")}) == ""
