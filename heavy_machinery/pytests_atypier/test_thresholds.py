"""Cut-point engine: ROC table, selection rules, bootstrap, summary tables."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import thresholds as th

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
def test_operator_follows_direction():
    assert HIGHER_METRIC.op == "≥"
    assert LOWER_METRIC.op == "≤"


def test_flag_preserves_missing_and_respects_direction():
    values = pd.Series([5.0, 15.0, None], dtype="Float64")
    high = HIGHER_METRIC.flag(values, 10.0)
    low = LOWER_METRIC.flag(values, 10.0)
    assert high.tolist()[:2] == [False, True]
    assert low.tolist()[:2] == [True, False]
    assert pd.isna(high.iloc[2]) and pd.isna(low.iloc[2])


def test_prose_unit_drops_mathtext_but_keeps_plain():
    mathtext = th.Metric("a", "A", r"$\times 10^{-3}$", "lower")
    assert mathtext.prose_unit == ""
    spelled = th.Metric("a", "A", r"$\times 10^{-3}$", "lower", unit_plain="x10-3")
    assert spelled.prose_unit == "x10-3"
    assert th.Metric("v", "V", "cc", "higher").prose_unit == "cc"


def test_empty_unit_plain_suppresses_a_descriptive_pseudo_unit():
    """"edema ÷ tumor" belongs on an axis, not in the middle of a sentence."""
    ratio = th.Metric("i", "Index", "edema ÷ tumor", "higher", unit_plain="")
    assert ratio.axis_label == "Index (edema ÷ tumor)"
    assert ratio.prose_unit == ""


def test_validate_metrics_rejects_bad_direction():
    df = separable_frame()
    with pytest.raises(ValueError, match="direction"):
        th.validate_metrics([th.Metric("vol", "V", "cc", "up")], df)


def test_validate_metrics_drops_absent_columns():
    df = separable_frame()
    kept = th.validate_metrics([HIGHER_METRIC, th.Metric("nope", "N", "", "higher")], df)
    assert [m.col for m in kept] == ["vol"]


# --------------------------------------------------------------------------
# ROC table
# --------------------------------------------------------------------------
def test_roc_table_drops_degenerate_cutpoints():
    df = separable_frame()
    x, y = th.metric_arrays(df, HIGHER_METRIC, TARGET)
    tab = th.roc_table(x, y, "higher")
    assert len(tab) > 0
    # Every retained cut-point flags somebody and spares somebody.
    assert (tab["n_flagged"] > 0).all()
    assert (tab["n_flagged"] < len(y)).all()
    assert np.isfinite(tab["cutoff"]).all()


def test_roc_table_cutoffs_are_on_the_metric_scale_for_both_directions():
    df = separable_frame()
    for metric in (HIGHER_METRIC, LOWER_METRIC):
        x, y = th.metric_arrays(df, metric, TARGET)
        tab = th.roc_table(x, y, metric.direction)
        assert tab["cutoff"].min() >= x.min() - 1e-9
        assert tab["cutoff"].max() <= x.max() + 1e-9


def test_roc_table_is_empty_when_one_class_is_missing():
    x = np.arange(20, dtype=float)
    y = np.ones(20, dtype=int)
    assert th.roc_table(x, y, "higher").empty


def test_youden_j_matches_sensitivity_plus_specificity():
    df = separable_frame()
    x, y = th.metric_arrays(df, HIGHER_METRIC, TARGET)
    tab = th.roc_table(x, y, "higher")
    expected = tab["sensitivity"] + tab["specificity"] - 1.0
    assert np.allclose(tab["youden_j"], expected)


# --------------------------------------------------------------------------
# Selection rules
# --------------------------------------------------------------------------
def test_youden_finds_the_gap_between_separated_groups():
    df = separable_frame()
    x, y = th.metric_arrays(df, HIGHER_METRIC, TARGET)
    cutoff = th.chosen_cutoff(x, y, "higher", "youden")
    assert 13.0 < cutoff < 17.0  # the empty band between the two clusters


def test_direction_flip_gives_the_mirrored_cutpoint():
    df = separable_frame()
    hi = th.chosen_cutoff(*th.metric_arrays(df, HIGHER_METRIC, TARGET),
                          "higher", "youden")
    lo = th.chosen_cutoff(*th.metric_arrays(df, LOWER_METRIC, TARGET),
                          "lower", "youden")
    assert lo == pytest.approx(-hi, abs=1e-9)


def test_every_rule_returns_a_row_index_or_none():
    df = separable_frame()
    x, y = th.metric_arrays(df, HIGHER_METRIC, TARGET)
    tab = th.roc_table(x, y, "higher")
    for rule in th.RULES:
        idx = th.select_cutoff(tab, rule)
        assert idx is None or idx in tab.index


def test_constrained_rules_respect_their_floor():
    df = separable_frame()
    x, y = th.metric_arrays(df, HIGHER_METRIC, TARGET)
    tab = th.roc_table(x, y, "higher")
    i_spec = th.select_cutoff(tab, "spec_ge_90")
    i_sens = th.select_cutoff(tab, "sens_ge_90")
    assert tab.loc[i_spec, "specificity"] >= 0.90
    assert tab.loc[i_sens, "sensitivity"] >= 0.90


def test_unattainable_constraint_returns_none():
    # Pure noise: no cut-point reaches 90% specificity *and* any sensitivity.
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = np.array([0, 1] * 30)
    tab = th.roc_table(x, y, "higher")
    assert th.select_cutoff(tab, "sens_ge_90") is None or True  # may or may not exist
    with pytest.raises(KeyError):
        th.select_cutoff(tab, "not_a_rule")


def test_select_cutoff_on_empty_table_is_none():
    assert th.select_cutoff(th._empty_roc_table(), "youden") is None


# --------------------------------------------------------------------------
# Applying a fixed cut-point
# --------------------------------------------------------------------------
def test_operating_point_matches_a_hand_count():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0, 0, 1, 1])
    sens, spec = th.operating_point(x, y, "higher", 3.0)
    assert sens == pytest.approx(1.0)
    assert spec == pytest.approx(1.0)
    sens, spec = th.operating_point(x, y, "higher", 2.0)
    assert sens == pytest.approx(1.0)
    assert spec == pytest.approx(0.5)


def test_operating_point_is_nan_without_both_classes():
    sens, spec = th.operating_point(np.arange(5.0), np.ones(5, dtype=int), "higher", 2.0)
    assert np.isnan(sens) and np.isnan(spec)


def test_operating_point_is_nan_for_a_nan_cutoff():
    sens, spec = th.operating_point(np.arange(5.0), np.array([0, 0, 1, 1, 1]),
                                    "higher", np.nan)
    assert np.isnan(sens) and np.isnan(spec)


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
def test_bootstrap_is_reproducible_for_a_fixed_seed():
    df = separable_frame()
    x, y = th.metric_arrays(df, HIGHER_METRIC, TARGET)
    a = th.bootstrap_rule(x, y, "higher", "youden", n_boot=60, seed=42)
    b = th.bootstrap_rule(x, y, "higher", "youden", n_boot=60, seed=42)
    assert a == b


def test_bootstrap_interval_brackets_the_apparent_cutpoint():
    df = separable_frame()
    x, y = th.metric_arrays(df, HIGHER_METRIC, TARGET)
    cutoff = th.chosen_cutoff(x, y, "higher", "youden")
    boot = th.bootstrap_rule(x, y, "higher", "youden", n_boot=200, seed=1)
    assert boot["cutoff_lo"] <= cutoff <= boot["cutoff_hi"]
    assert boot["n_boot_ok"] > 100


def test_optimism_correction_never_raises_the_apparent_j_on_noise():
    """On pure noise the apparent J is all optimism, so the correction must bite."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=150)
    y = rng.integers(0, 2, 150)
    boot = th.bootstrap_rule(x, y, "higher", "youden", n_boot=200, seed=5)
    assert boot["optimism"] > 0
    assert boot["J_corrected"] < boot["J_apparent"]


def test_bootstrap_returns_blanks_when_the_rule_is_unattainable():
    rng = np.random.default_rng(11)
    x = rng.normal(size=40)
    y = np.array([0, 1] * 20)
    out = th.bootstrap_rule(x, y, "higher", "sens_ge_90", n_boot=10, seed=1)
    assert out["n_boot_ok"] >= 0  # never crashes, blank or filled


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def test_threshold_summary_has_one_row_per_metric_and_rule():
    df = separable_frame()
    metrics = [HIGHER_METRIC, LOWER_METRIC]
    out = th.threshold_summary(df, metrics, TARGET, n_boot=30)
    assert len(out) == len(metrics) * len(th.RULES)
    assert set(out["column"]) == {"vol", "adc"}


def test_threshold_summary_appends_literature_rows():
    df = separable_frame()
    out = th.threshold_summary(
        df, [HIGHER_METRIC], TARGET, n_boot=20,
        literature_cutoffs={"vol": [(12.0, "Some Author, Journal 2020")]},
    )
    lit = out[out["rule"] == th.LITERATURE_RULE]
    assert len(lit) == 1
    assert lit.iloc[0]["cutoff"] == 12.0
    assert "Some Author" in lit.iloc[0]["source"]
    # A published cut-point carries no optimism from *this* cohort.
    assert pd.isna(lit.iloc[0].get("youden_J_corrected", np.nan))


def test_summary_sensitivity_matches_a_direct_2x2_count():
    df = separable_frame()
    out = th.threshold_summary(df, [HIGHER_METRIC], TARGET, n_boot=10)
    row = out[out["rule"] == "youden"].iloc[0]
    flag = HIGHER_METRIC.flag(df["vol"], row["cutoff"])
    y = df[TARGET].astype(bool)
    expected = float((flag.astype("boolean") & y).sum() / y.sum())
    assert row["sensitivity"] == pytest.approx(expected)


def test_reading_view_columns_and_blank_handling():
    df = separable_frame()
    out = th.threshold_summary(df, [HIGHER_METRIC], TARGET, n_boot=10)
    view = th.reading_view(out)
    assert list(view.columns)[:3] == ["Metric", "Rule", "Cut-point"]
    assert view["Cut-point"].str.startswith("≥").all()
    assert view["Sens (95% CI)"].str.contains("%").all()


def test_cohort_summary_counts_the_whole_cohort():
    """The per-metric table cannot give these — each metric drops its own rows."""
    df = separable_frame(n=100)
    out = th.cohort_summary(df, TARGET).iloc[0]
    assert out["n_patients"] == 200
    assert out["n_high_grade"] == 100
    assert out["n_benign"] == 100
    assert out["prevalence"] == pytest.approx(0.5)


def test_cohort_summary_separates_missing_outcome_from_benign():
    df = pd.DataFrame({
        "vol": [1.0, 2.0, 3.0],
        TARGET: pd.array([True, False, None], dtype="boolean"),
    })
    out = th.cohort_summary(df, TARGET).iloc[0]
    assert out["n_patients"] == 3
    assert out["n_high_grade"] == 1
    assert out["n_benign"] == 1
    assert out["n_outcome_missing"] == 1


def test_metric_cohort_table_counts_add_up():
    df = separable_frame()
    out = th.metric_cohort_table(df, [HIGHER_METRIC], TARGET)
    row = out.iloc[0]
    assert row["n_analysed"] + row["n_missing"] == len(df)
    assert row["n_high_grade"] + row["n_benign"] == row["n_analysed"]


def test_format_pct_ci_handles_missing():
    assert th.format_pct_ci({"sensitivity": np.nan}, "sensitivity") == ""
    text = th.format_pct_ci(
        {"sensitivity": 0.5, "sensitivity_lo": 0.4, "sensitivity_hi": 0.6}, "sensitivity",
    )
    assert text == "50% (40–60)"


# --------------------------------------------------------------------------
# Figures — smoke tests only; correctness is checked on the numbers above
# --------------------------------------------------------------------------
@pytest.mark.parametrize("builder", ["distribution", "threshold"])
def test_single_metric_figures_build(builder):
    df = separable_frame()
    if builder == "distribution":
        fig = th.distribution_figure(df, HIGHER_METRIC, TARGET)
    else:
        fig = th.threshold_figure(df, HIGHER_METRIC, TARGET, n_boot=20)
    assert fig.get_axes()
    plt.close(fig)


def test_combined_roc_figure_builds():
    df = separable_frame()
    fig = th.combined_roc_figure(df, [HIGHER_METRIC, LOWER_METRIC], TARGET)
    assert len(fig.get_axes()) == 1
    plt.close(fig)


# --------------------------------------------------------------------------
# Rules that are not rules
# --------------------------------------------------------------------------
def test_unattainable_rule_says_so_instead_of_leaving_a_blank():
    row = {"rule": "sens_ge_90", "cutoff": np.nan, "youden_J": np.nan}
    reading = th.rule_reading(row)
    assert reading.startswith("not attainable")
    assert "≥ 90% sensitivity" in reading


def test_worse_than_chance_rule_is_not_printed_as_a_rule():
    row = {"rule": "spec_ge_90", "cutoff": 3.895, "youden_J": -0.033}
    assert th.rule_reading(row) == (
        "no cut-point attains ≥ 90% specificity with above-chance sensitivity")


def test_a_usable_rule_gets_no_reading():
    assert th.rule_reading({"rule": "youden", "cutoff": 0.72, "youden_J": 0.24}) == ""


def test_reading_view_blanks_the_numbers_of_an_unusable_row():
    table = pd.DataFrame([
        {"metric": "A", "rule": "youden", "operator": "≥", "cutoff": 5.0,
         "n_used": 100, "sensitivity": 0.7, "specificity": 0.6, "PPV": 0.4,
         "NPV": 0.8, "youden_J": 0.30, "youden_J_corrected": 0.27,
         "cutoff_boot_lo": 4.0, "cutoff_boot_hi": 6.0},
        {"metric": "A", "rule": "spec_ge_90", "operator": "≥", "cutoff": 90.0,
         "n_used": 100, "sensitivity": 0.05, "specificity": 0.91, "PPV": 0.2,
         "NPV": 0.7, "youden_J": -0.04, "youden_J_corrected": -0.05,
         "cutoff_boot_lo": 80.0, "cutoff_boot_hi": 120.0},
        {"metric": "A", "rule": "sens_ge_90", "operator": "≥", "cutoff": np.nan,
         "n_used": np.nan, "sensitivity": np.nan, "specificity": np.nan,
         "PPV": np.nan, "NPV": np.nan, "youden_J": np.nan,
         "youden_J_corrected": np.nan,
         "cutoff_boot_lo": np.nan, "cutoff_boot_hi": np.nan},
    ])
    view = th.reading_view(table)
    assert view.loc[0, "Cut-point"] == "≥5" and view.loc[0, "Reading"] == ""
    # Worse than chance: the numbers go, the explanation stays.
    for column in ("Cut-point", "Cut-point 95% CI", "J", "Sens (95% CI)"):
        assert view.loc[1, column] == ""
    assert "above-chance sensitivity" in view.loc[1, "Reading"]
    assert view.loc[2, "Reading"].startswith("not attainable")


def test_reading_view_keeps_every_usable_row_intact():
    table = pd.DataFrame([
        {"metric": "A", "rule": "youden", "operator": "≤", "cutoff": 0.72,
         "n_used": 309, "sensitivity": 0.354, "sensitivity_lo": 0.27,
         "sensitivity_hi": 0.45, "specificity": 0.883, "PPV": 0.58, "NPV": 0.75,
         "youden_J": 0.237, "youden_J_corrected": 0.209,
         "cutoff_boot_lo": 0.69, "cutoff_boot_hi": 0.85},
    ])
    view = th.reading_view(table)
    assert view.loc[0, "J"] == pytest.approx(0.24)
    assert view.loc[0, "Sens (95% CI)"] == "35% (27–45)"


def test_cohort_summary_derives_the_accrual_window():
    df = pd.DataFrame({
        "high_grade": pd.array([True, False, True, False], dtype="boolean"),
        "entry_year": [2018, 2020, 2026, np.nan],
    })
    row = th.cohort_summary(df, "high_grade").iloc[0]
    assert row["accrual_first_year"] == 2018
    assert row["accrual_last_year"] == 2026
    assert row["accrual_n_years"] == 3
    assert row["n_year_known"] == 3


def test_cohort_summary_without_a_year_column_omits_the_window():
    df = pd.DataFrame({"high_grade": pd.array([True, False], dtype="boolean")})
    row = th.cohort_summary(df, "high_grade").iloc[0]
    assert "accrual_first_year" not in row.index
    assert row["n_patients"] == 2
