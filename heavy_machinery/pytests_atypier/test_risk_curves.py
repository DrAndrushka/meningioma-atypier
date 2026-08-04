"""Risk curves: spline basis, non-linearity test, steepest point, crossings.

The tests that matter here are the ones that check the module *refuses* to
report a threshold: on straight-line risk there is no knee to find, and a
module that invents one would put a fabricated number on a poster.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import risk_curves as rc
from thresholds import Metric, metric_arrays

TARGET = "high_grade"
VOL = Metric("vol", "Volume", "cc", "higher")
ADC = Metric("adc", "ADC", "unit", "lower")


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def linear_risk_frame(n: int = 600, seed: int = 1) -> pd.DataFrame:
    """Risk rises as a straight line on the log-odds scale — no knee exists."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 20, n)
    y = rng.binomial(1, logistic(-3.0 + 0.3 * x))
    return pd.DataFrame({"vol": x, TARGET: pd.array(y.astype(bool), dtype="boolean")})


def step_risk_frame(n: int = 1500, seed: int = 2) -> pd.DataFrame:
    """A genuine threshold: 5% risk below x = 10, 70% above, with real overlap.

    Deliberately *not* a steep logistic — a near-perfect separation is fitted
    just as well by a straight log-odds line, so it would leave no curvature
    for the spline to find. A step in the probability itself is what a
    threshold effect actually looks like.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 20, n)
    y = rng.binomial(1, np.where(x < 10.0, 0.05, 0.70))
    return pd.DataFrame({"vol": x, TARGET: pd.array(y.astype(bool), dtype="boolean")})


# --------------------------------------------------------------------------
# Spline basis
# --------------------------------------------------------------------------
def test_rcs_basis_has_k_minus_one_columns():
    x = np.linspace(0, 10, 50)
    for k in (3, 4, 5):
        knots = np.linspace(1, 9, k)
        assert rc.rcs_basis(x, knots).shape == (50, k - 1)


def test_rcs_first_column_is_x_itself():
    """The nested LR test is only valid because dropping the tail columns
    leaves exactly the linear model."""
    x = np.linspace(0, 10, 30)
    basis = rc.rcs_basis(x, np.array([1.0, 5.0, 9.0]))
    assert np.allclose(basis[:, 0], x)


def test_rcs_basis_is_linear_beyond_the_outer_knots():
    """'Restricted' means the fit is a straight line outside the knot range."""
    knots = np.array([2.0, 5.0, 8.0])
    beyond = np.array([9.0, 10.0, 11.0, 12.0])
    basis = rc.rcs_basis(beyond, knots)
    combo = basis @ np.array([0.7, -0.4])  # any coefficients
    second_diff = np.diff(combo, n=2)
    assert np.allclose(second_diff, 0.0, atol=1e-8)


def test_rcs_falls_back_to_linear_with_too_few_knots():
    x = np.linspace(0, 5, 10)
    assert rc.rcs_basis(x, np.array([1.0, 2.0])).shape == (10, 1)


def test_default_knots_reduce_when_values_are_tied():
    """A zero-inflated metric collapses the low quantiles onto one value."""
    x = np.concatenate([np.zeros(60), np.linspace(1, 10, 40)])
    knots = rc.default_knots(x, n_knots=5)
    assert knots.size == np.unique(knots).size
    assert knots.size <= 5


def test_default_knots_are_sorted_and_distinct():
    x = np.linspace(0, 100, 500)
    knots = rc.default_knots(x, n_knots=4)
    assert knots.size == 4
    assert np.all(np.diff(knots) > 0)


# --------------------------------------------------------------------------
# Non-linearity test — the gatekeeper
# --------------------------------------------------------------------------
def test_straight_line_risk_is_not_flagged_as_non_linear():
    df = linear_risk_frame()
    curve = rc.fit_risk_curve(*metric_arrays(df, VOL, TARGET), direction="higher")
    assert curve.spline_fitted
    assert not curve.nonlinear
    assert not curve.knee_found


def test_step_shaped_risk_is_flagged_as_non_linear():
    df = step_risk_frame()
    curve = rc.fit_risk_curve(*metric_arrays(df, VOL, TARGET), direction="higher")
    assert curve.nonlinear
    assert curve.lr_p < 0.01


def test_steepest_point_lands_on_the_true_step():
    df = step_risk_frame()
    curve = rc.fit_risk_curve(*metric_arrays(df, VOL, TARGET), direction="higher")
    assert curve.knee_found
    assert curve.steepest_x == pytest.approx(10.0, abs=2.0)


def test_linear_logodds_still_has_an_interior_slope_peak_but_no_knee():
    """The trap this module exists to avoid.

    A straight log-odds line makes an S-shaped probability curve, so its slope
    peaks in the interior — at the 50% crossing. That is not a threshold
    effect, and ``knee_found`` must not report it as one.
    """
    df = linear_risk_frame()
    curve = rc.fit_risk_curve(*metric_arrays(df, VOL, TARGET), direction="higher")
    assert np.isfinite(curve.steepest_x)
    assert not curve.steepest_at_boundary  # the peak is genuinely interior
    assert not curve.nonlinear             # but there is no curvature
    assert curve.knee_found is False
    # And it sits essentially on top of the 50% crossing, carrying nothing new.
    assert curve.steepest_x == pytest.approx(curve.crossings[0.5], rel=0.15)


def test_interiority_is_judged_in_patients_not_axis_units():
    """A long right tail must not exile a knee sitting in the middle of the data.

    Edema volume runs 0–116 cc with half the cohort below 4.5 cc: a knee at
    3.5 cc is 3% along the axis but at the 48th percentile of patients.
    """
    rng = np.random.default_rng(11)
    # Risk steps at x = 3; values are heavily right-skewed like a volume.
    x = np.concatenate([rng.uniform(0, 6, 700), rng.uniform(6, 300, 120)])
    y = rng.binomial(1, np.where(x < 3.0, 0.08, 0.62))
    curve = rc.fit_risk_curve(x, y, direction="higher")
    assert curve.nonlinear
    # Well inside the patients even though it is a sliver of the axis.
    assert curve.steepest_pct > rc.BOUNDARY_PERCENTILE
    assert (curve.steepest_x - curve.grid[0]) / (curve.grid[-1] - curve.grid[0]) < 0.10
    assert curve.knee_found


def test_boundary_still_fires_when_few_patients_lie_beyond_the_peak():
    df = linear_risk_frame()
    x, y = metric_arrays(df, VOL, TARGET)
    curve = rc.fit_risk_curve(x, y, direction="higher")
    assert 0.0 <= curve.steepest_pct <= 100.0


def test_non_linearity_is_scale_dependent_and_the_summary_says_so():
    """The bug this pair of columns exists to make visible.

    A relationship that is a straight line in log(x) is curved in x. The
    threshold is reported in clinical units, so the primary test runs in those
    units; the log-scale result rides along as a sensitivity analysis.
    """
    rng = np.random.default_rng(12)
    x = rng.lognormal(2.5, 1.0, 900)
    # Linear in log(x): curved on the raw scale, straight on the log scale.
    y = rng.binomial(1, logistic(-4.0 + 1.0 * np.log1p(x)))
    raw = rc.fit_risk_curve(x, y, direction="higher", log_fit=False)
    log = rc.fit_risk_curve(x, y, direction="higher", log_fit=True)
    assert raw.lr_p != pytest.approx(log.lr_p)
    assert not log.nonlinear  # straight by construction on its own scale

    df = pd.DataFrame({"vol": x, TARGET: pd.array(y.astype(bool), dtype="boolean")})
    table, _ = rc.risk_curve_summary(df, [VOL], TARGET, n_boot=20)
    row = table.iloc[0]
    assert row["fitted_on"] == "raw x (clinical units)"
    assert "nonlinearity_p_log_scale" in table.columns
    assert row["scale_sensitive"] == (row["nonlinear"] != row["nonlinear_log_scale"])


def test_log_fit_refuses_values_at_or_below_minus_one():
    """log1p is undefined there; it used to reach numpy and raise LinAlgError."""
    rng = np.random.default_rng(13)
    x = -rng.uniform(1, 40, 400)  # mirrored metric
    y = rng.binomial(1, 0.3, 400)
    curve = rc.fit_risk_curve(x, y, direction="lower", log_fit=True)
    assert not curve.spline_fitted
    assert "log fit" in curve.note


def test_summary_survives_a_metric_the_log_fit_cannot_take():
    """risk_curve_summary always runs the log sensitivity fit — on every metric."""
    df = step_risk_frame()
    df["adc"] = -df["vol"]
    table, curves = rc.risk_curve_summary(df, [ADC], TARGET, n_boot=20)
    assert len(table) == 1
    assert curves["adc"].spline_fitted           # the primary fit is fine
    assert pd.isna(table.iloc[0]["nonlinearity_p_log_scale"])


def test_lr_degrees_of_freedom_match_the_extra_basis_columns():
    df = step_risk_frame()
    curve = rc.fit_risk_curve(*metric_arrays(df, VOL, TARGET),
                              direction="higher", n_knots=4)
    assert curve.lr_df == curve.n_knots - 2


# --------------------------------------------------------------------------
# Direction handling
# --------------------------------------------------------------------------
def test_lower_direction_finds_the_step_too():
    """Mirroring the metric must mirror the answer, not lose it."""
    df = step_risk_frame()
    df["adc"] = -df["vol"]
    curve = rc.fit_risk_curve(*metric_arrays(df, ADC, TARGET), direction="lower")
    assert curve.nonlinear
    assert curve.steepest_x == pytest.approx(-10.0, abs=2.0)


def test_auc_is_oriented_above_half_for_both_directions():
    df = step_risk_frame()
    df["adc"] = -df["vol"]
    up = rc.fit_risk_curve(*metric_arrays(df, VOL, TARGET), direction="higher")
    down = rc.fit_risk_curve(*metric_arrays(df, ADC, TARGET), direction="lower")
    assert up.auc > 0.5 and down.auc > 0.5
    assert up.auc == pytest.approx(down.auc, abs=1e-9)


# --------------------------------------------------------------------------
# Risk crossings
# --------------------------------------------------------------------------
def test_crossing_is_the_boundary_of_the_high_risk_region():
    grid = np.linspace(0, 10, 101)
    risk = grid / 10.0  # rises 0 → 1
    assert rc.risk_crossing(grid, risk, 0.5, "higher") == pytest.approx(5.0, abs=0.1)


def test_crossing_flips_side_with_direction():
    grid = np.linspace(0, 10, 101)
    risk = 1.0 - grid / 10.0  # falls 1 → 0
    assert rc.risk_crossing(grid, risk, 0.5, "lower") == pytest.approx(5.0, abs=0.1)


def test_crossing_is_nan_when_the_level_is_never_reached():
    grid = np.linspace(0, 10, 51)
    risk = np.full_like(grid, 0.2)
    assert np.isnan(rc.risk_crossing(grid, risk, 0.9, "higher"))


def test_crossing_is_nan_when_risk_never_drops_below_the_level():
    grid = np.linspace(0, 10, 51)
    risk = np.full_like(grid, 0.95)
    assert np.isnan(rc.risk_crossing(grid, risk, 0.5, "higher"))


def test_fitted_crossings_are_ordered_with_the_levels():
    df = step_risk_frame()
    curve = rc.fit_risk_curve(*metric_arrays(df, VOL, TARGET),
                              direction="higher", risk_levels=(0.3, 0.5, 0.7))
    values = [curve.crossings[l] for l in (0.3, 0.5, 0.7)]
    assert all(np.isfinite(values))
    assert values[0] < values[1] < values[2]  # higher risk needs a bigger tumor


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------
def test_tiny_sample_returns_a_blank_curve_rather_than_crashing():
    curve = rc.fit_risk_curve(np.arange(10.0), np.array([0, 1] * 5), direction="higher")
    assert not curve.spline_fitted
    assert curve.grid.size == 0
    assert "too few" in curve.note


def test_constant_metric_returns_a_blank_curve():
    x = np.full(200, 3.0)
    y = np.array([0, 1] * 100)
    curve = rc.fit_risk_curve(x, y, direction="higher")
    assert not curve.spline_fitted


def test_zero_inflated_metric_still_produces_a_curve():
    """A quarter of the real cohort has exactly zero edema."""
    rng = np.random.default_rng(4)
    x = np.concatenate([np.zeros(150), rng.uniform(1, 50, 350)])
    y = rng.binomial(1, logistic(-1.5 + 0.04 * x))
    curve = rc.fit_risk_curve(x, y, direction="higher", log_fit=True)
    assert curve.grid.size > 0
    assert np.all(np.isfinite(curve.risk))


def test_log_fit_reports_on_the_original_scale():
    rng = np.random.default_rng(6)
    x = rng.lognormal(2.5, 1.0, 700)
    y = rng.binomial(1, logistic(-4.0 + 1.0 * np.log1p(x)))
    curve = rc.fit_risk_curve(x, y, direction="higher", log_fit=True)
    assert curve.log_fitted
    # The grid is in cc, not log-cc.
    assert curve.grid.min() >= np.quantile(x, 0.02)
    assert curve.grid.max() <= np.quantile(x, 0.98)


# --------------------------------------------------------------------------
# Observed bins
# --------------------------------------------------------------------------
def test_observed_bins_track_a_known_risk_gradient():
    df = step_risk_frame()
    x, y = metric_arrays(df, VOL, TARGET)
    bins = rc.observed_bins(x, y)
    assert len(bins) >= 4
    assert bins["observed"].iloc[0] < bins["observed"].iloc[-1]
    # Wilson bounds are solved numerically, so allow a floating-point hair.
    assert (bins["lo"] <= bins["observed"] + 1e-9).all()
    assert (bins["observed"] <= bins["hi"] + 1e-9).all()


def test_observed_bins_survive_heavy_ties():
    x = np.concatenate([np.zeros(200), np.linspace(1, 10, 200)])
    y = np.concatenate([np.zeros(200, dtype=int), np.ones(200, dtype=int)])
    bins = rc.observed_bins(x, y)
    assert len(bins) >= 1


# --------------------------------------------------------------------------
# Bootstrap and summary
# --------------------------------------------------------------------------
def test_bootstrap_interval_contains_the_point_estimate():
    df = step_risk_frame()
    x, y = metric_arrays(df, VOL, TARGET)
    curve = rc.fit_risk_curve(x, y, direction="higher")
    boot = rc.bootstrap_risk_curve(x, y, direction="higher", n_boot=80, seed=1)
    assert boot["steepest_lo"] <= curve.steepest_x <= boot["steepest_hi"]


def test_bootstrap_is_reproducible():
    df = step_risk_frame()
    x, y = metric_arrays(df, VOL, TARGET)
    kw = dict(direction="higher", n_boot=40, seed=99)
    a = rc.bootstrap_risk_curve(x, y, **kw)
    b = rc.bootstrap_risk_curve(x, y, **kw)
    assert a["steepest_lo"] == b["steepest_lo"]
    assert a["n_boot_ok"] == b["n_boot_ok"]


def test_knee_rate_is_low_when_risk_is_linear():
    """The honest signal that the "šķēre" is not reproducible."""
    df = linear_risk_frame()
    x, y = metric_arrays(df, VOL, TARGET)
    boot = rc.bootstrap_risk_curve(x, y, direction="higher", n_boot=80, seed=2)
    assert boot["knee_rate_boot"] < 0.5


def test_knee_rate_is_high_when_the_threshold_is_real():
    df = step_risk_frame()
    x, y = metric_arrays(df, VOL, TARGET)
    boot = rc.bootstrap_risk_curve(x, y, direction="higher", n_boot=80, seed=2)
    assert boot["knee_rate_boot"] > 0.8


def test_summary_has_one_row_per_metric_with_the_expected_columns():
    df = step_risk_frame()
    table, curves = rc.risk_curve_summary(df, [VOL], TARGET, n_boot=40)
    assert len(table) == 1
    for col in ("nonlinearity_p", "steepest_x", "knee_found", "risk_30_x",
                "risk_50_x", "verdict"):
        assert col in table.columns
    assert set(curves) == {"vol"}


def test_reading_view_says_no_threshold_for_linear_risk():
    df = linear_risk_frame()
    table, _ = rc.risk_curve_summary(df, [VOL], TARGET, n_boot=30)
    view = rc.risk_curve_reading_view(table)
    assert view["Steepest rise"].iloc[0] == "no interior threshold"
    assert "no curvature" in view["Reading"].iloc[0]


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def test_risk_curve_figure_builds():
    df = step_risk_frame()
    _, curves = rc.risk_curve_summary(df, [VOL], TARGET, n_boot=20)
    fig = rc.risk_curve_figure(df, VOL, TARGET, curves["vol"])
    assert len(fig.get_axes()) == 2
    plt.close(fig)


def test_risk_curve_panel_builds_and_blanks_unused_axes():
    df = step_risk_frame()
    df["adc"] = -df["vol"]
    _, curves = rc.risk_curve_summary(df, [VOL, ADC], TARGET, n_boot=20)
    fig = rc.risk_curve_panel(df, [VOL, ADC], TARGET, curves)
    assert len(fig.get_axes()) == 2
    plt.close(fig)


def test_panel_survives_a_blank_curve():
    small = pd.DataFrame({
        "vol": np.arange(10.0),
        TARGET: pd.array([True, False] * 5, dtype="boolean"),
    })
    _, curves = rc.risk_curve_summary(small, [VOL], TARGET, n_boot=5)
    fig = rc.risk_curve_panel(small, [VOL], TARGET, curves)
    plt.close(fig)


# --------------------------------------------------------------------------
# Zero inflation
# --------------------------------------------------------------------------
def zero_inflated_frame(n: int = 400, seed: int = 21) -> pd.DataFrame:
    """40% exactly zero at low risk; among the rest, risk is FLAT in volume.

    The whole-cohort spline should find a bend anyway — the bend is the step
    between absent and present — and the non-zero refit should find nothing.
    That is the failure mode the three-way split exists to expose.
    """
    rng = np.random.default_rng(seed)
    zero = rng.random(n) < 0.40
    x = np.where(zero, 0.0, rng.uniform(0.5, 60.0, n))
    y = rng.binomial(1, np.where(zero, 0.10, 0.45))
    return pd.DataFrame({"vol": x, TARGET: pd.array(y.astype(bool), dtype="boolean")})


def test_zero_share_counts_exact_zeros_and_both_risks():
    df = zero_inflated_frame()
    row = rc.zero_share(df, [VOL], TARGET).iloc[0]
    assert row["n_zero"] == int((df["vol"] == 0).sum())
    assert row["pct_zero"] == pytest.approx(100 * row["n_zero"] / row["n_analysed"])
    assert row["zero_inflated"]
    assert row["risk_positive"] > row["risk_zero"]
    assert row["risk_ratio"] == pytest.approx(row["risk_positive"] / row["risk_zero"])


def test_zero_share_reports_percent_not_a_fraction():
    """The shared CSV formatter rounds pct_* on a 0-100 scale."""
    df = zero_inflated_frame()
    assert rc.zero_share(df, [VOL], TARGET).iloc[0]["pct_zero"] > 1.0


def test_a_metric_with_no_zeros_is_not_flagged():
    df = linear_risk_frame()
    df["vol"] = df["vol"] + 1.0
    assert not rc.zero_share(df, [VOL], TARGET).iloc[0]["zero_inflated"]


def test_presence_rule_skips_a_metric_nobody_has_a_zero_of():
    """An all-positive flag has an empty 2x2 column and no valid chi-square."""
    df = linear_risk_frame()
    df["vol"] = df["vol"] + 1.0
    assert rc.presence_rules(df, [VOL], TARGET).empty


def test_presence_rule_scores_present_versus_absent():
    df = zero_inflated_frame()
    row = rc.presence_rules(df, [VOL], TARGET).iloc[0]
    assert row["TP"] + row["FN"] == int(df[TARGET].astype("boolean").sum())
    assert row["youden_J"] == pytest.approx(row["sensitivity"] + row["specificity"] - 1)


def test_curvature_from_the_zeros_alone_does_not_survive_the_refit():
    """The point of P1.3, as a regression test on synthetic data."""
    df = zero_inflated_frame()
    whole, _ = rc.risk_curve_summary(df, [VOL], TARGET, n_boot=60, seed=3)
    subset = rc.positive_only(df, VOL)
    nonzero, _ = rc.risk_curve_summary(subset, [VOL], TARGET, n_boot=60, seed=3)

    assert whole.iloc[0]["nonlinearity_p"] < 0.05      # the whole-cohort "threshold"
    assert nonzero.iloc[0]["nonlinearity_p"] > 0.05    # nothing left once zeros go
    assert len(subset) == int((df["vol"] > 0).sum())


def test_zero_inflation_comparison_has_one_row_per_fit():
    df = zero_inflated_frame()
    whole, _ = rc.risk_curve_summary(df, [VOL], TARGET, n_boot=20, seed=1)
    nonzero, _ = rc.risk_curve_summary(rc.positive_only(df, VOL), [VOL], TARGET,
                                       n_boot=20, seed=1)
    table = rc.zero_inflation_comparison(
        whole, rc.presence_rules(df, [VOL], TARGET), nonzero, VOL)
    assert len(table) == 3
    assert list(table["Fitted on"])[1].startswith("Present vs absent")
