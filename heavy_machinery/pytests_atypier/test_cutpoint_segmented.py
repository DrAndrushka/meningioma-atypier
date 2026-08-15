"""Segmented regression — the breakpoint, its interval, and Davies' correction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import segmented as sg


def _broken(n: int = 600, psi: float = 5.0, seed: int = 0):
    """Risk flat below the break, rising steeply above it."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    logit = np.where(x < psi, -2.0, -2.0 + 1.6 * (x - psi))
    return x, rng.binomial(1, 1 / (1 + np.exp(-logit)))


def _straight(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    return x, rng.binomial(1, 1 / (1 + np.exp(-(-3.0 + 0.5 * x))))


# --- the candidate grid ----------------------------------------------------
def test_candidates_stay_inside_the_data():
    """A break in the outermost few percent is a tail artefact, not a threshold."""
    x = np.linspace(0, 100, 500)
    grid = sg.candidate_breakpoints(x)
    assert grid.min() >= np.quantile(x, 0.10) - 1e-9
    assert grid.max() <= np.quantile(x, 0.90) + 1e-9


def test_a_constant_predictor_yields_no_candidates():
    assert sg.candidate_breakpoints(np.ones(50)).size == 0


# --- estimating the breakpoint --------------------------------------------
def test_a_real_break_is_recovered_near_where_it_was_put():
    x, y = _broken(psi=5.0)
    fit = sg.fit_segmented(x, y)
    assert abs(fit.breakpoint - 5.0) < 1.5


def test_the_two_slopes_differ_across_a_real_break():
    x, y = _broken(psi=5.0)
    fit = sg.fit_segmented(x, y)
    assert fit.slope_above > fit.slope_below
    assert fit.slope_change == pytest.approx(fit.slope_above - fit.slope_below)


def test_the_profile_interval_brackets_the_estimate():
    x, y = _broken()
    fit = sg.fit_segmented(x, y)
    assert fit.ci_lo <= fit.breakpoint <= fit.ci_hi


def test_a_stronger_break_gives_a_tighter_interval():
    weak = sg.fit_segmented(*_broken(n=600, seed=1))
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, 600)
    logit = np.where(x < 5, -3.0, -3.0 + 4.0 * (x - 5))
    strong = sg.fit_segmented(x, rng.binomial(1, 1 / (1 + np.exp(-logit))))
    assert strong.ci_width < weak.ci_width


def test_the_profile_is_kept_so_the_surface_can_be_inspected():
    """A grid sees local maxima that an iterative fit would settle into."""
    x, y = _broken()
    fit = sg.fit_segmented(x, y)
    assert fit.grid.size == fit.profile_llf.size > 0
    assert np.isfinite(fit.profile_llf).any()


def test_the_best_candidate_is_the_one_with_the_highest_likelihood():
    x, y = _broken()
    fit = sg.fit_segmented(x, y)
    best = fit.grid[int(np.nanargmax(np.where(np.isfinite(fit.profile_llf),
                                              fit.profile_llf, -np.inf)))]
    assert fit.breakpoint == pytest.approx(best)


# --- Davies' correction ----------------------------------------------------
def test_davies_is_more_conservative_than_the_uncorrected_tail():
    """It charges for having searched every candidate for the best one."""
    z = np.array([0.2, 1.1, 2.4, 3.1, 2.0, 0.7])
    from scipy.stats import norm
    assert sg.davies_p(z) > 2 * norm.sf(np.max(np.abs(z)))


def test_a_jagged_profile_is_penalised_more_than_a_smooth_one():
    """A maximum found by luck should cost more than one that was always there."""
    smooth = np.linspace(0.1, 3.0, 30)
    jagged = np.abs(np.sin(np.linspace(0, 30, 30))) * 3.0
    jagged[15] = 3.0
    assert sg.davies_p(jagged) > sg.davies_p(smooth)


def test_davies_never_exceeds_one():
    assert sg.davies_p(np.array([0.1, 0.2, 0.1, 0.3] * 20)) <= 1.0


def test_davies_needs_more_than_one_candidate():
    assert np.isnan(sg.davies_p(np.array([2.0])))


def test_a_straight_line_does_not_yield_a_supported_breakpoint():
    """The failure mode the correction exists to prevent."""
    x, y = _straight()
    assert not sg.fit_segmented(x, y).supported


def test_a_real_break_survives_the_correction():
    x, y = _broken()
    fit = sg.fit_segmented(x, y)
    assert fit.supported and fit.davies_p < 0.05


def test_support_is_judged_on_davies_not_the_naive_p():
    x, y = _broken()
    fit = sg.fit_segmented(x, y)
    fit_naive_only = sg.SegmentedFit(**{**fit.__dict__, "davies_p": 0.40})
    assert not fit_naive_only.supported


# --- AIC -------------------------------------------------------------------
def test_delta_aic_charges_for_both_extra_parameters():
    """The slope change and the breakpoint itself — two, not one."""
    x, y = _broken()
    fit = sg.fit_segmented(x, y)
    assert fit.delta_aic == pytest.approx(4.0 - fit.lr_stat)


def test_a_real_break_gives_a_negative_delta_aic():
    assert sg.fit_segmented(*_broken()).delta_aic < 0


def test_a_straight_line_gives_a_positive_delta_aic():
    assert sg.fit_segmented(*_straight()).delta_aic > 0


# --- graceful failure ------------------------------------------------------
def test_too_few_patients_is_blank_rather_than_fatal():
    fit = sg.fit_segmented(np.linspace(0, 1, 20), np.array([0, 1] * 10))
    assert np.isnan(fit.breakpoint) and "too few patients" in fit.note


def test_too_few_events_is_blank():
    y = np.zeros(200, dtype=int)
    y[:4] = 1
    fit = sg.fit_segmented(np.linspace(0, 10, 200), y)
    assert "too few patients or events" in fit.note


def test_a_blank_fit_is_not_reported_as_supported():
    assert not sg.fit_segmented(np.linspace(0, 1, 20),
                                np.array([0, 1] * 10)).supported


def test_every_candidate_keeps_enough_patients_on_both_sides():
    """Otherwise the 'break' is fitted to a handful of tail patients."""
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.uniform(0, 1, 300), rng.uniform(50, 60, 25)])
    y = rng.binomial(1, 0.3, 325)
    fit = sg.fit_segmented(x, y)
    if np.isfinite(fit.breakpoint):
        assert min(int(np.sum(x < fit.breakpoint)),
                   int(np.sum(x >= fit.breakpoint))) >= sg.MIN_PER_SIDE


# --- the table -------------------------------------------------------------
def _cohort(seed: int = 4, n: int = 340) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x, y = _broken(n, 5.0, seed)
    edema = np.where(rng.random(n) < 0.35, 0.0, rng.gamma(2, 5, n) + y)
    return pd.DataFrame({
        "high_grade": y,
        "adc_value": rng.normal(0.9 - 0.12 * y, 0.15),
        "tumor_volume": x,
        "edema_volume_cm3": edema,
        "edema_index": edema / (rng.gamma(2, 8, n) + 1),
        "max_diameter_cm": rng.normal(3.8 + 0.4 * y, 1.2),
    })


def _eligible(df):
    import bend_location as bl
    import eligibility as el
    import nonlinearity as nl
    import separation as sep
    fits = nl.fit_all(df)
    return el.eligible(el.carry_forward(sep.separation_table(df),
                                        bl.bend_table(df, fits=fits)))


def test_the_table_reports_both_p_values_side_by_side():
    df = _cohort()
    table = sg.segmented_table(df, _eligible(df))
    assert {"lr_p_naive", "davies_p"} <= set(table.columns)


def test_the_table_rounds_the_breakpoint_to_the_measurements_precision():
    df = _cohort()
    table = sg.segmented_table(df, _eligible(df)).set_index("col")
    if np.isfinite(table.loc["adc_value", "breakpoint"]):
        assert table.loc["adc_value", "breakpoint"] == round(
            table.loc["adc_value", "breakpoint"], 2)


def test_describe_says_plainly_when_nothing_survives():
    df = _cohort()
    table = sg.segmented_table(df, _eligible(df))
    table["breakpoint_supported"] = False
    assert "No breakpoint survives" in sg.describe_segmented(table)


# --- against the real cohort ----------------------------------------------
def test_the_real_cohort_supports_a_breakpoint_for_adc(real_cohort):
    table = sg.segmented_table(real_cohort,
                               _eligible(real_cohort)).set_index("col")
    row = table.loc["adc_value"]
    assert row["breakpoint_supported"]
    assert row["davies_p"] < 0.05
    assert row["ci_lo"] < row["breakpoint"] < row["ci_hi"]


def test_the_published_adc_cutpoint_lies_inside_the_breakpoint_interval(real_cohort):
    """0.72 was derived by Youden; the breakpoint is a different estimate."""
    table = sg.segmented_table(real_cohort,
                               _eligible(real_cohort)).set_index("col")
    row = table.loc["adc_value"]
    assert row["ci_lo"] <= 0.72 <= row["ci_hi"]


def test_max_diameter_gets_no_breakpoint(real_cohort):
    """Consistent with the spline: risk climbs steadily, with nothing to break."""
    table = sg.segmented_table(real_cohort,
                               _eligible(real_cohort)).set_index("col")
    assert not table.loc["max_diameter_cm", "breakpoint_supported"]
    assert table.loc["max_diameter_cm", "delta_aic"] > 0
