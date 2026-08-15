"""Step 5 — the knee, the risk crossings, and the guards on quoting either."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import bend_location as bl
import measurements as ms
import nonlinearity as nl


# --- risk crossings --------------------------------------------------------
def test_crossing_interpolates_between_grid_points():
    grid = np.array([0.0, 1.0, 2.0])
    risk = np.array([0.20, 0.40, 0.60])
    assert bl.risk_crossing(grid, risk, 0.30, ms.HIGHER) == pytest.approx(0.5)


def test_crossing_takes_the_outer_edge_for_an_upward_measurement():
    """A curve that dips back below the level must not report the dip."""
    grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    risk = np.array([0.10, 0.40, 0.20, 0.45, 0.50])
    assert bl.risk_crossing(grid, risk, 0.30, ms.HIGHER) == pytest.approx(2 / 3)


def test_crossing_takes_the_other_edge_for_a_downward_measurement():
    grid = np.array([0.0, 1.0, 2.0])
    risk = np.array([0.60, 0.40, 0.20])
    assert bl.risk_crossing(grid, risk, 0.30, ms.LOWER) == pytest.approx(1.5)


def test_a_level_never_reached_is_blank():
    grid, risk = np.array([0.0, 1.0]), np.array([0.10, 0.20])
    assert np.isnan(bl.risk_crossing(grid, risk, 0.50, ms.HIGHER))


def test_a_level_always_exceeded_is_blank():
    """Everywhere above the line means the line was never crossed."""
    grid, risk = np.array([0.0, 1.0]), np.array([0.70, 0.80])
    assert np.isnan(bl.risk_crossing(grid, risk, 0.50, ms.HIGHER))


def test_an_empty_curve_is_blank_not_an_error():
    assert np.isnan(bl.risk_crossing(np.array([]), np.array([]), 0.3, ms.HIGHER))


# --- the knee --------------------------------------------------------------
def _bent(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    logit = np.where(x < 5, -2.0, -2.0 + 1.4 * (x - 5))
    return x, rng.binomial(1, 1 / (1 + np.exp(-logit)))


def test_the_knee_lands_where_the_curve_actually_turns():
    x, y = _bent()
    knee = bl.steepest_point(nl.fit_spline(x, y), ms.HIGHER, x)["knee"]
    assert 4.0 < knee < 8.0


def test_interiority_is_judged_in_patients_not_axis_units():
    """Half this cohort sits below 4.5 cm3 while the axis runs to 197."""
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.uniform(0, 5, 300), rng.uniform(50, 200, 20)])
    y = rng.binomial(1, np.clip(x / 300 + 0.2, 0, 1))
    out = bl.steepest_point(nl.fit_spline(x, y), ms.HIGHER, x)
    if np.isfinite(out["knee"]) and out["knee"] < 5:
        assert out["knee_percentile"] > bl.BOUNDARY_PERCENTILE


def test_a_knee_at_the_very_edge_is_flagged():
    grid = np.linspace(0, 10, 50)
    fit = nl.SplineFit(
        column="x", stratum="all", n=100, events=30, log_fitted=False,
        knots=np.array([]), n_knots=0, spline_fitted=True,
        lr_stat=9.0, lr_df=1, lr_p=0.003,
        grid=grid, risk=np.exp(-grid), risk_lo=np.exp(-grid),
        risk_hi=np.exp(-grid), prevalence=0.3, note="")
    out = bl.steepest_point(fit, ms.LOWER, np.linspace(0, 10, 100))
    assert out["knee_at_boundary"]


def test_a_blank_fit_yields_a_blank_knee():
    fit = nl.fit_spline(np.arange(10.0), np.array([0, 1] * 5))
    out = bl.steepest_point(fit, ms.HIGHER, np.arange(10.0))
    assert np.isnan(out["knee"]) and out["knee_at_boundary"]


# --- the observed dots -----------------------------------------------------
def test_observed_bins_use_equal_counts_not_equal_width():
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.uniform(0, 1, 200), rng.uniform(50, 60, 40)])
    y = rng.binomial(1, 0.3, 240)
    bins = bl.observed_bins(x, y)
    assert bins["n"].min() >= 10
    assert bins["n"].max() - bins["n"].min() <= bins["n"].mean()


def test_observed_bins_carry_wilson_intervals():
    x, y = _bent()
    bins = bl.observed_bins(x, y)
    assert ((bins["lo"] <= bins["observed"]) & (bins["observed"] <= bins["hi"])).all()


def test_too_few_patients_gives_no_bins_rather_than_junk():
    assert bl.observed_bins(np.arange(8.0), np.array([0, 1] * 4)).empty


# --- parity with the implementation this replaces -------------------------
# Frozen from threshold_phase.risk_curves on the real cohort, verified equal
# before that module was retired. The first four are the knee locations the
# Methods reproducibility sentence quotes.
_FROZEN = {
    "adc_value": {"knee": 0.6619, "r30": 0.7913, "r50": 0.6631},
    "tumor_volume": {"knee": 12.7176, "r30": 18.5193, "r50": 62.2617},
    "edema_volume_cm3": {"knee": 3.5126, "r30": 9.9225, "r50": np.nan},
    "edema_index": {"knee": 0.0915, "r30": 0.3833, "r50": np.nan},
    "max_diameter_cm": {"knee": 3.5578, "r30": 4.0418, "r50": 6.8187},
}


@pytest.mark.parametrize("col", sorted(_FROZEN))
def test_reproduces_the_retired_knee_and_crossings(col, real_cohort):
    m = ms.MEASUREMENTS_BY_COL[col]
    x = pd.to_numeric(real_cohort[col], errors="coerce").to_numpy()
    y = pd.to_numeric(real_cohort["high_grade"], errors="coerce").to_numpy()
    fit = nl.fit_spline(x, y, column=col, log_fit=False)
    expected = _FROZEN[col]
    assert bl.steepest_point(fit, m.direction, x)["knee"] == pytest.approx(
        expected["knee"], abs=5e-4)
    for level, key in ((0.30, "r30"), (0.50, "r50")):
        got = bl.risk_crossing(fit.grid, fit.risk, level, m.direction)
        if np.isnan(expected[key]):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(expected[key], abs=5e-4)


# --- the table -------------------------------------------------------------
def _cohort(n: int = 320, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x, y = _bent(n, seed)
    edema = np.where(rng.random(n) < 0.4, 0.0, rng.gamma(2, 5, n) + y)
    return pd.DataFrame({
        "high_grade": y,
        "adc_value": rng.normal(0.9 - 0.1 * y, 0.15),
        "tumor_volume": x,
        "edema_volume_cm3": edema,
        "edema_index": edema / (rng.gamma(2, 8, n) + 1),
        "max_diameter_cm": rng.normal(3.8 + 0.3 * y, 1.2),
    })


def test_table_has_a_row_per_measurement_per_stratum():
    assert len(bl.bend_table(_cohort())) == 7


def test_a_knee_is_only_quotable_when_the_bend_was_real():
    table = bl.bend_table(_cohort())
    assert not table.loc[~table["bend_is_real"], "quotable"].any()


def test_a_boundary_knee_is_never_quotable():
    table = bl.bend_table(_cohort())
    assert not table.loc[table["knee_at_boundary"], "quotable"].any()


def test_step_four_verdict_is_carried_forward_not_recomputed():
    df = _cohort()
    fits = nl.fit_all(df)
    bend = bl.bend_table(df, fits=fits).set_index("measurement")
    nonlin = nl.nonlinearity_table(df, fits=fits).set_index("measurement")
    assert (bend["bend_is_real"] == nonlin["bent_clinical"]).all()


def test_both_risk_levels_get_a_column():
    table = bl.bend_table(_cohort())
    assert {"risk_30", "risk_50"} <= set(table.columns)


# --- the summary line ------------------------------------------------------
def test_describe_says_plainly_when_nothing_may_be_quoted():
    table = bl.bend_table(_cohort())
    table["quotable"] = False
    assert bl.describe_bend(table).startswith("No knee can be quoted")


def test_describe_marks_a_scale_dependent_knee():
    table = bl.bend_table(_cohort())
    table["quotable"] = True
    table["scales_agree"] = False
    assert "(scale-dependent)" in bl.describe_bend(table)


def test_describe_counts_what_it_withheld():
    table = bl.bend_table(_cohort())
    if (~table["quotable"]).any() and table["quotable"].any():
        assert "withheld" in bl.describe_bend(table)
