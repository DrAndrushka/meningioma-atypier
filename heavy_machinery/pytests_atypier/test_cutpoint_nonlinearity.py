"""Step 4 — the spline basis, the likelihood-ratio test, and the scale check."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import measurements as ms
import nonlinearity as nl


# --- knots -----------------------------------------------------------------
def test_three_knots_are_the_default():
    assert nl.DEFAULT_N_KNOTS == 3


def test_knots_sit_at_harrells_quantiles():
    x = np.arange(1001, dtype=float)
    knots = nl.default_knots(x, 3)
    assert knots.size == 3
    assert knots == pytest.approx(np.quantile(x, (0.10, 0.50, 0.90)))


def test_knots_reduce_rather_than_collapse_on_a_zero_pile():
    """A third of this cohort sits at zero, which ties the lower quantiles."""
    x = np.concatenate([np.zeros(60), np.arange(1, 41, dtype=float)])
    assert nl.default_knots(x, 3).size < 3


def test_knots_give_up_cleanly_on_a_constant():
    assert nl.default_knots(np.ones(50), 3).size == 0


# --- the basis -------------------------------------------------------------
def test_basis_first_column_is_the_variable_itself():
    """This nesting is what makes the likelihood-ratio test valid."""
    x = np.linspace(0, 10, 50)
    basis = nl.rcs_basis(x, nl.default_knots(x, 3))
    assert basis[:, 0] == pytest.approx(x)


def test_basis_has_one_column_fewer_than_knots():
    x = np.linspace(0, 10, 60)
    assert nl.rcs_basis(x, nl.default_knots(x, 3)).shape[1] == 2


def test_basis_falls_back_to_linear_without_enough_knots():
    x = np.linspace(0, 10, 30)
    assert nl.rcs_basis(x, np.array([1.0, 2.0])).shape[1] == 1


def test_basis_is_linear_beyond_the_outer_knots():
    """Restricted means the tails cannot flap — that is the whole restriction."""
    x = np.linspace(0, 100, 400)
    knots = nl.default_knots(x, 3)
    basis = nl.rcs_basis(x, knots)
    tail = x > knots[-1]
    combined = basis[tail, 0] + basis[tail, 1]
    assert np.ptp(np.diff(np.diff(combined))) == pytest.approx(0.0, abs=1e-8)


# --- the test itself -------------------------------------------------------
def _bent(n: int = 400, seed: int = 0):
    """Risk flat then rising — a real bend."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    logit = np.where(x < 5, -2.0, -2.0 + 1.4 * (x - 5))
    return x, rng.binomial(1, 1 / (1 + np.exp(-logit)))


def _straight(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    return x, rng.binomial(1, 1 / (1 + np.exp(-(-3.0 + 0.5 * x))))


def test_a_real_bend_is_detected():
    x, y = _bent()
    assert nl.fit_spline(x, y).bent


def test_a_straight_climb_is_not_called_bent():
    x, y = _straight()
    assert not nl.fit_spline(x, y).bent


def test_the_test_has_the_expected_degrees_of_freedom():
    x, y = _bent()
    assert nl.fit_spline(x, y).lr_df == 1     # 3 knots -> 2 columns -> 1 extra


def test_the_fitted_curve_is_returned_on_the_original_scale():
    x, y = _bent()
    fit = nl.fit_spline(x, y, log_fit=True)
    assert fit.log_fitted
    assert fit.grid.min() >= x.min() and fit.grid.max() <= x.max()


def test_the_band_brackets_the_curve():
    x, y = _bent()
    fit = nl.fit_spline(x, y)
    assert (fit.risk_lo <= fit.risk).all() and (fit.risk <= fit.risk_hi).all()


def test_the_grid_is_trimmed_to_where_the_patients_are():
    x, y = _bent()
    fit = nl.fit_spline(x, y)
    assert fit.grid.min() >= np.quantile(x, 0.025) - 1e-9
    assert fit.grid.max() <= np.quantile(x, 0.975) + 1e-9


# --- graceful failure ------------------------------------------------------
def test_too_few_patients_is_blank_not_fatal():
    fit = nl.fit_spline(np.arange(10.0), np.array([0, 1] * 5))
    assert not fit.spline_fitted
    assert "too few patients" in fit.note
    assert np.isnan(fit.lr_p)


def test_too_few_events_is_blank_not_fatal():
    y = np.zeros(60, dtype=int)
    y[:2] = 1
    fit = nl.fit_spline(np.linspace(0, 10, 60), y)
    assert "too few patients or events" in fit.note


def test_a_blank_fit_is_not_reported_as_bent():
    assert not nl.fit_spline(np.arange(10.0), np.array([0, 1] * 5)).bent


# --- parity with the implementation this replaces -------------------------
# Frozen from the previous threshold_phase.risk_curves fit on the real cohort,
# verified equal to four decimals before that module was retired. These are the
# numbers the Methods reproducibility sentence refers to; if one moves, the
# claim in the manuscript has stopped being true.
_FROZEN_LR_P = {
    ("adc_value", False): 0.0093, ("adc_value", True): 0.0153,
    ("tumor_volume", False): 0.0210, ("tumor_volume", True): 0.9738,
    ("edema_volume_cm3", False): 0.0074, ("edema_volume_cm3", True): 0.1587,
    ("edema_index", False): 0.0473, ("edema_index", True): 0.0019,
    ("max_diameter_cm", False): 0.5028, ("max_diameter_cm", True): 0.7730,
}


@pytest.mark.parametrize("key,expected", sorted(_FROZEN_LR_P.items()))
def test_reproduces_the_retired_implementation(key, expected, real_cohort):
    col, log_fit = key
    x = pd.to_numeric(real_cohort[col], errors="coerce").to_numpy()
    y = pd.to_numeric(real_cohort["high_grade"], errors="coerce").to_numpy()
    fit = nl.fit_spline(x, y, column=col, log_fit=log_fit)
    assert fit.lr_p == pytest.approx(expected, abs=5e-5)


# --- the table -------------------------------------------------------------
def _cohort(n: int = 300, seed: int = 4) -> pd.DataFrame:
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
    assert len(nl.nonlinearity_table(_cohort())) == 7


def test_table_reports_both_scales_for_every_row():
    table = nl.nonlinearity_table(_cohort())
    assert {"lr_p", "lr_p_log", "bent_clinical", "bent_log",
            "scales_agree"} <= set(table.columns)


def test_scales_agree_is_false_when_the_two_fits_disagree():
    table = nl.nonlinearity_table(_cohort())
    mismatched = table["bent_clinical"] != table["bent_log"]
    assert (~table.loc[mismatched, "scales_agree"]).all()


def test_fit_all_covers_both_scales_for_every_stratum():
    fits = nl.fit_all(_cohort())
    assert len(fits) == 14                      # 7 rows x 2 scales
    assert {log for _, _, log in fits} == {False, True}


# --- the summary line ------------------------------------------------------
def test_describe_names_what_bent():
    line = nl.describe_nonlinearity(nl.nonlinearity_table(_cohort()))
    assert "Bend detected" in line or "No measurement shows a bend" in line


def test_describe_flags_scale_dependence_when_present():
    table = nl.nonlinearity_table(_cohort())
    table.loc[0, ["bent_clinical", "bent_log", "scales_agree"]] = [True, False, False]
    assert "no claim about a bend is scale-free" in nl.describe_nonlinearity(table)


def test_describe_says_both_scales_agree_when_they_do():
    table = nl.nonlinearity_table(_cohort())
    table["scales_agree"] = True
    assert "Both scales agree everywhere." in nl.describe_nonlinearity(table)
